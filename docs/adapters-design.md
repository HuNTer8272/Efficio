# Design: Hugging Face Trainer and PyTorch Lightning adapters

Status: **design for review**. Nothing here is implemented yet; `efficio/adapters/hf_trainer.py`
and `efficio/adapters/lightning.py` are still planned stubs. Target release: 0.3.0.

## Goal

Someone who trains with `transformers.Trainer` or a Lightning `Trainer` should get Efficio's
budget-aware stop and save, checkpoint modes (exact / light / smallest), trainable-only
checkpoints, storage sync (Hugging Face Hub, Google Drive) and cross-platform resume, without
rewriting their training loop.

The main difference from `efficio.Run` is that **the framework owns the loop**. Efficio can only
react in the framework's callback hooks, and it can only *ask* the trainer to stop. It cannot
stop the trainer at a point Efficio chooses.

## 1. Public API

Names follow the style in `efficio/__init__.py`: CamelCase nouns, with acronyms in capitals
(`GPULoader`). The classes live in `efficio.adapters` and are **not** exported from the top-level
`efficio` package, so `import efficio` never imports `transformers` or `lightning`.
`efficio/adapters/__init__.py` imports each submodule lazily, the first time its class is
accessed.

| Name | Module | Needs |
|---|---|---|
| `efficio.adapters.HFTrainerCallback` | `adapters/hf_trainer.py` | `pip install efficio[hf]` (transformers) |
| `efficio.adapters.LightningCallback` | `adapters/lightning.py` | `pip install efficio[lightning]` (lightning) |

The constructor arguments mirror `Run` wherever they mean the same thing: `budget`,
`checkpoint_dir`, `save_every_minutes`, `resume`, `checkpoint_mode`, `save_trainable_only`,
`async_save`, `storage`, `keep_remote`, `overwrite_storage`, `upload_timeout`,
`shutdown_when_done`, `verbose`.

These `Run` arguments are left out, because the framework owns what they control:
- `grad_accum_steps`: read from the trainer instead.
- `optimize_data`: the framework builds the DataLoader.
- `channels_last`: the framework owns the model's placement.
- `autocast()`: precision is the trainer's setting. See question Q3.

### Hugging Face Trainer

```python
from transformers import Trainer, TrainingArguments
from efficio.adapters import HFTrainerCallback

efficio_cb = HFTrainerCallback(
    budget="2h", storage="hf://user/repo"
)  # create first: the budget clock starts here
args = TrainingArguments(output_dir="out", num_train_epochs=3, fp16=True, save_strategy="no")
trainer = Trainer(model=model, args=args, train_dataset=train_ds, callbacks=[efficio_cb])
trainer.train(resume_from_checkpoint=efficio_cb.resume_from(trainer))  # None on a fresh start
efficio_cb.report()
```

### PyTorch Lightning

```python
import lightning.pytorch as pl
from efficio.adapters import LightningCallback

efficio_cb = LightningCallback(
    budget="2h", storage="hf://user/repo"
)  # create first: the budget clock starts here
trainer = pl.Trainer(
    max_epochs=20,
    precision="16-mixed",
    callbacks=[efficio_cb],
    plugins=[efficio_cb.checkpoint_io],
    enable_checkpointing=False,
)
trainer.fit(
    model, train_loader, ckpt_path=efficio_cb.resume_from(trainer, model)
)  # None on a fresh start
efficio_cb.report()
```

`resume_from(...)` has to be called even on a fresh start. It is the only point where the
adapter gets a reference to the trainer: HF callbacks never receive the trainer object, and the
adapter needs it to reach the GradScaler and to materialize a resume checkpoint (section 5). It
also checks the storage owner before anything is written there, as `Run.prepare()` does today.

The budget clock starts when the callback is created, as with `Run(...)`. Model download and
tokenization count against the session quota, so the docs will tell users to create the callback
first.

## 2. Hooks used

### Hugging Face `TrainerCallback`

| Hook | Use |
|---|---|
| `on_init_end` | Check the setup and warn once (see the list below). Only runs if the callback was passed to `Trainer(...)`; `trainer.add_callback` after init skips it, so `on_train_begin` repeats the checks. |
| `on_train_begin` | Install the SIGTERM flag handler. Unwrap `model` / `optimizer` (see E3) and check the run fingerprint against the resumed checkpoint (the optimizer type is known only now). Start the GPU utilization sampler. |
| `on_step_end` | Called once per **optimizer** step, after gradient accumulation. Update the TrainState, check `remote.check()` and `ckpt.check()`, then decide: budget nearly used or SIGTERM → blocking save + upload, then `control.should_training_stop = True`. Periodic interval reached → background save. |
| `on_save` | Not used to save. It only detects that HF's own checkpointing is on, and warns once (see Q1). |
| `on_train_end` | Final blocking save + upload if training finished normally. Restore the SIGTERM handler. Stop the sampler. |

`on_init_end` warns once about:
- HF checkpointing still being on (`save_strategy != "no"`, see Q1);
- `load_best_model_at_end` (E5);
- fp32 on a GPU, or bf16 on a GPU with compute capability below 8.0 (Q3).

### Lightning `Callback`

| Hook | Use |
|---|---|
| `on_train_start` | The same setup work as HF `on_train_begin`: signal flag, fingerprint check, sampler. |
| `on_train_batch_end` | Called once per **micro-batch**. Act only at an accumulation boundary, when `(batch_idx + 1) % trainer.accumulate_grad_batches == 0` or on the last batch. Saving mid-accumulation would drop the partial gradients. Then the same decision as HF `on_step_end`. The save goes through `trainer.save_checkpoint(path)`, as Lightning's own `ModelCheckpoint(every_n_train_steps=...)` does at this point. The stop request is `trainer.should_stop = True`. |
| `on_save_checkpoint` | Adds an `"efficio"` entry to Lightning's checkpoint dict: fingerprint, Efficio version, mode. A Lightning checkpoint written by the user's own `ModelCheckpoint` can then be identified too. |
| `on_load_checkpoint` | Reads that entry back for the log line ("resumed from step N") and the report. |
| `on_train_end` | Final save + upload, restore the signal handler, stop the sampler. |

## 3. Budget stop without owning the loop

`Run` checks the budget after every optimizer step and stops *inside* `run.loop()`. The
adapters can only check at a hook and **ask** the trainer to stop:

- **HF:** `control.should_training_stop = True` from `on_step_end`. The trainer leaves the
  epoch loop after this step.
- **Lightning:** `trainer.should_stop = True` from `on_train_batch_end`. The fit loop ends after
  this batch. Lightning ignores the request until `min_epochs` / `min_steps` are reached (E6).

What stays exact and what becomes approximate:

- **The saved state is exact.** Efficio writes the checkpoint synchronously inside the hook,
  *before* it asks the trainer to stop. The hook runs at an optimizer-step boundary, so the
  saved model, optimizer, scheduler, scaler and step counter are consistent.
- **When the stop happens is approximate, not exact to the step as in `Run`:**
  - **No mid-step stop.** The budget is checked only once per optimizer step. One long step
    (big gradient accumulation, a very large batch) can run past the safety margin.
  - **Work after the request is not covered.** After the request, the framework can still do
    things Efficio does not control and cannot shorten:
    - an evaluation scheduled at that step;
    - logging and reporting integrations;
    - HF `load_best_model_at_end`;
    - Lightning's end-of-epoch validation.
  - **SIGTERM saves later.** A SIGTERM only sets a flag. The save happens at the next hook
    call, the same as `Run` today. But a hook call can be further away than `Run`'s next
    `step()` (eval in between, for example).

  The default 10-minute safety margin is meant to cover this. The docs will say that the stop
  time is approximate with the adapters, and that users should raise `budget`'s safety margin
  if they evaluate rarely but for a long time.
- **Mid-epoch resume (Lightning).** Lightning itself warns that resuming from a mid-epoch
  checkpoint can be unreliable with a plain DataLoader: it does not fast-forward the data the
  way `Run.loop()` does. The adapter will not claim an exact mid-epoch resume for Lightning
  until a GPU check shows it (Q6).
- **Mid-epoch resume (HF).** HF skips the already-trained batches itself
  (`skip_first_batches`), so the HF adapter can aim for an exact mid-epoch resume, subject to
  a GPU check.

## 4. Reuse of `checkpoint/manager.py` versus adapter glue

Nothing about saving, loading, mode encoding, RNG, fingerprints, frozen-weight signatures,
storage, manifests or sha256 checks is duplicated in the adapters. Two refactors make that
possible. Both are internal; the public API is unchanged, and the existing test suite is the
regression guard.

**R1. Split `CheckpointManager.load()` into read and restore.**
- `read(path, fingerprint=None)` returns the **decoded** payload: full-precision dicts in any
  mode, plus meta. It runs the fingerprint check before anything is changed.
- `restore(payload, model, optimizer, ...)` is what `load()` does today after reading.
- `load()` becomes `restore(read(...))`, so its behavior is unchanged.
- A helper fills frozen weights from the live model after the signature check, for
  trainable-only checkpoints (E7).

The adapters need the decoded state dicts *without* loading them into objects, so they can hand
them to the framework (section 5). The write side is also split so it accepts a ready-made dict:
- `encode + atomic write + keep_last + async` gets a `save_payload(step, payload)` entry point.
- `save(objects...)` keeps building the payload from objects, then calls `save_payload`.

The Lightning `checkpoint_io` uses that entry point (section 5).

**R2. Move the orchestration out of `Run` into one shared object** (working name
`CheckpointSession`, in `efficio/checkpoint/session.py`; torch code stays in `checkpoint/`). Today
`run.py` holds this logic as private methods:
- storage owner check (`_check_storage_owner`, `StorageInUseError`, `overwrite_storage`);
- choosing the local or the newer stored checkpoint, with sha256-verified download
  (`_resume_file`);
- different-run handling (`_different_run`, `_move_aside_local_checkpoints`);
- save + upload, and waiting for the upload (`save`, `_upload`);
- the budget / SIGTERM stop decision and stop reason (`_maybe_save_or_stop`, the signal handler);
- the final upload status for `shutdown_when_done`;
- the checkpoint and storage lines of the report.

`Run` keeps its public behavior and delegates to the session. The adapters use the same session.
Budget parsing and `Budget.nearly_out()` stay in `interfaces.py` and are not touched.

**What stays adapter-specific (glue only):**

| Glue | HF | Lightning |
|---|---|---|
| Objects to save | `model` (unwrapped), `optimizer` (unwrapped from accelerate's wrapper), `lr_scheduler`, scaler from `trainer.accelerator.scaler` | Lightning's own checkpoint dict, through `checkpoint_io` |
| Framework state kept with the checkpoint | `TrainerState` JSON (`state.to_json_string()`) in the manager's `extra` dict | Already inside Lightning's dict (`loops`, `callbacks`, precision state) |
| TrainState translation | `step = state.global_step`, `epoch = int(state.epoch)`, `batch_size = args.per_device_train_batch_size`, `grad_accum_steps = args.gradient_accumulation_steps`, `batch_in_epoch` derived from `global_step` and steps per epoch (only informational: HF does its own skip) | `step = trainer.global_step`, `epoch = trainer.current_epoch`, `batch_in_epoch = batch_idx + 1`, `grad_accum_steps = trainer.accumulate_grad_batches` |
| Resume hand-off | Write an HF-layout checkpoint folder (section 5) | Return the Efficio file path as `ckpt_path`; `checkpoint_io.load_checkpoint` decodes it |
| Stop request | `control.should_training_stop` | `trainer.should_stop` |

`TrainState` is filled for the manifest, the log lines and the report. Framework state that
`TrainState` cannot hold goes in the manager's existing `extra` dict. **No change to
`efficio/interfaces.py` is needed.**

## 5. Resume

**Why Efficio cannot just load everything in `on_train_begin` / `on_train_start`.** Both
frameworks decide where to continue **before** these hooks run:
- HF computes `epochs_trained` and `steps_trained_in_current_epoch` from `resume_from_checkpoint`
  before `on_train_begin`.
- Lightning restores its loop progress from `ckpt_path` before `on_train_start`.

If Efficio loaded the weights in the hook, the trainer would still start at epoch 0 with fresh
loop counters and a different data position. So **resume goes through each framework's own
mechanism**. Efficio picks the checkpoint and translates it.

### HF: `efficio_cb.resume_from(trainer)`

1. **Choose the checkpoint.** `CheckpointSession` checks the storage owner, then picks the local
   checkpoint or the newer stored one (download, sha256). The rules are the same as `Run`.
2. **Read it.** `manager.read()` decodes it (any mode), checks the parameter part of the
   fingerprint, and checks the frozen-weight signature. The optimizer type is checked later, in
   `on_train_begin`. See E3.
3. **Materialize an HF-layout folder** `<checkpoint_dir>/hf_resume/checkpoint-<step>/`:
   - **weights:** load the decoded weights into the live model, then call `trainer.save_model(dir)`,
     so HF writes its own weight format (safetensors, PEFT adapter files) and Efficio never has
     to know those file names;
   - `optimizer.pt`, `scheduler.pt`, `scaler.pt` from the decoded dicts;
   - `trainer_state.json` from `extra`;
   - `rng_state.pth`, written in HF's format, with the CUDA states **trimmed or padded to this
     machine's GPU count**. HF's own loader uses `set_rng_state_all`, which fails when the GPU
     count changes, the same bug Efficio fixed for Kaggle 2x T4 → Colab T4.
4. **Return that folder.** HF then restores everything, skips the trained batches and restores
   the RNG at its own correct moment.
5. **Clean up.** The folder is deleted in `on_train_begin`, after HF has loaded it.

The cost is one extra full-size write at resume time: some seconds, and temporary disk use.

### Lightning: `efficio_cb.resume_from(trainer, model)` and `efficio_cb.checkpoint_io`

Lightning has a public checkpoint I/O plugin (`CheckpointIO`: `save_checkpoint`,
`load_checkpoint`, `remove_checkpoint`), so Efficio plugs in at that layer:
- **Saving.** `save_checkpoint(dict, path)` encodes Lightning's complete dict with the chosen
  mode: 8-bit optimizer states, and fp16 weights for smallest. It writes the file atomically
  through the manager's write path (R1), keeps the last N and starts the upload.
- **Resume.** `resume_from(trainer, model)` picks and downloads the checkpoint exactly as for HF,
  and returns its local path. Lightning passes that path to `load_checkpoint`, which decodes it
  and fills frozen weights (E7).
- **What Lightning restores itself:** `loops`, the callbacks' state and the precision plugin's
  scaler.
- **What Efficio restores:** the global RNG states, which Lightning does not restore. Efficio
  uses its safe per-GPU restore, in `on_train_start` (Q6).

The alternative was to do it the HF way: save objects through `manager.save()` and rebuild a
Lightning dict at resume time. That would mean depending on the private layout of Lightning's
`loops` / `callbacks` keys. The CheckpointIO route keeps Lightning's dict intact and versioned by
Lightning itself. The price is the extra `plugins=[efficio_cb.checkpoint_io]` line (Q2).

## 6. Open questions and edge cases

**Q1. The user's own checkpointing is also on.**
- HF `save_strategy="steps"` or `"epoch"`: two independent sets of checkpoints, twice the disk
  use (limited on Colab/Kaggle) and twice the save time.
- Lightning's default `ModelCheckpoint`: its saves would also go through `efficio_cb.checkpoint_io`,
  so they would be encoded, but would they also be uploaded and counted in retention?

Proposal: warn once, and recommend `save_strategy="no"` / `enable_checkpointing=False`. Do not
change the user's settings silently. In Lightning, upload only files in Efficio's
`checkpoint_dir`, and only encode the other files. **Open:** should a Lightning `ModelCheckpoint`
with `monitor=` (best-model saving) be kept fully separate instead?

**Q2. API friction.** Should there be a one-call wrapper (`efficio_cb.train(trainer)` /
`efficio_cb.fit(trainer, model, ...)`) instead of `resume_from(...)` plus `plugins=[...]`? The
proposal keeps the framework's own `train()` / `fit()` call visible.

**Q3. Precision.** It stays the trainer's job: Efficio never wraps autocast in the adapters.
Efficio only compares the trainer's setting with `detect_hardware()`, and warns:
- fp32 on a GPU: suggest `fp16=True` on a T4, `bf16=True` on compute capability 8.0 and newer;
- bf16 on a GPU below 8.0: emulated and slow; the native-bf16 rule is compute capability, not
  `is_bf16_supported()`.

The GradScaler state is saved through the framework (HF: `scaler.pt`; Lightning: the precision
plugin's state). **Open:** warn only, or raise for bf16 on a T4?

**Q4. `save_trainable_only="auto"` with PEFT.** It fits well: LoRA freezes the base model, so
"auto" saves only the adapter weights plus a signature of the base. That is a large size win.
**Open:**
- What does the frozen-weight signature do with a quantized base, such as a bitsandbytes 4-bit
  base (`Params4bit`, extra quant-state keys in `state_dict()`)?
- How long does it take to compute signatures for thousands of frozen tensors of a 7B base?

This needs a measurement before any claim. It overlaps with the PEFT adapter item in 0.3.0.

**Q5. Light mode with 8-bit optimizers.** QLoRA setups often use `optim="adamw_bnb_8bit"`, whose
states are already 8-bit. Does `light.encode_optimizer` skip or break on them? Proposal: skip
states that are already integer, and test it. The size claims stay the ones measured for `Run`.

**Q6. RNG and data order at resume.**
- **HF:** HF restores the RNG from `rng_state.pth` right before skipping batches. Efficio's
  trimmed/padded file should give the same data order as an unbroken run. Needs a GPU check.
- **Lightning:** restoring the RNG in `on_train_start` may come before or after the DataLoader
  iterator draws its shuffle seed. Needs a test of the exact order. Is the mid-epoch resume
  through Lightning's `StatefulDataLoader` support worth using where it is available?

**Q7. Lightning has its own SIGTERM handling** (its signal connector sets
`trainer.received_sigterm`). Proposal: read that flag instead of installing a second handler.
Install Efficio's own only when Lightning has none. HF installs none, so the adapter installs one
there, as `Run` does.

**Q8. Package and version floors.**
- `lightning.pytorch.Callback` and `pytorch_lightning.Callback` are different classes, and
  Lightning checks the callback type. Proposal: support `lightning.pytorch` first, and
  `pytorch_lightning` through a second class chosen by which package is installed.
- **Open:** the minimum versions of transformers and lightning, and whether CI installs them for
  tests (they are heavy) or the tests use small fakes plus one optional real run.

**Q9. Can the budget clock use the session start?** Colab/Kaggle quota starts at the session
start, not when the callback is created. This is the same limitation as `Run`; no change is
proposed here.

**Edge cases the implementation must handle:**
- **E1. Multi-process training.**
  - Affected setups: HF launched with `torchrun`, Lightning `strategy="ddp"` / `"ddp_notebook"`.
  - Every rank would run the callback.
  - Only rank 0 may save or upload, and the stop decision must be the same on all ranks, or
    they hang.
  - This belongs with the DDP work planned for 0.3.0. Until then the adapters raise a clear
    error on more than one process.
- **E2. HF DataParallel.** HF wraps the model in `DataParallel` on Kaggle 2x T4 by default. That
  is a single process, so it is supported. The fingerprint uses the unwrapped model's names (no
  `module.` prefix), so a Kaggle → Colab resume matches.
- **E3. Wrapped objects.**
  - HF passes accelerate's `AcceleratedOptimizer`, and Lightning can hand out a
    `LightningOptimizer`. The fingerprint must use the inner optimizer's type, or every resume
    would look like a different run.
  - HF creates the optimizer inside `train()`. `resume_from()` checks the parameter part of the
    fingerprint first, and `on_train_begin` checks the optimizer type.
  - **Open:** if the optimizer type differs, HF has already loaded the materialized checkpoint
    by then. Abort, or warn and go on?
- **E4. Callback added after init.** HF `trainer.add_callback(...)` after `Trainer(...)` misses
  `on_init_end`, so the checks are repeated in `on_train_begin`.
- **E5. `load_best_model_at_end=True` (HF).** It replaces the weights after Efficio's final
  save, so the last checkpoint holds the last weights, not the best ones. Warn once.
- **E6. Lightning `min_epochs` / `min_steps`.** A stop request is ignored until they are
  reached. Warn when the budget stop is ignored, because the run can then overrun the budget.
- **E7. Trainable-only checkpoints and strict loading.** Lightning loads strictly, and HF warns
  on missing keys. After the signature check, the adapter fills the frozen tensors from the live
  model, so the framework sees a complete state dict.
- **E8. Iterable datasets / `max_steps` without epochs (HF).** `state.epoch` is fractional or
  meaningless, so `TrainState.epoch` is informational only.

## 7. `efficio/interfaces.py`

**No change is needed.** `TrainState` covers what the manifest and the report use. Framework
state rides in the checkpoint's existing `extra` dict, and `Budget` / `HardwareInfo` are used as
they are. If the review decides `TrainState` should grow a field, such as a framework name for
the manifest, that is a separate change and needs both maintainers to agree.

## Implementation plan (after this design is approved)

1. **Refactor the manager** (R1): `read` / `restore` and `save_payload`. Run the existing tests
   unchanged.
2. **Extract `CheckpointSession`** (R2); `Run` delegates to it. Run the existing tests unchanged.
3. **HF adapter**, with CPU tests on a tiny model:
   - budget stop;
   - periodic, final and SIGTERM saves;
   - resume through the materialized folder;
   - each mode;
   - trainable-only;
   - fingerprint mismatch;
   - GPU-count RNG trimming (mocked).
4. **Lightning adapter and `checkpoint_io`**, with the same test list.
5. **Packaging:** `efficio[hf]` and `efficio[lightning]` extras. Remove the planned stubs.
6. **Docs:** a README section, CHANGELOG, troubleshooting entries. The README status stays
   "Planned" or "needs GPU verification" until the GPU checks pass.
7. **GPU verification:** Colab T4 budget stop + resume for both adapters, and a Kaggle 2x T4 →
   Colab T4 resume for HF.
