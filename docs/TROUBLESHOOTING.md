# Troubleshooting

Common problems with Efficio 0.2.x, what causes them and what to do. Each entry follows what the code does today. Error messages are quoted as Efficio prints them, so you can search this page for the text you see.

- [Hugging Face token not found](#hugging-face-token-not-found)
- [Resume fails: the checkpoint was saved by a different run](#resume-fails-the-checkpoint-was-saved-by-a-different-run)
- [Resume fails: frozen weights don't match](#resume-fails-frozen-weights-dont-match)
- [StorageInUseError: the storage holds another run's checkpoints](#storageinuseerror-the-storage-holds-another-runs-checkpoints)
- [Resuming on a machine with a different number of GPUs](#resuming-on-a-machine-with-a-different-number-of-gpus)
- [The Colab runtime did not shut down after run.report()](#the-colab-runtime-did-not-shut-down-after-runreport)
- [Out of memory during training](#out-of-memory-during-training)
- [pip install problems with the hub extra](#pip-install-problems-with-the-hub-extra)
- [Other storage errors](#other-storage-errors)

## Hugging Face token not found

**Symptom:** `StorageError: Could not open or create the Hugging Face repo user/name (...)`, followed by the lines "Add a secret named HF_TOKEN holding a Hugging Face token with write access".

**Cause:** Efficio found no token. It looks in this order: the `token=` argument of `HuggingFaceHub`, the `HF_TOKEN` environment variable, a Colab secret named `HF_TOKEN`, a Kaggle secret named `HF_TOKEN`. If none is found, it falls back to a saved `huggingface-cli login`, if any. A secret with a different name, or one that notebook access is not turned on for, is not found, and Efficio does not report why: it simply gets no value.

**Fix:**
- Name the secret exactly `HF_TOKEN`.
- Colab: open the key icon in the left sidebar, add the secret and turn on notebook access.
- Kaggle: open Add-ons > Secrets and tick the secret for this notebook.
- Elsewhere: set the `HF_TOKEN` environment variable.
- The token needs write access (create it at huggingface.co/settings/tokens).
- When the connection works, Efficio logs `Hugging Face token: from <source>`. Check that line to see where the token came from. The token itself is never printed.

If a token was found but the error remains, the message in brackets (for example a permission or repo-name error) comes from the Hugging Face Hub, with the token removed. `repo_id` must look like `user/repo-name`.

## Resume fails: the checkpoint was saved by a different run

**Symptom:** `DifferentRunError: <checkpoint> was saved by a different run (...) than this one (...): the model's parameter names or shapes, or the optimizer type, differ. Use another checkpoint_dir or storage for this run.` With the default `resume="auto"` you see a warning instead: `WARNING: not resuming from ...: it was saved by a different run ... Starting fresh.`

**Cause:** Each checkpoint stores a fingerprint of the run: the model's parameter names and shapes, and the optimizer type. If either differs from the model and optimizer you pass to `run.prepare()` (a changed layer, a renamed module, `SGD` instead of `AdamW`), the checkpoint is not resumed. The checkpoint mode and settings such as the learning rate are not part of the fingerprint. Checkpoints written without a fingerprint (older versions) are accepted.

**What happens:**
- `resume="auto"` (default): Efficio warns, moves the other run's local checkpoints into an `other_run_<time>` folder inside `checkpoint_dir`, and starts fresh.
- `resume=True`: the error is raised and nothing is changed.

**Fix:**
- To continue the old run, build the same model and optimizer as before `run.prepare()`.
- To start a new run, use a different `checkpoint_dir` (and a different `storage`).
- To get the error instead of a fresh start, pass `resume=True`.

## Resume fails: frozen weights don't match

**Symptom:** `CheckpointMismatchError: <checkpoint> was saved with trainable parameters only, and the frozen parameters of this model don't match the ones it was trained with (...)`

**Cause:** For models with frozen parameters, checkpoints keep only the trainable weights (`save_trainable_only="auto"`) and store a signature of the frozen ones. The model you passed has different frozen weights, a different shape or dtype, or a different set of frozen layers. This error is not turned into a fresh start, even with `resume="auto"`.

**Fix:** Load the same pretrained weights and freeze the same layers before `run.prepare()`, or start a new run with a different `checkpoint_dir`.

## StorageInUseError: the storage holds another run's checkpoints

**Symptom:** `StorageInUseError: <storage> holds checkpoints of a different run (...). Nothing was written there. Use a different storage path, or pass overwrite_storage=True to replace that run's checkpoints.`

**Cause:** `run.prepare()` found checkpoints in `storage` whose fingerprint (see above) differs from this run's, for example two different experiments pointing at the same `hf://user/repo` or the same Drive folder. Efficio refuses before writing anything there. `StorageInUseError` is a kind of `DifferentRunError`, so `except DifferentRunError` catches both.

**Fix:**
- Give each run its own storage, for example `hf://user/repo/exp1` and `hf://user/repo/exp2`, or a separate folder.
- If the old checkpoints are not needed, pass `Run(overwrite_storage=True)`. Efficio then warns, does not resume them, and deletes them once this run's first checkpoint is uploaded.

Runs with the same parameter names and shapes and the same optimizer type have the same fingerprint, so Efficio treats them as one run.

## Resuming on a machine with a different number of GPUs

**What Efficio handles:** Checkpoints are loaded to the CPU first, and the random-number states of the GPUs that both machines have are restored. Resuming a checkpoint from Kaggle 2x T4 on a Colab 1x T4 works (verified on a GPU). The code restores the states of the GPUs both machines have in either direction, and Efficio trains on GPU 0.

**What it doesn't handle:** Efficio uses one GPU. It does not split training over several GPUs. Using both Kaggle T4 GPUs is on the roadmap.

**If it still fails:**
- `IndexError` while restoring CUDA random states: before 0.2.0, resuming a checkpoint from a machine with more GPUs on one with fewer failed this way. Upgrade to 0.2.0 or later.
- Any `DifferentRunError` or `CheckpointMismatchError` has nothing to do with the GPU count; see the entries above.

## The Colab runtime did not shut down after run.report()

**Symptom:** You passed `shutdown_when_done=True` but the runtime kept running.

**Cause and what Efficio prints:**

| Message | Cause |
| --- | --- |
| `shutdown_when_done: no storage is set, so the runtime will NOT be released` | Shutdown needs `storage=`, because the final checkpoint would be lost with the runtime. |
| `NOT shutting down: the final checkpoint is not confirmed in storage` | The final upload failed, or was still running after `upload_timeout` (default 120 seconds). Efficio only releases the runtime once the final upload is confirmed. |
| `not shutting down: run.loop() has not ended` | `run.report()` was called before the training loop finished. |
| `shutdown_when_done: Kaggle sessions can't be stopped from code ...` or `nothing to shut down on a local machine.` | Only Colab can be stopped from code. On Kaggle, stop the session yourself or use Save Version. |
| `WARNING: shutdown_when_done: the Colab runtime is still running 60 s after it was released` | Efficio released the runtime, but Colab had not removed it after 60 seconds. It may still use GPU quota. |

**Fix:** Set `storage=`, wait for the upload (raise `upload_timeout` for large checkpoints or slow connections), and call `run.report()` after the loop. If the 60-second warning appears, use Runtime > Disconnect and delete runtime. `shutdown_when_done` is off by default. Code after `run.report()` does not run when the shutdown works.

## Out of memory during training

**Symptom:** `torch.cuda.OutOfMemoryError` (or a CPU out-of-memory crash).

**Cause:** The batch, model or activations don't fit in memory. Efficio 0.2.x does **not** handle this: it does not catch the error, lower the batch size or retry. Automatic batch size with out-of-memory recovery is on the [roadmap](../README.md#roadmap) and not released yet.

**Workaround:**
- Lower the batch size in your DataLoader and raise `Run(grad_accum_steps=...)` to keep the effective batch size.
- Checkpoints still protect the run. After a crash, the latest checkpoint (periodic saves every `save_every_minutes`, default 15) is there, and running the same code again resumes from it. The saved position is an epoch and a batch number, so for a changed batch size start a new run with a different `checkpoint_dir`.
- `checkpoint_mode="light"` or `"smallest"` makes checkpoints smaller, but does not reduce training memory.

## pip install problems with the hub extra

**Symptom:** `ImportError: Saving to the Hugging Face Hub needs huggingface_hub: pip install "efficio[hub]"`, raised when you use `storage="hf://..."` or `HuggingFaceHub`.

**Cause:** The Hub support needs the optional `huggingface_hub` package (version 0.23 or newer), which `pip install efficio` does not install.

**Fix:** Install the extra, with the quotes, so your shell doesn't treat the brackets as a pattern:

```
pip install "efficio[hub]"
```

On Colab or Kaggle, put `!` in front. Efficio needs Python 3.10 or newer. The extra is only for the Hugging Face Hub; folder and Google Drive storage need nothing extra.

## Other storage errors

- **`Google Drive is not mounted, so <path> would be on the session's own disk and lost when the session ends`:** Mount Drive first (`from google.colab import drive; drive.mount('/content/drive')`). Efficio doesn't mount it for you.
- **`storage (<path>) must be a different folder from checkpoint_dir`:** `storage` holds copies of the checkpoints, so it can't be the same folder as `checkpoint_dir`.
- **`Uploading a checkpoint to <storage> failed: ...`:** An upload failed after 3 attempts. The message is raised at the next `run.step()`. The local checkpoint is fine; fix the cause (connection, token, permissions) and start again to resume from it.
- **`Could not read the checkpoint list in <storage>`:** Check the connection and token. With nothing local to resume from, pass `resume=False` to start fresh.
- **`doesn't match its sha256 in the manifest ... so it was not used`:** A downloaded checkpoint is damaged or was changed. Efficio did not use it. Delete the file and its manifest entry, or start without resume.
