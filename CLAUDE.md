# Efficio: project context

## What it is
Efficio is a Python library that gets more training out of every GPU hour on Google Colab, Kaggle and local GPUs. Free GPUs have strict limits (Colab: a few hours/day, Kaggle: ~30 h/week, 12 h sessions), and much of that time is wasted: idle sessions, slow data loading, full-precision math, and runs lost when a session dies. Users add a few lines to their PyTorch training loop.
- On Colab/Kaggle: faster training, less wasted quota, progress saved before the limit hits so a run continues after the reset or on the other platform.
- On local GPUs: there is NO quota. The benefit is speed and crash safety (power cuts, restarts, OOM) and help with small-memory GPUs. Never claim Efficio extends local time limits.

## Scope: v1 only
Target users: students and individuals on free tiers and personal GPUs. Do NOT add or mention company cloud training, AWS/GCP, spot instances, multi-node training or FSDP/DeepSpeed tuning. That is reserved for a possible v2.
Framework: PyTorch now. TensorFlow/Keras support (a Keras callback for model.fit) is planned for Phase 4.

## User-facing API
```python
import efficio
run = efficio.Run(budget="2h")                    # stop and save before the budget ends
model, optimizer = run.prepare(model, optimizer)  # auto-resumes from the latest checkpoint
for batch in run.loop(loader, epochs=20):
    with run.autocast():
        loss = compute_loss(model, batch)
    run.backward(loss)
    run.step()
run.report()
```

## Repo layout
- efficio/interfaces.py: shared data objects HardwareInfo, Budget, TrainState. This is the contract between the two teammates. Do not change it unless I say both teammates agreed.
- efficio/env/: detect.py (platform: colab/kaggle/local, default checkpoint dir), hardware.py (GPU name, count, VRAM, compute capability, precision)
- efficio/checkpoint/: manager.py (atomic save via temp file + os.replace, keeps last 2, full resume state), storage.py and light.py (planned)
- efficio/run.py: the Run class tying everything together
- efficio/speed/, budget/, report/, adapters/, preflight.py: placeholders for planned features (each file's docstring says owner and phase)
- examples/quickstart.py: tiny demo; run with --budget 20s, then again to see resume
- benchmarks/: baseline vs Efficio comparisons
- tests/: pytest

## Features and status
Available and verified on Colab (1x T4) and Kaggle (2x T4):
- Platform and GPU detection; precision choice (fp16 + GradScaler on T4/older, bf16 on compute capability >= 8.0, fp32 on CPU)
- Mixed precision via run.autocast(), gradient accumulation
- Time budget with safety margin: stops and saves before the limit; SIGTERM triggers a save at the next step
- Checkpointing: model, optimizer, scheduler, GradScaler, RNG states (python/numpy/torch/cuda), epoch, step, batch position; atomic writes; periodic saves; exact mid-epoch resume (verified: both platforms ended at the identical step count after resuming)
- Basic end-of-run report
Planned:
- Speed: auto batch size with OOM recovery + gradient accumulation; preprocessing cache; both Kaggle GPUs via accelerate.notebook_launcher
- Quota: CPU pre-flight check; budget planner (time first steps, fit epochs + LR schedule to the budget); auto-shutdown (Colab: google.colab.runtime.unassign()); cheaper validation
- Checkpointing: light mode (trainable params only, fp16 weights, 8-bit optimizer state; "exact" vs "light" modes); Hugging Face Hub and Google Drive storage for cross-platform resume; with multi-GPU only rank 0 saves
- Adapters: Hugging Face Trainer, PyTorch Lightning, TensorFlow/Keras
- Visibility: detailed report (GPU utilization via nvidia-smi/pynvml, data vs compute vs idle time, wasted quota, concrete tips); Telegram/email notifications

## Key technical decisions (do not undo)
- Native bf16 is decided by compute capability >= 8.0, NOT torch.cuda.is_bf16_supported(), which reports emulated bf16 on a T4.
- Checkpoints are always loaded with map_location="cpu"; load_state_dict moves tensors to the model's device. RNG states must be CPU ByteTensors (this fixed a real GPU resume bug).
- Keep interfaces.py, env/detect.py, budget and report code free of torch imports so TensorFlow/Keras support can reuse them. Torch-specific code stays in checkpoint/, run.py and speed/.

## Roadmap
- Phase 1: env + GPU detection, mixed precision, local checkpoint/resume, budget stop. DONE and verified.
- Phase 2: Hugging Face Hub storage, light checkpoints, auto batch size, pre-flight check, profiler.
- Phase 3: Google Drive storage, auto-shutdown, data cache, budget planner, full report.
- Phase 4: multi-GPU, notifications, HF Trainer, Lightning and TensorFlow/Keras adapters.
- Phase 5: benchmarks (ResNet-18/CIFAR-10, DistilBERT fine-tune, small LoRA), docs, PyPI release. Headline metric: epochs per quota hour, before vs after.

## Team
Two people. Person A: Run class, env/hardware, checkpointing, storage, budget timer, emergency save, auto-shutdown, Lightning adapter. Person B: AMP, auto batch size, cache, multi-GPU, cheaper eval, budget planner, pre-flight, profiler, report, notifications, HF Trainer adapter, benchmarks. Each works on their own branch and merges into main every few days.

## Rules for working in this repo
- Never add Co-Authored-By trailers or any AI attribution to commits or PRs. Claude must never appear as an author, co-author or contributor. Allowed human authors: Durvesh-dev and Hassaan Tole (HuNTer8272).
- Never commit .venv, data/, efficio_checkpoints/ or secrets. Never put tokens in code.
- The developer is on Windows with PowerShell. Give PowerShell-compatible commands.
- Ask before git push --force, deleting files, or changing interfaces.py.
- The benchmark baseline scripts must stay plain PyTorch with no Efficio features; never "optimize" the baseline.
- Never invent benchmark numbers. Only report measured results.
- Add or update tests for new features. Keep README feature statuses accurate (Available only after it is verified on a GPU).
- After finishing a task, append a dated entry to PROGRESS.md: what changed, files touched, how it was tested, what still needs a GPU test, and open questions.
