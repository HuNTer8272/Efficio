# Efficio

**Get more training out of every GPU hour.**

Efficio makes training faster, cuts wasted GPU time so free quotas on Colab and Kaggle last longer, and saves your progress so a run can continue after the limit resets or on another platform. It works the same on Colab, Kaggle and local GPUs.

## Quick start

```python
import efficio

run = efficio.Run(budget="2h")            # stop and save 10 minutes before the budget ends
model, optimizer = run.prepare(model, optimizer)   # resumes automatically if a checkpoint exists

for batch in run.loop(loader, epochs=20):
    with run.autocast():                  # fp16 on T4, bf16 on newer GPUs, fp32 on CPU
        loss = compute_loss(model, batch)
    run.backward(loss)
    run.step()

run.report()
```

Try it: `python examples/quickstart.py --budget 20s`, then run it again without `--budget` to watch it resume.

## Status

| Feature | Owner | Phase | Status |
|---|---|---|---|
| Environment + GPU detection | A | 1 | done (needs testing on Colab/Kaggle) |
| Checkpoint + resume (local, exact) | A | 1 | done |
| Mixed precision | B | 1 | basic version in `Run.autocast` |
| Budget timer + emergency save | A | 2 | basic version in `Run` |
| Profiler | B | 1 | todo |
| Light checkpoints, HF Hub storage | A | 2 | todo |
| Auto batch size, pre-flight check | B | 2 | todo |
| Drive storage, auto-shutdown | A | 3 | todo |
| Data cache, budget planner, report | B | 3 | todo |
| Multi-GPU, notifications, adapters | A/B | 4 | todo |

## Development

```bash
pip install -e ".[dev]"
pytest
```

Shared data objects live in `efficio/interfaces.py`. Change them together.
