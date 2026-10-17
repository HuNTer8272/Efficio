"""Same benchmark as resnet_cifar_baseline.py, trained through Efficio.

Run from the repo root (after `pip install -e .`):
    python benchmarks/resnet_cifar_efficio.py --epochs 5
"""

import argparse
import tempfile
import time

import torch
from torch import nn

import efficio
from common import (GpuUtilSampler, evaluate, make_loaders, make_model, make_optimizer,
                    save_results, sync)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--no-optimize-data", action="store_true",
                    help="turn off DataLoader tuning and channels_last to measure them separately")
args = parser.parse_args()
optimize = not args.no_optimize_data

torch.manual_seed(0)

# Same order as the baseline so the seeded RNG produces the same model and data order
train_loader, test_loader = make_loaders(batch_size=args.batch_size)
model = make_model()
optimizer, scheduler = make_optimizer(model, args.epochs, len(train_loader))
loss_fn = nn.CrossEntropyLoss()

# Fresh temporary checkpoint dir and resume=False, so every benchmark run starts from scratch
run = efficio.Run(checkpoint_dir=tempfile.mkdtemp(prefix="efficio_bench_"), resume=False,
                  optimize_data=optimize, channels_last="auto" if optimize else False)
model, optimizer = run.prepare(model, optimizer, scheduler)

epoch_times = []
with GpuUtilSampler() as sampler:
    sync()
    start = time.perf_counter()
    prev_epoch = run.state.epoch
    for x, y in run.loop(train_loader, epochs=args.epochs):
        # run.loop yields one flat stream of batches, so an epoch ends when run.state.epoch changes
        if run.state.epoch != prev_epoch:
            sync()
            epoch_times.append(time.perf_counter() - start)
            print(f"epoch {prev_epoch + 1}/{args.epochs}: {epoch_times[-1]:.1f} s")
            start = time.perf_counter()
            prev_epoch = run.state.epoch
        with run.autocast():
            loss = loss_fn(model(x), y)
        run.backward(loss)
        run.step()          # also steps the scheduler; do not call scheduler.step() here
    # The last epoch's time also includes Efficio's final checkpoint save at the end of run.loop
    sync()
    epoch_times.append(time.perf_counter() - start)
    print(f"epoch {prev_epoch + 1}/{args.epochs}: {epoch_times[-1]:.1f} s")

test_acc = evaluate(model, test_loader, run.device, autocast=run.autocast)
run.report()
save_results("efficio" if optimize else "efficio-no-data-opt", epoch_times,
             len(train_loader) * args.batch_size, sampler.average, test_acc,
             extra={"precision": run.hw.precision, "batch_size": args.batch_size,
                    "optimizations": run.optimizations})
