"""Baseline: plain PyTorch, fp32, no Efficio and no optimizations.

Run from the repo root:
    python benchmarks/resnet_cifar_baseline.py --epochs 5

Keep this script plain. Do not add mixed precision, cudnn.benchmark, pin_memory or any
other speed-up here; it is the reference the Efficio run is compared against.
"""

import argparse
import time

import torch
from torch import nn

from common import (GpuUtilSampler, evaluate, make_loaders, make_model, make_optimizer,
                    save_results, sync)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--data-dir", default="./data", help="where CIFAR-10 is stored")
args = parser.parse_args()

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = make_loaders(batch_size=args.batch_size, data_dir=args.data_dir)
model = make_model().to(device)
optimizer, scheduler = make_optimizer(model, args.epochs, len(train_loader))
loss_fn = nn.CrossEntropyLoss()

epoch_times = []
with GpuUtilSampler() as sampler:
    for epoch in range(args.epochs):
        model.train()
        sync()
        start = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        sync()
        epoch_times.append(time.perf_counter() - start)
        print(f"epoch {epoch + 1}/{args.epochs}: {epoch_times[-1]:.1f} s, loss {loss.item():.3f}")

test_acc = evaluate(model, test_loader, device)
save_results("baseline", epoch_times, len(train_loader) * args.batch_size, sampler.average,
             test_acc, extra={"precision": "fp32", "batch_size": args.batch_size})
