"""Smallest working example. Runs on CPU, Colab or Kaggle.

Try the resume feature: run it with a tiny budget, then run it again.
    python examples/quickstart.py --budget 20s
    python examples/quickstart.py            # picks up where the first run stopped
"""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import efficio

parser = argparse.ArgumentParser()
parser.add_argument("--budget", default=None, help='e.g. "2h", "90m", "20s"')
parser.add_argument("--epochs", type=int, default=200)
args = parser.parse_args()

# Fake data: 20,000 samples, 64 features, 10 classes
X, y = torch.randn(20_000, 64), torch.randint(0, 10, (20_000,))
loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 10))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# A budget of "20s" with a 10s safety margin stops after about 10 seconds
run = efficio.Run(budget=args.budget, checkpoint_dir="./efficio_checkpoints",
                  save_every_minutes=1)
run.budget.safety_margin_seconds = 10
model, optimizer = run.prepare(model, optimizer)

for xb, yb in run.loop(loader, epochs=args.epochs):
    with run.autocast():
        loss = loss_fn(model(xb), yb)
    run.backward(loss)
    run.step()

run.report()
