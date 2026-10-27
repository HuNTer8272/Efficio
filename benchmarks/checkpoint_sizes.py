"""Checkpoint size and save time: exact vs light vs trainable-only.

    python benchmarks/checkpoint_sizes.py            # GPU if available, else CPU
    python benchmarks/checkpoint_sizes.py --quick    # smaller models, for a CPU smoke test

Three setups, each trained for a few steps on fixed synthetic data with learnable labels:
- ResNet-18 (CIFAR stem) with SGD + momentum
- a small transformer encoder with AdamW
- a frozen ResNet-18 backbone with a small trainable head, AdamW

For every setup and checkpoint kind it reports the file size, the time a synchronous save
blocks training, and with background saving the time it blocks plus the time written in the
background. For light mode it also resumes from the checkpoint and compares with an
uninterrupted run: the max weight difference right after loading, and the loss after
RESUME_STEPS more steps. As a yardstick it also reports the loss difference of an exact
resume with 1e-6 relative noise on the weights: training is sensitive, so differences of
that order are normal run-to-run variation, not damage from light mode.
Results go to benchmarks/results/checkpoint_sizes_<platform>_<time>.json.
"""

import argparse
import copy
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from efficio.checkpoint import CheckpointManager
from efficio.env import detect_platform
from efficio.interfaces import TrainState

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TRAIN_STEPS, RESUME_STEPS = 20, 20


# ---------------------------------------------------------------- setups
def resnet18(num_classes=10):
    import torchvision
    model = torchvision.models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class TinyTransformer(nn.Module):
    def __init__(self, vocab=8000, d=256, layers=4, classes=10):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=4 * d, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, classes)

    def forward(self, x):
        return self.head(self.encoder(self.embed(x)).mean(1))


def _image_data(n):
    """Random images whose label is a fixed function of the pixels, so a model can learn it
    (random labels make training chaotic and the resume comparison meaningless)."""
    gen = torch.Generator().manual_seed(123)
    x = torch.randn(n, 3, 32, 32, generator=gen)
    proj = torch.randn(3 * 32 * 32, 10, generator=gen)
    return x, (x.flatten(1) @ proj).argmax(1)


def _token_data(n):
    gen = torch.Generator().manual_seed(123)
    x = torch.randint(0, 8000, (n, 64), generator=gen)
    return x, x[:, :4].sum(1) % 10               # learnable: depends on the first tokens


def make_setup(name, device, quick):
    torch.manual_seed(0)
    n = 64 if quick else 256
    if name == "resnet18_sgd":
        model = resnet18()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        x, y = _image_data(n)
    elif name == "transformer_adamw":
        model = TinyTransformer(d=128 if quick else 256, layers=2 if quick else 4)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        x, y = _token_data(n)
    elif name == "frozen_backbone_head":
        backbone = resnet18(num_classes=10)
        backbone.fc = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad_(False)
        model = nn.Sequential(backbone, nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        x, y = _image_data(n)
    else:
        raise ValueError(name)
    return model.to(device), opt, x.to(device), y.to(device)


def train(model, opt, x, y, steps, seed):
    """Fixed mini-batches drawn from x, y with a seeded generator; returns the last loss."""
    gen = torch.Generator().manual_seed(seed)
    loss = None
    for _ in range(steps):
        idx = torch.randint(0, len(x), (max(len(x) // 2, 1),), generator=gen).to(x.device)
        loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return loss.item()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------- measurements
def measure(name, kind, device, quick, tmp):
    model, opt, x, y = make_setup(name, device, quick)
    model.train()
    train(model, opt, x, y, TRAIN_STEPS, seed=1)
    mode = "light" if kind == "light" else "exact"
    trainable_only = kind == "trainable_only"
    if trainable_only and all(p.requires_grad for p in model.parameters()):
        return None                                    # nothing frozen: same as exact

    def manager(sub, async_save=False):
        return CheckpointManager(os.path.join(tmp, f"{name}_{kind}_{sub}"), mode=mode,
                                 trainable_only=trainable_only, async_save=async_save)

    # Synchronous save
    sync_mgr = manager("sync")
    sync(device)
    start = time.perf_counter()
    path = sync_mgr.save(TrainState(step=TRAIN_STEPS), model, opt)
    sync_s = time.perf_counter() - start
    size = os.path.getsize(path)

    # Background save: time training is blocked, and time written in the background
    bg_mgr = manager("background", async_save=True)
    sync(device)
    start = time.perf_counter()
    bg_mgr.save(TrainState(step=TRAIN_STEPS), model, opt, blocking=False)
    blocked_s = time.perf_counter() - start
    bg_mgr.wait()
    background_s = bg_mgr.stats["background_s"]

    result = {"setup": name, "kind": kind, "file_mb": round(size / 1024**2, 2),
              "sync_save_s": round(sync_s, 3), "background_blocking_s": round(blocked_s, 3),
              "background_write_s": round(background_s, 3)}

    if mode == "light":
        # Resume from the light checkpoint and compare with an uninterrupted run
        reference, ref_opt, _, _ = make_setup(name, device, quick)    # uninterrupted copy
        reference.load_state_dict(model.state_dict())
        ref_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        fresh, fresh_opt, _, _ = make_setup(name, device, quick)
        sync_mgr.load(fresh, fresh_opt, map_location=device)
        result["max_weight_diff_after_load"] = max(
            (a - b).abs().max().item() for a, b in zip(fresh.state_dict().values(),
                                                       model.state_dict().values())
            if a.is_floating_point())
        # Sensitivity baseline: an exact resume with 1e-6 relative noise on the weights.
        # A light-mode loss difference of the same order is just normal run-to-run variation.
        noisy, noisy_opt, _, _ = make_setup(name, device, quick)
        noisy.load_state_dict(_with_noise(model.state_dict(), 1e-6))
        noisy_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        ref_loss = train(reference, ref_opt, x, y, RESUME_STEPS, seed=2)
        resumed_loss = train(fresh, fresh_opt, x, y, RESUME_STEPS, seed=2)
        noisy_loss = train(noisy, noisy_opt, x, y, RESUME_STEPS, seed=2)
        result.update(loss_after_resume=round(resumed_loss, 6),
                      loss_uninterrupted=round(ref_loss, 6),
                      loss_diff=round(abs(resumed_loss - ref_loss), 6),
                      loss_diff_noise_1e6=round(abs(noisy_loss - ref_loss), 6))
    return result


def _with_noise(state_dict, rel):
    gen = torch.Generator().manual_seed(5)
    out = {}
    for k, v in state_dict.items():
        if v.is_floating_point() and v.numel():
            noise = torch.randn(v.shape, generator=gen).to(v.device, v.dtype)
            out[k] = v + noise * v.abs().mean() * rel
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller models (CPU smoke test)")
    parser.add_argument("--cpu", action="store_true", help="use the CPU even with a GPU")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "none"
    print(f"device: {device} ({gpu}), torch {torch.__version__}")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("resnet18_sgd", "transformer_adamw", "frozen_backbone_head"):
            kinds = (["exact_all_params", "trainable_only", "light"]
                     if name == "frozen_backbone_head" else ["exact", "light"])
            for kind in kinds:
                k = "exact" if kind == "exact_all_params" else kind
                row = measure(name, k, device, args.quick, tmp)
                if row is None:
                    continue
                row["kind"] = kind                       # exact_all_params: frozen too
                rows.append(row)
                print(json.dumps(row))

    print()
    header = (f"{'setup':<22}{'kind':<18}{'file MB':>9}{'sync s':>9}{'bg block s':>11}"
              f"{'bg write s':>11}{'max w diff':>12}{'loss diff':>11}{'noise diff':>11}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['setup']:<22}{r['kind']:<18}{r['file_mb']:>9.2f}{r['sync_save_s']:>9.3f}"
              f"{r['background_blocking_s']:>11.3f}{r['background_write_s']:>11.3f}"
              f"{_fmt(r.get('max_weight_diff_after_load')):>12}{_fmt(r.get('loss_diff')):>11}"
              f"{_fmt(r.get('loss_diff_noise_1e6')):>11}")
    print("\nloss diff: light resume vs uninterrupted run after RESUME_STEPS steps. noise diff: "
          "the same for an exact resume with 1e-6 relative weight noise (normal variation).")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"checkpoint_sizes_{detect_platform()}_{stamp}.json"
    out.write_text(json.dumps({"device": str(device), "gpu": gpu, "torch": torch.__version__,
                               "quick": args.quick, "train_steps": TRAIN_STEPS,
                               "resume_steps": RESUME_STEPS, "rows": rows}, indent=2))
    print(f"\nsaved {out}")


def _fmt(v):
    return "-" if v is None else f"{v:.2e}"


if __name__ == "__main__":
    main()
