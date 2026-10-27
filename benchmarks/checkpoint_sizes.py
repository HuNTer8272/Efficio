"""Checkpoint size, save time and resume quality: exact vs light vs smallest (and
trainable-only for a frozen backbone).

    python benchmarks/checkpoint_sizes.py            # GPU if available, else CPU
    python benchmarks/checkpoint_sizes.py --quick    # smaller models, for a CPU smoke test

Three setups, each trained for TRAIN_STEPS steps on fixed synthetic data with learnable labels:
- ResNet-18 (CIFAR stem) with SGD + momentum
- a small transformer encoder with AdamW
- a frozen ResNet-18 backbone with a small trainable head, AdamW

For every setup and checkpoint mode it reports:
- the file size, the time a synchronous save blocks training, and with background saving the
  time training is blocked plus the time written in the background;
- resume quality against an uninterrupted run of the same steps:
  - the max weight difference right after loading,
  - loss and accuracy (on the whole synthetic set, eval mode) after one resume and
    RESUME_STEPS more steps,
  - the same after three consecutive save -> load -> RESUME_STEPS cycles.
As a yardstick the same differences are reported for an exact resume with 1e-6 relative
noise on the weights ("noise" rows): training is sensitive, so differences of that order
are normal run-to-run variation, not damage from the checkpoint mode.
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
TRAIN_STEPS = 20
MODES = ("exact", "light", "smallest")


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
    """Fixed mini-batches drawn from x, y with a seeded generator. The global RNG is seeded
    too, so dropout draws the same masks in the reference and the resumed run."""
    model.train()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (max(len(x) // 2, 1),), generator=gen).to(x.device)
        loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def evaluate(model, x, y):
    """(loss, accuracy %) on the whole set in eval mode."""
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(x[i:i + 128]) for i in range(0, len(x), 128)]).float()
    model.train()
    accuracy = 100.0 * (out.argmax(1) == y).float().mean().item()
    return nn.functional.cross_entropy(out, y).item(), accuracy


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def clone(name, device, quick, model, opt):
    """A fresh setup holding a copy of the model and optimizer state."""
    m, o, _, _ = make_setup(name, device, quick)
    m.load_state_dict(model.state_dict())
    o.load_state_dict(copy.deepcopy(opt.state_dict()))
    return m, o


def with_noise(state_dict, rel=1e-6):
    gen = torch.Generator().manual_seed(5)
    out = {}
    for k, v in state_dict.items():
        if v.is_floating_point() and v.numel():
            noise = torch.randn(v.shape, generator=gen).to(v.device, v.dtype)
            out[k] = v + noise * v.abs().mean() * rel
        else:
            out[k] = v
    return out


def max_weight_diff(a, b):
    return max(((x.float() - y.float()).abs().max().item()
                for x, y in zip(a.state_dict().values(), b.state_dict().values())
                if x.is_floating_point()), default=0.0)


# ---------------------------------------------------------------- measurements
def sizes_and_times(name, mode, trainable_only, device, quick, tmp, model, opt):
    def manager(sub, async_save=False):
        return CheckpointManager(os.path.join(tmp, f"{name}_{mode}_{trainable_only}_{sub}"),
                                 mode=mode, trainable_only=trainable_only, async_save=async_save)

    sync_mgr = manager("sync")
    sync(device)
    start = time.perf_counter()
    path = sync_mgr.save(TrainState(step=TRAIN_STEPS), model, opt)
    sync_s = time.perf_counter() - start

    bg_mgr = manager("background", async_save=True)
    sync(device)
    start = time.perf_counter()
    bg_mgr.save(TrainState(step=TRAIN_STEPS), model, opt, blocking=False)
    blocked_s = time.perf_counter() - start
    bg_mgr.wait()
    return {"file_mb": round(os.path.getsize(path) / 1024**2, 2),
            "sync_save_s": round(sync_s, 3), "background_blocking_s": round(blocked_s, 3),
            "background_write_s": round(bg_mgr.stats["background_s"], 3)}


def resume_quality(name, mode, device, quick, tmp, model, opt, x, y, resume_steps):
    """Compare resumed training with an uninterrupted run (after 1 and after 3 resumes).
    mode=None: the noise baseline (exact state + 1e-6 relative weight noise, once)."""
    ref_model, ref_opt = clone(name, device, quick, model, opt)
    run_model, run_opt = clone(name, device, quick, model, opt)
    out, first_diff = {}, None
    for cycle in (1, 2, 3):
        if mode is None:
            if cycle == 1:
                run_model.load_state_dict(with_noise(run_model.state_dict()))
                first_diff = max_weight_diff(run_model, ref_model)
        else:
            mgr = CheckpointManager(os.path.join(tmp, f"q_{name}_{mode}_{cycle}"), mode=mode)
            mgr.save(TrainState(step=cycle), run_model, run_opt)
            run_model, run_opt = make_setup(name, device, quick)[:2]
            mgr.load(run_model, run_opt, map_location=device)
            if cycle == 1:
                first_diff = max_weight_diff(run_model, ref_model)
        train(ref_model, ref_opt, x, y, resume_steps, seed=10 + cycle)
        train(run_model, run_opt, x, y, resume_steps, seed=10 + cycle)
        if cycle in (1, 3):
            (rl, ra), (ml, ma) = evaluate(ref_model, x, y), evaluate(run_model, x, y)
            out[f"loss_diff_{cycle}"] = round(abs(ml - rl), 6)
            out[f"acc_diff_{cycle}"] = round(abs(ma - ra), 3)
            out[f"loss_{cycle}"], out[f"loss_ref_{cycle}"] = round(ml, 6), round(rl, 6)
            out[f"acc_{cycle}"], out[f"acc_ref_{cycle}"] = round(ma, 3), round(ra, 3)
    out["max_weight_diff_after_load"] = first_diff
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller models (CPU smoke test)")
    parser.add_argument("--cpu", action="store_true", help="use the CPU even with a GPU")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "none"
    resume_steps = 10 if args.quick else 20
    print(f"device: {device} ({gpu}), torch {torch.__version__}")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("resnet18_sgd", "transformer_adamw", "frozen_backbone_head"):
            model, opt, x, y = make_setup(name, device, args.quick)
            train(model, opt, x, y, TRAIN_STEPS, seed=1)
            frozen = any(not p.requires_grad for p in model.parameters())
            variants = [(m, False) for m in MODES] + ([("exact", True)] if frozen else [])
            for mode, trainable_only in variants:
                kind = mode + (" (trainable only)" if trainable_only else "")
                row = {"setup": name, "kind": kind,
                       **sizes_and_times(name, mode, trainable_only, device, args.quick, tmp,
                                         model, opt)}
                if not trainable_only:
                    row.update(resume_quality(name, mode, device, args.quick, tmp, model, opt,
                                              x, y, resume_steps))
                rows.append(row)
                print(json.dumps(row))
            rows.append({"setup": name, "kind": "noise baseline",
                         **resume_quality(name, None, device, args.quick, tmp, model, opt,
                                          x, y, resume_steps)})
            print(json.dumps(rows[-1]))

    print()
    header = (f"{'setup':<22}{'kind':<26}{'file MB':>8}{'sync s':>8}{'bg blk s':>9}"
              f"{'max w diff':>11}{'dloss 1':>9}{'dacc 1':>8}{'dloss 3':>9}{'dacc 3':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['setup']:<22}{r['kind']:<26}{_f(r.get('file_mb'), '.2f'):>8}"
              f"{_f(r.get('sync_save_s'), '.3f'):>8}{_f(r.get('background_blocking_s'), '.3f'):>9}"
              f"{_f(r.get('max_weight_diff_after_load'), '.1e'):>11}"
              f"{_f(r.get('loss_diff_1'), '.4f'):>9}{_f(r.get('acc_diff_1'), '.1f'):>8}"
              f"{_f(r.get('loss_diff_3'), '.4f'):>9}{_f(r.get('acc_diff_3'), '.1f'):>8}")
    print(f"\ndloss/dacc 1: |loss| and |accuracy %| difference to an uninterrupted run after one "
          f"resume and {resume_steps} more steps; 3: after three consecutive resumes. "
          "noise baseline: an exact resume with 1e-6 relative weight noise (normal variation).")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"checkpoint_sizes_{detect_platform()}_{stamp}.json"
    out.write_text(json.dumps({"device": str(device), "gpu": gpu, "torch": torch.__version__,
                               "quick": args.quick, "train_steps": TRAIN_STEPS,
                               "resume_steps": resume_steps, "rows": rows}, indent=2))
    print(f"\nsaved {out}")


def _f(v, spec):
    return "-" if v is None else format(v, spec)


if __name__ == "__main__":
    main()
