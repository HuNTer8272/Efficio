"""Final GPU check before a release. Prints PASS, FAIL or SKIP for each check.

    python benchmarks/verify_gpu_features.py

Checks:
1. run.clip_grad_norm with the real CUDA fp16 GradScaler: gradients with a known large norm
   are clipped to max_norm=1.0 in real (unscaled) units and match clip_grad_norm_ on an
   fp32 copy, without and with gradient accumulation of 2.            (CUDA only)
2. The ImageCache report tip: 2 epochs through Efficio on a plain Dataset that decodes
   512x512 PNGs on the CPU every epoch; run.report() must suggest ImageCache. (CUDA only)
3. A time-budget stop in the middle of training, then a resume with a fresh Run: the run
   ends at the expected step with the same weights as an uninterrupted run. (any device)

On a machine without CUDA the CUDA-only checks print SKIP. Exits with code 1 if any check
fails. Uses the installed efficio package (run it from the repo root).
"""

import contextlib
import copy
import io
import os
import sys
import tempfile
import types

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

import efficio

CUDA = torch.cuda.is_available()
results = []


def record(status, name, detail=""):
    results.append(status)
    print(f"{status:<4}  {name}" + (f": {detail}" if detail else ""), flush=True)


def new_run(ckpt_dir, **kwargs):
    kwargs.setdefault("verbose", False)
    return efficio.Run(checkpoint_dir=ckpt_dir, **kwargs)


# ---------------------------------------------------------------- 1. clip_grad_norm, fp16
D_IN, D_OUT, BATCH = 512, 512, 8


def _micro_batch(level):
    """Inputs and loss weight chosen so every weight gradient is exactly 0.5 * level: a
    known total norm of about 256 * level, while the gradients scaled by 65536 still fit
    in fp16 during the backward pass."""
    x = torch.full((BATCH, D_IN), float(level), device="cuda")
    return x, 1.0 / (2 * BATCH)                    # weight grad = c * BATCH * level


def _grads(model):
    return torch.cat([p.grad.detach().float().flatten() for p in model.parameters()])


def check_clip(accum):
    name = f"clip_grad_norm with the fp16 GradScaler (gradient accumulation {accum})"
    if not CUDA:
        return record("SKIP", name, "needs CUDA (the fp16 GradScaler only runs on a GPU)")
    torch.manual_seed(0)
    model = nn.Linear(D_IN, D_OUT)
    reference = copy.deepcopy(model).cuda()
    with tempfile.TemporaryDirectory() as tmp:
        run = new_run(tmp, resume=False, grad_accum_steps=accum, optimize_data=False,
                      channels_last=False)
        if run.hw.precision != "fp16":
            return record("SKIP", name, f"this GPU uses {run.hw.precision}, not fp16")
        model, optimizer = run.prepare(model, torch.optim.SGD(model.parameters(), lr=0.0))
        levels = [1.0, 0.5][:accum]
        total = None
        for i, level in enumerate(levels):
            x, c = _micro_batch(level)
            with run.autocast():
                loss = c * model(x).sum()
            run.backward(loss)
            total = run.clip_grad_norm(1.0)
            if i < accum - 1 and total is not None:
                return record("FAIL", name, "clipped before the last micro-batch")
            if i == accum - 1:
                clipped = _grads(model)
            run.step()
        scale = run.scaler.get_scale()

    for level in levels:                           # fp32 reference, no scaler
        x, c = _micro_batch(level)
        (c * reference(x).sum() / accum).backward()
    ref_total = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    ref = _grads(reference)

    norm_after = clipped.norm().item()
    ok = (total is not None and torch.isfinite(total)
          and abs(total.item() - ref_total.item()) <= 1e-3 * ref_total.item()
          and norm_after <= 1.0 + 1e-3
          and torch.allclose(clipped, ref, rtol=1e-3, atol=1e-7))
    before = f"{total.item():.2f}" if total is not None else "none"
    detail = (f"norm before {before} (fp32: {ref_total.item():.2f}), after {norm_after:.4f}, "
              f"max diff vs fp32 {(clipped - ref).abs().max().item():.2e}, "
              f"scaler scale {scale:.0f}")
    record("PASS" if ok else "FAIL", name, detail)


# ---------------------------------------------------------------- 2. ImageCache report tip
class ImageFiles(Dataset):
    """The usual approach: open, decode and resize every PNG on the CPU in every epoch."""

    def __init__(self, paths, labels, size):
        self.paths, self.labels, self.size = paths, labels, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as img:
            img = img.convert("L").resize((self.size, self.size), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        return x.unsqueeze(0).expand(3, -1, -1), torch.from_numpy(self.labels[i])


def check_image_cache_tip(count=640, epochs=2):
    name = "ImageCache report tip for image files decoded every epoch"
    if not CUDA:
        return record("SKIP", name, "needs CUDA (the data-wait tips are only given on a GPU)")
    # fake PNG data, as in the image benchmark
    import image_common as ic
    data_dir = os.path.join(tempfile.gettempdir(), "efficio_verify_images")
    args = types.SimpleNamespace(fake=True, fake_count=count, fake_size=512, images=None,
                                 labels=None, data_dir=data_dir, limit=None)
    paths, labels, _, _ = ic.load_dataset(args)
    torch.manual_seed(0)
    loader = DataLoader(ImageFiles(paths, labels, 224), batch_size=64, shuffle=True,
                        drop_last=True, num_workers=2)
    model = ic.make_model(labels.shape[1], pretrained=False)
    loss_fn = nn.BCEWithLogitsLoss()
    with tempfile.TemporaryDirectory() as tmp:
        run = new_run(tmp, resume=False)
        model, optimizer = run.prepare(model, ic.make_optimizer(model))
        for x, y in run.loop(loader, epochs=epochs):
            with run.autocast():
                loss = loss_fn(model(x), y)
            run.backward(loss)
            run.step()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            run.report()
    report = out.getvalue()
    wait = run.profiler.data_wait_pct
    # Look for the data-wait tip for image files. The setup-time tip also names ImageCache,
    # but that one doesn't count here.
    if "Decode them once into an ImageCache" in report:
        record("PASS", name, f"data-wait {wait:.0f}%, the report suggests ImageCache")
    else:
        record("FAIL", name, f"data-wait {wait:.0f}%, no ImageCache data-wait tip. "
                             f"Report:\n{report}")


# ---------------------------------------------------------------- 3. budget stop + resume
def _train(ckpt_dir, epochs, stop_after_step=None):
    """Train a small model through Run. With stop_after_step, let the time budget run out
    right after that step, like a session that is about to hit its limit."""
    torch.manual_seed(0)
    data = TensorDataset(torch.randn(2048, 32), torch.randint(0, 4, (2048,)))
    loader = DataLoader(data, batch_size=64, shuffle=True, drop_last=True)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))
    run = new_run(ckpt_dir)
    model, optimizer = run.prepare(model, torch.optim.AdamW(model.parameters(), lr=1e-3))
    for x, y in run.loop(loader, epochs=epochs):
        with run.autocast():
            loss = nn.functional.cross_entropy(model(x), y)
        run.backward(loss)
        run.step()
        if stop_after_step is not None and run.state.step == stop_after_step:
            run.budget.safety_margin_seconds = 0
            run.budget.total_seconds = max(run.budget.elapsed() - 1.0, 1e-6)  # already over
    return run, model


def check_budget_resume(epochs=3, stop_after_step=40):
    device = "cuda" if CUDA else "cpu"
    name = f"budget stop and resume with a fresh Run (on {device})"
    with tempfile.TemporaryDirectory() as tmp:
        full_run, full_model = _train(os.path.join(tmp, "full"), epochs)
        stopped, _ = _train(os.path.join(tmp, "split"), epochs, stop_after_step)
        resumed, model = _train(os.path.join(tmp, "split"), epochs)
    expected = full_run.state.step
    diff = max((a - b).abs().max().item()
               for a, b in zip(model.parameters(), full_model.parameters()))
    ok = (stopped.should_stop and stopped.state.step == stop_after_step + 1
          and resumed._resumed_from is not None and resumed.state.step == expected
          and diff <= 1e-4)
    record("PASS" if ok else "FAIL", name,
           f"stopped at step {stopped.state.step} ({stopped._stop_reason}), resumed from "
           f"epoch/batch {resumed._resumed_from}, ended at step {resumed.state.step} of "
           f"{expected}, max weight diff vs uninterrupted run {diff:.1e}")


def main():
    print(f"efficio {efficio.__version__} from {os.path.dirname(efficio.__file__)}")
    gpu = torch.cuda.get_device_name(0) if CUDA else "none"
    print(f"torch {torch.__version__}, CUDA {'yes' if CUDA else 'no'}, GPU: {gpu}\n")
    checks = [("clip_grad_norm (accumulation 1)", lambda: check_clip(1)),
              ("clip_grad_norm (accumulation 2)", lambda: check_clip(2)),
              ("ImageCache report tip", check_image_cache_tip),
              ("budget stop and resume", check_budget_resume)]
    for name, check in checks:
        try:
            check()
        except Exception as e:                    # a crash is a failure, keep checking
            record("FAIL", name, f"{type(e).__name__}: {e}")
    print(f"\n{results.count('PASS')} passed, {results.count('FAIL')} failed, "
          f"{results.count('SKIP')} skipped")
    sys.exit(1 if "FAIL" in results else 0)


if __name__ == "__main__":
    main()
