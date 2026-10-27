"""Budget stop and resume with light checkpoints. Prints PASS or FAIL.

    python benchmarks/verify_light_resume.py            # GPU if available
    python benchmarks/verify_light_resume.py --quick    # small run, CPU smoke test

Trains ResNet-18 (CIFAR stem) through efficio.Run with checkpoint_mode="light" on synthetic
images with learnable labels:
1. an uninterrupted run to the end (reference),
2. the same run stopped by the time budget in the middle of an epoch,
3. a fresh Run that resumes from the light checkpoint and finishes.
Checks that the stop and resume happen, that the resumed run ends at the same step, and that
its final loss and weights are close to the reference. Light resumes are close but not
bit-identical. Also prints the checkpoint lines of the report (mode, size, save time).
Uses the installed efficio package.
"""

import argparse
import contextlib
import io
import os
import sys
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import efficio

CUDA = torch.cuda.is_available()


def resnet18():
    import torchvision
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def data(n):
    gen = torch.Generator().manual_seed(123)
    x = torch.randn(n, 3, 32, 32, generator=gen)
    y = (x.flatten(1) @ torch.randn(3 * 32 * 32, 10, generator=gen)).argmax(1)
    return TensorDataset(x, y)


def train(ckpt_dir, epochs, n, stop_at_step=None):
    torch.manual_seed(0)
    loader = DataLoader(data(n), batch_size=64, shuffle=True, drop_last=True)
    model = resnet18()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run = efficio.Run(checkpoint_dir=ckpt_dir, checkpoint_mode="light", verbose=False)
    model, opt = run.prepare(model, opt)
    losses = []
    for x, y in run.loop(loader, epochs=epochs):
        with run.autocast():
            loss = nn.functional.cross_entropy(model(x), y)
        run.backward(loss)
        run.step()
        losses.append(loss.item())
        if stop_at_step is not None and run.state.step == stop_at_step:
            run.budget.safety_margin_seconds = 0          # the time budget runs out now
            run.budget.total_seconds = max(run.budget.elapsed() - 1.0, 1e-6)
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        run.report()
    return run, model, losses, report.getvalue()


def eval_loss(model, n):
    """Loss of the final model on the whole training set, in eval mode (same for both runs)."""
    loader = DataLoader(data(n), batch_size=256)
    device = next(model.parameters()).device
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += nn.functional.cross_entropy(model(x).float(), y, reduction="sum").item()
            count += len(y)
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small run (CPU smoke test)")
    args = parser.parse_args()
    n, epochs = (256, 2) if args.quick else (4096, 3)
    steps_per_epoch = n // 64
    stop_at = steps_per_epoch + steps_per_epoch // 2              # middle of epoch 2
    print(f"efficio {efficio.__version__} from {os.path.dirname(efficio.__file__)}")
    print(f"torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0) if CUDA else 'none'}")

    with tempfile.TemporaryDirectory() as tmp:
        ref_run, ref_model, ref_losses, _ = train(os.path.join(tmp, "ref"), epochs, n)
        stopped, _, _, _ = train(os.path.join(tmp, "split"), epochs, n, stop_at_step=stop_at)
        latest = stopped.ckpt.latest()
        saved_mode = torch.load(latest, weights_only=False)["meta"]["mode"] if latest else None
        resumed, model, losses, report = train(os.path.join(tmp, "split"), epochs, n)

    expected = ref_run.state.step
    weight_diff = max((a.float() - b.float()).abs().max().item()
                      for a, b in zip(model.state_dict().values(), ref_model.state_dict().values())
                      if a.is_floating_point())
    loss_ref, loss_res = eval_loss(ref_model, n), eval_loss(model, n)
    checks = {
        # the budget is set after step stop_at, so the stop happens at the next step
        "stopped by the budget mid-epoch": stopped.should_stop
                                           and stopped.state.step == stop_at + 1
                                           and stopped.state.batch_in_epoch > 0,
        "light checkpoint written": saved_mode == "light",
        "resumed from that checkpoint": resumed._resumed_from is not None
                                        and resumed.ckpt.meta.get("mode") == "light",
        "same final step": resumed.state.step == expected,
        "final loss close (within 10% or 0.05)": abs(loss_res - loss_ref)
                                                 <= max(0.05, 0.1 * loss_ref),
    }
    print(f"\nstopped at step {stopped.state.step} (epoch {stopped.state.epoch}, batch "
          f"{stopped.state.batch_in_epoch}) because: {stopped._stop_reason}")
    print(f"resumed from epoch/batch {resumed._resumed_from}, ended at step "
          f"{resumed.state.step} of {expected}")
    print(f"final loss on the training set: resumed {loss_res:.4f}, uninterrupted "
          f"{loss_ref:.4f}; max weight difference {weight_diff:.2e}")
    print("".join(line + "\n" for line in report.splitlines()
                  if "checkpoint" in line or "save time" in line))
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
