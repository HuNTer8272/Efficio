"""Budget stop and resume with the compressed checkpoint modes. Prints PASS or FAIL.

    python benchmarks/verify_light_resume.py            # GPU if available
    python benchmarks/verify_light_resume.py --quick    # small run, CPU smoke test

Trains ResNet-18 (CIFAR stem) through efficio.Run on synthetic images with learnable labels:
1. an uninterrupted run to the end (reference),
then, for checkpoint_mode "light" and "smallest":
2. the same run stopped by the time budget in the middle of an epoch,
3. a fresh Run that resumes from that checkpoint and finishes.
Checks that the stop and resume happen in the right mode, that the weights right after
loading are exact (light) or within fp16 rounding (smallest), that the resumed run ends at
the same step, and that its final loss and accuracy are close to the reference. Also prints
the checkpoint lines of the report (mode, size, save time). Uses the installed efficio
package.
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
FP16_TOL = 1e-2          # max weight difference allowed after a smallest-mode load


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


def train(ckpt_dir, epochs, n, mode, stop_at_step=None, after_prepare=None):
    torch.manual_seed(0)
    loader = DataLoader(data(n), batch_size=64, shuffle=True, drop_last=True)
    model = resnet18()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run = efficio.Run(checkpoint_dir=ckpt_dir, checkpoint_mode=mode, verbose=False)
    model, opt = run.prepare(model, opt)
    if after_prepare is not None:
        after_prepare(model)
    for x, y in run.loop(loader, epochs=epochs):
        with run.autocast():
            loss = nn.functional.cross_entropy(model(x), y)
        run.backward(loss)
        run.step()
        if stop_at_step is not None and run.state.step == stop_at_step:
            run.budget.safety_margin_seconds = 0          # the time budget runs out now
            run.budget.total_seconds = max(run.budget.elapsed() - 1.0, 1e-6)
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        run.report()
    return run, model, report.getvalue()


def evaluate(model, n):
    """(loss, accuracy %) of the final model on the whole training set, in eval mode."""
    device = next(model.parameters()).device
    model.eval()
    total, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in DataLoader(data(n), batch_size=256):
            x, y = x.to(device), y.to(device)
            out = model(x).float()
            total += nn.functional.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            count += len(y)
    return total / count, 100.0 * correct / count


def weight_diff(a, b):
    return max((x.float() - y.float()).abs().max().item() for x, y in zip(a.values(), b.values())
               if x.is_floating_point())


def check_mode(mode, tmp, epochs, n, stop_at, ref_run, ref_loss, ref_acc):
    folder = os.path.join(tmp, mode)
    stopped, stopped_model, _ = train(folder, epochs, n, mode, stop_at_step=stop_at)
    saved_weights = {k: v.detach().clone() for k, v in stopped_model.state_dict().items()}
    latest = stopped.ckpt.latest()
    saved_mode = torch.load(latest, weights_only=False)["meta"]["mode"] if latest else None
    loaded = {}

    def capture(model):
        loaded["diff"] = weight_diff(model.state_dict(), saved_weights)

    resumed, model, report = train(folder, epochs, n, mode, after_prepare=capture)
    loss, acc = evaluate(model, n)
    load_ok = loaded["diff"] == 0.0 if mode == "light" else 0.0 < loaded["diff"] <= FP16_TOL
    print(f"\n=== checkpoint_mode={mode!r}")
    print(f"stopped at step {stopped.state.step} (epoch {stopped.state.epoch}, batch "
          f"{stopped.state.batch_in_epoch}) because: {stopped._stop_reason}")
    print(f"weights right after loading differ from the saved ones by at most "
          f"{loaded['diff']:.2e}")
    print(f"resumed from epoch/batch {resumed._resumed_from}, ended at step "
          f"{resumed.state.step} of {ref_run.state.step}")
    print(f"final loss {loss:.4f} (uninterrupted {ref_loss:.4f}), accuracy {acc:.1f}% "
          f"(uninterrupted {ref_acc:.1f}%)")
    print("".join(line + "\n" for line in report.splitlines()
                  if "checkpoint mode" in line or "save time" in line), end="")
    return {
        # the budget is set after step stop_at, so the stop happens at the next step
        "stopped by the budget mid-epoch": stopped.should_stop
                                           and stopped.state.step == stop_at + 1
                                           and stopped.state.batch_in_epoch > 0,
        f"{mode} checkpoint written": saved_mode == mode,
        "resumed from that checkpoint": resumed._resumed_from is not None
                                        and resumed.ckpt.meta.get("mode") == mode,
        ("weights loaded exactly" if mode == "light"
         else f"weights within fp16 rounding (<= {FP16_TOL})"): load_ok,
        "same final step": resumed.state.step == ref_run.state.step,
        "final loss close (within 10% or 0.05)": abs(loss - ref_loss)
                                                 <= max(0.05, 0.1 * ref_loss),
        "final accuracy close (within 3 points)": abs(acc - ref_acc) <= 3.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small run (CPU smoke test)")
    args = parser.parse_args()
    n, epochs = (256, 2) if args.quick else (4096, 3)
    steps_per_epoch = n // 64
    stop_at = steps_per_epoch + steps_per_epoch // 2              # middle of epoch 2
    print(f"efficio {efficio.__version__} from {os.path.dirname(efficio.__file__)}")
    print(f"torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0) if CUDA else 'none'}")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        ref_run, ref_model, _ = train(os.path.join(tmp, "ref"), epochs, n, "exact")
        ref_loss, ref_acc = evaluate(ref_model, n)
        for mode in ("light", "smallest"):
            checks = check_mode(mode, tmp, epochs, n, stop_at, ref_run, ref_loss, ref_acc)
            for name, ok in checks.items():
                print(f"{'PASS' if ok else 'FAIL'}  [{mode}] {name}")
                results.append(ok)
    print(f"\n{sum(results)} passed, {len(results) - sum(results)} failed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
