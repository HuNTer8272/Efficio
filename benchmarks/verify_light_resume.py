"""Budget stop and resume with the compressed checkpoint modes, with a noise baseline and a
diagnostic of the restored state. Prints PASS or FAIL.

    python benchmarks/verify_light_resume.py            # GPU if available
    python benchmarks/verify_light_resume.py --quick    # small run, CPU smoke test

Trains ResNet-18 (CIFAR stem) with AdamW through efficio.Run on synthetic images with
learnable labels. First an uninterrupted run (the reference), then for each variant the same
run stopped by the time budget in the middle of an epoch and resumed with a fresh Run:
- "exact": an exact checkpoint; must match the reference exactly (same device and code path)
- "noise": an exact checkpoint with 1e-6 relative noise added to the weights after loading;
  shows how far normal tiny differences drift by the end (the noise baseline)
- "light" and "smallest": the compressed modes

For every resume it prints a diagnostic of the state right after loading (per optimizer
tensor: max and mean relative error against the state that was saved, how many second-moment
values came back as zero or lost more than half their value; GradScaler scale and growth
tracker, scheduler LR and step count), the losses of the first 10 steps after the resume next
to the reference, and the final loss and accuracy. Light and smallest also pass only if their
final loss and accuracy gaps are at most 3x the noise baseline's gaps (at least 0.05 loss
and 3 accuracy points): long training amplifies tiny differences, so the noise baseline,
not a fixed threshold, says what is normal. Uses the installed efficio package.
"""

import argparse
import contextlib
import copy
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
NOISE = 1e-6             # relative weight noise of the noise baseline
LOSS_FLOOR, ACC_FLOOR = 0.05, 3.0   # smallest allowed final-gap limits (loss, accuracy points)
VARIANTS = ("exact", "noise", "light", "smallest")


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
    """Returns (run, model, {step: loss}, report text)."""
    torch.manual_seed(0)
    loader = DataLoader(data(n), batch_size=64, shuffle=True, drop_last=True)
    model = resnet18()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    total = epochs * (n // 64)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    run = efficio.Run(checkpoint_dir=ckpt_dir, checkpoint_mode=mode, verbose=False)
    model, opt = run.prepare(model, opt, sched)
    if after_prepare is not None:
        after_prepare(run, model)
    losses = {}
    for x, y in run.loop(loader, epochs=epochs):
        with run.autocast():
            loss = nn.functional.cross_entropy(model(x), y)
        run.backward(loss)
        run.step()
        losses[run.state.step] = loss.item()
        if stop_at_step is not None and run.state.step == stop_at_step:
            run.budget.safety_margin_seconds = 0          # the time budget runs out now
            run.budget.total_seconds = max(run.budget.elapsed() - 1.0, 1e-6)
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        run.report()
    return run, model, losses, report.getvalue()


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
    """Max difference between two state dicts; compared on the CPU (one may be on the GPU)."""
    return max((x.detach().float().cpu() - y.detach().float().cpu()).abs().max().item()
               for x, y in zip(a.values(), b.values()) if x.is_floating_point())


def snapshot(run, model):
    """Everything the resume should restore, on CPU."""
    return {"model": {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()
                      if v.is_floating_point()},
            "opt": copy.deepcopy({k: {n: t.detach().cpu().clone() if torch.is_tensor(t) else t
                                      for n, t in st.items()}
                                  for k, st in run.optimizer.state_dict()["state"].items()}),
            "scaler": run.scaler.state_dict(),
            "lr": run.scheduler.get_last_lr() if run.scheduler else None,
            "step": run.state.step}


def diagnose(saved, run, model):
    """Compare the state right after loading with what was saved. Returns printable lines and
    a dict of summary numbers."""
    lines, summary = [], {"v_zeroed": 0, "v_halved": 0, "v_total": 0, "max_rel_err": {}}
    now = run.optimizer.state_dict()["state"]
    for key in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
        max_rel, mean_rel, count = 0.0, 0.0, 0
        for pid, st in saved["opt"].items():
            if key not in st or not torch.is_tensor(st[key]):
                continue
            a = st[key].float()
            b = now[pid][key].detach().float().cpu()
            scale = a.abs().max().clamp_min(1e-30)
            rel = (b - a).abs() / scale               # error relative to the tensor's largest value
            max_rel = max(max_rel, rel.max().item())
            mean_rel += rel.mean().item()
            count += 1
            if key == "exp_avg_sq":
                nz = a > 0
                summary["v_total"] += int(nz.sum())
                summary["v_zeroed"] += int((nz & (b == 0)).sum())
                summary["v_halved"] += int((nz & (b < 0.5 * a)).sum())
        if count:
            summary["max_rel_err"][key] = max_rel
            lines.append(f"  {key:<16} max rel err {max_rel:.2e}, mean {mean_rel / count:.2e} "
                         f"(relative to each tensor's largest value)")
    if summary["v_total"]:
        total = summary["v_total"]
        zeroed, halved = summary["v_zeroed"], summary["v_halved"]
        lines.append(f"  exp_avg_sq values that came back as 0: {zeroed} of {total} "
                     f"({100 * zeroed / total:.1f}%); lost more than half their value: "
                     f"{halved} ({100 * halved / total:.1f}%)")
    sc_now, sc_saved = run.scaler.state_dict(), saved["scaler"]
    same_scaler = (sc_now.get("scale") == sc_saved.get("scale")
                   and sc_now.get("_growth_tracker") == sc_saved.get("_growth_tracker"))
    lr_now = run.scheduler.get_last_lr() if run.scheduler else None
    adam_steps = {float(st["step"]) for st in now.values() if "step" in st}
    lines.append(f"  GradScaler scale/growth tracker {sc_now.get('scale')}/"
                 f"{sc_now.get('_growth_tracker')} vs saved {sc_saved.get('scale')}/"
                 f"{sc_saved.get('_growth_tracker')}: {'match' if same_scaler else 'DIFFER'}")
    lines.append(f"  scheduler last LR {lr_now} vs saved {saved['lr']}: "
                 f"{'match' if lr_now == saved['lr'] else 'DIFFER'}")
    lines.append(f"  step {run.state.step} vs saved {saved['step']}; optimizer step counts "
                 f"{sorted(adam_steps)}")
    summary["exact_meta"] = (same_scaler and lr_now == saved["lr"]
                             and run.state.step == saved["step"])
    return lines, summary


def add_noise(model, rel):
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn(p.shape, generator=gen).to(p.device, p.dtype)
            p.add_(noise * p.abs().mean() * rel)


def run_variant(variant, tmp, epochs, n, stop_at, ref):
    mode = "exact" if variant in ("exact", "noise") else variant
    folder = os.path.join(tmp, variant)
    stopped, stopped_model, _, _ = train(folder, epochs, n, mode, stop_at_step=stop_at)
    saved = snapshot(stopped, stopped_model)
    latest = stopped.ckpt.latest()
    saved_mode = torch.load(latest, weights_only=False)["meta"]["mode"] if latest else None
    info = {}

    def after_prepare(run, model):
        info["diff"] = weight_diff({k: v for k, v in model.state_dict().items()
                                    if v.is_floating_point()}, saved["model"])
        info["lines"], info["summary"] = diagnose(saved, run, model)
        if variant == "noise":
            add_noise(model, NOISE)

    resumed, model, losses, report = train(folder, epochs, n, mode, after_prepare=after_prepare)
    loss, acc = evaluate(model, n)
    ref_losses, ref_loss, ref_acc = ref["losses"], ref["loss"], ref["acc"]

    print(f"\n=== {variant} (checkpoint_mode={mode!r})")
    print(f"stopped at step {stopped.state.step} (epoch {stopped.state.epoch}, batch "
          f"{stopped.state.batch_in_epoch}); resumed from epoch/batch {resumed._resumed_from}; "
          f"ended at step {resumed.state.step} of {ref['steps']}")
    print(f"weights right after loading: max difference {info['diff']:.2e}")
    print("state right after loading:")
    print("\n".join(info["lines"]))
    first = sorted(losses)[:10]
    print("loss for the first 10 steps after resume (resumed / uninterrupted):")
    print("  " + "  ".join(f"{s}: {losses[s]:.4f}/{ref_losses[s]:.4f}" for s in first))
    print(f"final loss {loss:.4f} (uninterrupted {ref_loss:.4f}, gap {abs(loss - ref_loss):.4f}); "
          f"accuracy {acc:.1f}% (uninterrupted {ref_acc:.1f}%, gap {abs(acc - ref_acc):.1f})")
    print("".join(line + "\n" for line in report.splitlines()
                  if "checkpoint mode" in line or "save time" in line), end="")

    checks = {
        "stopped by the budget mid-epoch": stopped.should_stop
                                           and stopped.state.step == stop_at + 1
                                           and stopped.state.batch_in_epoch > 0,
        f"{mode} checkpoint written": saved_mode == mode,
        "resumed from that checkpoint": resumed._resumed_from is not None,
        "same final step": resumed.state.step == ref["steps"],
        "scaler, scheduler and step restored exactly": info["summary"]["exact_meta"],
    }
    if variant in ("exact", "noise", "light"):
        checks["weights loaded exactly"] = info["diff"] == 0.0
    else:
        checks[f"weights within fp16 rounding (<= {FP16_TOL})"] = 0.0 < info["diff"] <= FP16_TOL
    if variant == "exact":
        checks["final loss identical to the uninterrupted run"] = loss == ref_loss
    if variant in ("light", "smallest"):
        total = info["summary"]["v_total"] or 1
        checks["no second-moment value lost more than half (<= 0.1% of values)"] = \
            info["summary"]["v_halved"] / total <= 0.001
        # Final result against the noise baseline: at most 3x its gap, with a floor so a
        # lucky near-zero noise gap can't fail every run.
        noise = ref["noise_gap"]
        loss_limit = max(3 * noise["loss_gap"], LOSS_FLOOR)
        acc_limit = max(3 * noise["acc_gap"], ACC_FLOOR)
        checks[f"final loss gap <= {loss_limit:.4f} (3x noise gap, floor {LOSS_FLOOR})"] = \
            abs(loss - ref_loss) <= loss_limit
        checks[f"final accuracy gap <= {acc_limit:.1f} points (3x noise gap, floor "
               f"{ACC_FLOOR})"] = abs(acc - ref_acc) <= acc_limit
    return checks, {"loss_gap": abs(loss - ref_loss), "acc_gap": abs(acc - ref_acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small run (CPU smoke test)")
    parser.add_argument("--n", type=int, help="number of samples (default 4096, quick 256)")
    parser.add_argument("--epochs", type=int, help="epochs (default 3, quick 2)")
    args, _ = parser.parse_known_args()          # a notebook kernel passes its own arguments
    n, epochs = (256, 2) if args.quick else (4096, 3)
    n, epochs = args.n or n, args.epochs or epochs
    steps_per_epoch = n // 64
    stop_at = steps_per_epoch + steps_per_epoch // 2              # middle of epoch 2
    print(f"efficio {efficio.__version__} from {os.path.dirname(efficio.__file__)}")
    print(f"torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0) if CUDA else 'none'}")

    results, gaps = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        ref_run, ref_model, ref_losses, _ = train(os.path.join(tmp, "ref"), epochs, n, "exact")
        ref_loss, ref_acc = evaluate(ref_model, n)
        ref = {"losses": ref_losses, "loss": ref_loss, "acc": ref_acc,
               "steps": ref_run.state.step}
        print(f"uninterrupted: {ref['steps']} steps, final loss {ref_loss:.4f}, "
              f"accuracy {ref_acc:.1f}%")
        for variant in VARIANTS:
            checks, gaps[variant] = run_variant(variant, tmp, epochs, n, stop_at, ref)
            if variant == "noise":
                ref["noise_gap"] = gaps["noise"]   # judged against by light and smallest
            for name, ok in checks.items():
                print(f"{'PASS' if ok else 'FAIL'}  [{variant}] {name}")
                results.append(ok)
    print("\nfinal gaps to the uninterrupted run (loss / accuracy points):")
    for variant, g in gaps.items():
        print(f"  {variant:<9} {g['loss_gap']:.4f} / {g['acc_gap']:.1f}")
    print(f"\n{sum(results)} passed, {len(results) - sum(results)} failed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    exit_code = main()
    if "ipykernel" not in sys.modules:             # pasted into a notebook cell: no exit
        sys.exit(exit_code)
