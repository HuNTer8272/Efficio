"""Cross-platform resume through checkpoint storage: start on one machine, finish on another.
Prints PASS or FAIL.

    # phase start (e.g. Kaggle): train until the time budget stops it, upload the checkpoint
    python benchmarks/verify_cloud_resume.py --phase start --storage hf://USER/efficio-verify
    # phase finish (e.g. Colab): download the checkpoint, resume and finish; then compare
    python benchmarks/verify_cloud_resume.py --phase finish --storage hf://USER/efficio-verify
    # both phases in one process (CPU smoke test with a local folder as the storage)
    python benchmarks/verify_cloud_resume.py --phase both --storage ./verify_store --quick

On Colab, run it with %run in a notebook cell (not !python), so Efficio can read the HF_TOKEN
secret; Colab secrets are only available inside the notebook kernel.

Trains ResNet-18 (CIFAR stem) with AdamW on synthetic images with learnable labels, the same
setup as verify_light_resume.py. Phase start runs with resume=False, so old checkpoints in
the storage are never resumed by mistake; by default the time budget is made to run out in
the middle of epoch 2 (--budget 11m uses a real time budget instead), and the run settings
are stored next to the checkpoints for phase finish.

Phase finish uses an empty local folder, so the run must download the checkpoint. It checks:
the run resumed from the stored checkpoint at the stop step, the sha256 of the downloaded
file matches the manifest, it ends at the same final step as an uninterrupted run, and the
final loss and accuracy gaps to an uninterrupted run on this machine are at most 3x the
gaps of a noise baseline (the same run stopped and resumed locally at the same step with 1e-6
relative noise added to the weights), with floors of 0.05 loss and 3 accuracy points. The
uninterrupted run is the reference because the two halves ran on different machines.
"""

import argparse
import json
import os
import sys
import tempfile

import torch
from torch.utils.data import DataLoader

import efficio
from efficio.storage import open_storage, sha256_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # also under %run
from verify_light_resume import (  # noqa: E402
    ACC_FLOOR,
    LOSS_FLOOR,
    NOISE,
    add_noise,
    data,
    evaluate,
    resnet18,
)

SETTINGS = "verify_cloud_resume.json"
BATCH = 64


def train(ckpt_dir, cfg, storage=None, resume=True, stop_at=None, after_prepare=None,
          budget=None):
    """Returns the finished (or stopped) run and its model."""
    torch.manual_seed(0)
    loader = DataLoader(data(cfg["n"]), batch_size=BATCH, shuffle=True, drop_last=True)
    model = resnet18()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["total_steps"])
    run = efficio.Run(checkpoint_dir=ckpt_dir, checkpoint_mode=cfg["mode"], storage=storage,
                      resume=resume, budget=budget, verbose=storage is not None)
    if cfg["deterministic"]:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model, opt = run.prepare(model, opt, sched)
    if after_prepare is not None:
        after_prepare(run, model)
    for x, y in run.loop(loader, epochs=cfg["epochs"]):
        with run.autocast():
            loss = torch.nn.functional.cross_entropy(model(x), y)
        run.backward(loss)
        run.step()
        if stop_at is not None and run.state.step == stop_at:
            run.budget.safety_margin_seconds = 0          # the time budget runs out now
            run.budget.total_seconds = max(run.budget.elapsed() - 1.0, 1e-6)
    return run, model


def report(checks):
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return list(checks.values())


def phase_start(args, cfg, tmp):
    stop_at = None if args.budget else cfg["steps_per_epoch"] + cfg["steps_per_epoch"] // 2
    print(f"\n=== phase start: training until the time budget stops it, storage {args.storage}")
    run, _ = train(os.path.join(tmp, "start"), cfg, storage=args.storage, resume=False,
                   stop_at=stop_at, budget=args.budget)
    entry = run.remote.latest()
    run.remote.storage.upload(_write_json(tmp, cfg), SETTINGS)
    status = (run._final_upload or (None, None))[1]
    print(f"stopped at step {run.state.step} of {cfg['total_steps']} (epoch {run.state.epoch}, "
          f"batch {run.state.batch_in_epoch}); latest in storage: "
          f"{entry and entry['file']} (step {entry and entry['step']})")
    return report({
        "stopped by the time budget mid-run": run.should_stop
                                              and 0 < run.state.step < cfg["total_steps"],
        "stop checkpoint upload confirmed before training returned": status == "ok",
        "manifest's latest checkpoint is the stop checkpoint": entry is not None
                                                               and entry["step"] == run.state.step,
    })


def _write_json(tmp, cfg):
    path = os.path.join(tmp, SETTINGS)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return path


def _read_settings(storage, tmp):
    path = os.path.join(tmp, SETTINGS)
    open_storage(storage).download(SETTINGS, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def phase_finish(args, tmp):
    cfg = _read_settings(args.storage, tmp)
    print(f"\n=== phase finish: settings from phase start: {cfg}")
    info = {}

    def after_prepare(run, model):
        info["downloaded"] = run.remote.stats["downloaded"]
        info["resumed_step"] = run.state.step
        path = run.ckpt.latest()
        info["sha256"] = sha256_file(path) if path else None

    resumed, model = train(os.path.join(tmp, "finish"), cfg, storage=args.storage,
                           after_prepare=after_prepare)
    entry = info["downloaded"] or {}
    loss, acc = evaluate(model, cfg["n"])

    print("\n=== uninterrupted run on this machine (the reference)")
    ref_run, ref_model = train(os.path.join(tmp, "reference"), cfg)
    ref_loss, ref_acc = evaluate(ref_model, cfg["n"])

    stop = entry.get("step")
    print(f"\n=== noise baseline: stopped at step {stop} on this machine, resumed with "
          f"{NOISE:g} relative weight noise")
    noise_dir = os.path.join(tmp, "noise")
    stopped, _ = train(noise_dir, cfg, stop_at=stop - 1 if stop else None)
    _, noise_model = train(noise_dir, cfg, after_prepare=lambda run, m: add_noise(m, NOISE))
    noise_loss, noise_acc = evaluate(noise_model, cfg["n"])

    loss_gap, acc_gap = abs(loss - ref_loss), abs(acc - ref_acc)
    noise_loss_gap, noise_acc_gap = abs(noise_loss - ref_loss), abs(noise_acc - ref_acc)
    loss_limit = max(3 * noise_loss_gap, LOSS_FLOOR)
    acc_limit = max(3 * noise_acc_gap, ACC_FLOOR)
    here = resumed.hw.platform
    print(f"\ncheckpoint {entry.get('file')} saved on {entry.get('platform')} at "
          f"{entry.get('time')}, {entry.get('mode')} mode; resumed here on {here}")
    if entry.get("platform") == here:
        print("note: both phases ran on the same platform; for the cross-platform check run "
              "phase start on Kaggle and phase finish on Colab (or the other way round)")
    print(f"resumed at step {info['resumed_step']}, ended at step {resumed.state.step}; "
          f"uninterrupted ended at {ref_run.state.step}")
    print("final loss / accuracy:")
    print(f"  uninterrupted  {ref_loss:.4f} / {ref_acc:.1f}%")
    print(f"  resumed        {loss:.4f} / {acc:.1f}%  (gap {loss_gap:.4f} / {acc_gap:.1f})")
    print(f"  noise baseline {noise_loss:.4f} / {noise_acc:.1f}%  "
          f"(gap {noise_loss_gap:.4f} / {noise_acc_gap:.1f})")
    return report({
        "resumed from the stored checkpoint (downloaded, local folder was empty)":
            bool(entry) and resumed._resumed_from is not None,
        "resumed at the stop step": info["resumed_step"] == stop,
        "sha256 of the downloaded file matches the manifest":
            info["sha256"] is not None and info["sha256"] == entry.get("sha256"),
        "same final step as the uninterrupted run":
            resumed.state.step == ref_run.state.step == cfg["total_steps"],
        "noise baseline stopped at the same step": stopped.state.step == stop,
        f"final loss gap <= {loss_limit:.4f} (3x noise gap, floor {LOSS_FLOOR})":
            loss_gap <= loss_limit,
        f"final accuracy gap <= {acc_limit:.1f} points (3x noise gap, floor {ACC_FLOOR})":
            acc_gap <= acc_limit,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("start", "finish", "both"), required=True)
    parser.add_argument("--storage", required=True,
                        help="hf://<user>/<repo>, or a folder such as a mounted Drive folder")
    parser.add_argument("--mode", default="exact", choices=("exact", "light", "smallest"))
    parser.add_argument("--budget", help='phase start: a real time budget, e.g. "11m" '
                        "(stops 10 minutes before its end); default: run out mid-epoch 2")
    parser.add_argument("--deterministic", action="store_true",
                        help="deterministic cuDNN (slower; the gap rule works without it)")
    parser.add_argument("--quick", action="store_true", help="small run (CPU smoke test)")
    parser.add_argument("--n", type=int, help="number of samples (default 4096, quick 256)")
    parser.add_argument("--epochs", type=int, help="epochs (default 3, quick 2)")
    args, _ = parser.parse_known_args()          # a notebook kernel passes its own arguments
    n, epochs = (256, 2) if args.quick else (4096, 3)
    n, epochs = args.n or n, args.epochs or epochs
    cfg = {"n": n, "epochs": epochs, "mode": args.mode, "deterministic": args.deterministic,
           "steps_per_epoch": n // BATCH, "total_steps": epochs * (n // BATCH)}
    print(f"efficio {efficio.__version__} from {os.path.dirname(efficio.__file__)}")
    print(f"torch {torch.__version__}, GPU: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        if args.phase in ("start", "both"):
            results += phase_start(args, cfg, tmp)
        if args.phase in ("finish", "both"):
            results += phase_finish(args, tmp)
    print(f"\n{sum(results)} passed, {len(results) - sum(results)} failed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    exit_code = main()
    if "ipykernel" not in sys.modules:             # run in a notebook cell: no exit
        sys.exit(exit_code)
