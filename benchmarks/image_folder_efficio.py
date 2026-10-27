"""Same benchmark as image_folder_baseline.py, trained through Efficio.

The images are decoded and resized once into an ImageCache (timed separately and reused in
later runs), then read as uint8 arrays; conversion and the same augmentation run on the GPU.
The run also tests budget stop and resume: after 2 full epochs the time budget is set to run
out halfway through epoch 3, the run saves and stops, and a fresh Run (new model and
optimizer objects) resumes from the checkpoint and finishes. Run from the repo root
(after `pip install -e .`):
    python benchmarks/image_folder_efficio.py --fake --epochs 6
    python benchmarks/image_folder_efficio.py --images DIR --labels labels.csv
"""

import argparse
import os
import tempfile
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

import efficio
from efficio.data import CachedImageDataset, ImageCache
from efficio.speed import gpu_augment as A

import image_common as ic
from common import GpuUtilSampler, save_results, sync

STOP_AFTER_EPOCHS = 2        # the budget runs out halfway through the epoch after these


def train_phase(run, model, optimizer, loader, epochs, augment, loss_fn, stop_mid_epoch):
    """Train until run.loop ends. Returns [(epoch, seconds, complete)]: epochs that were
    split by the budget stop or the resume are marked incomplete."""
    times = []
    sync()
    start = time.perf_counter()
    epoch, partial = run.state.epoch, run.state.batch_in_epoch > 0
    for x, y in run.loop(loader, epochs=epochs, gpu_transform=efficio.on_inputs(augment)):
        if run.state.epoch != epoch:      # the previous epoch just ended
            sync()
            now = time.perf_counter()
            times.append((epoch, now - start, not partial))
            print(f"epoch {epoch + 1}/{epochs}: {now - start:.1f} s"
                  + ("" if not partial else " (resumed part)"))
            start, epoch, partial = now, run.state.epoch, False
            complete = sum(1 for t in times if t[2])
            if stop_mid_epoch and complete == STOP_AFTER_EPOCHS:
                # A real budget stop: let the budget run out halfway through this epoch
                run.budget.safety_margin_seconds = 0
                run.budget.total_seconds = run.budget.elapsed() + 0.5 * times[-1][1]
        with run.autocast():
            loss = loss_fn(model(x), y)
        run.backward(loss)
        run.step()
    sync()
    stopped = run.should_stop
    times.append((epoch, time.perf_counter() - start, not partial and not stopped))
    print(f"epoch {epoch + 1}/{epochs}: {times[-1][1]:.1f} s"
          + (" (stopped by the budget)" if stopped else ""))
    return times


def main():
    parser = argparse.ArgumentParser()
    ic.add_args(parser)
    parser.add_argument("--io-threads", type=int, default=32,
                        help="parallel file reads while building the cache")
    parser.add_argument("--no-resume-test", action="store_true",
                        help="train straight through without the budget stop and resume")
    args = parser.parse_args()
    s = args.size

    train_paths, train_labels, val_paths, val_labels = ic.load_dataset(args)
    # Decode and resize once. A finished cache with the same settings is reused (0 s).
    cache_dir = os.path.join(args.data_dir, f"cache_{s}" + ("_fake" if args.fake else ""))
    start = time.perf_counter()
    train_cache = ImageCache.build(train_paths, train_labels, os.path.join(cache_dir, "train"),
                                   size=(s, s), grayscale=True, io_threads=args.io_threads)
    ImageCache.build(val_paths, val_labels, os.path.join(cache_dir, "val"),
                     size=(s, s), grayscale=True, io_threads=args.io_threads)
    cache_build_s = time.perf_counter() - start

    to_model = [A.ToFloat(), A.RepeatChannels(3)]
    normalize = A.Normalize(ic.IMAGENET_MEAN, ic.IMAGENET_STD)
    augment = A.Compose(to_model + [
        A.RandomResizedCrop(s, scale=ic.CROP_SCALE, ratio=ic.CROP_RATIO),
        A.RandomAffine(ic.AFFINE_DEGREES, translate=ic.AFFINE_TRANSLATE, scale=ic.AFFINE_SCALE),
        A.RandomHorizontalFlip(),
        A.RandomBrightnessContrast(ic.BRIGHTNESS, ic.CONTRAST),
        normalize])
    for_eval = A.Compose(to_model + [normalize])

    train_loader = DataLoader(CachedImageDataset(os.path.join(cache_dir, "train")),
                              batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=ic.NUM_WORKERS)
    val_loader = DataLoader(CachedImageDataset(os.path.join(cache_dir, "val")),
                            batch_size=2 * args.batch_size, num_workers=ic.NUM_WORKERS)
    loss_fn = nn.BCEWithLogitsLoss()
    ckpt_dir = tempfile.mkdtemp(prefix="efficio_imgbench_")   # every benchmark run starts fresh
    resume_test = not args.no_resume_test and args.epochs > STOP_AFTER_EPOCHS

    def new_run():
        torch.manual_seed(0)              # same model initialization as the baseline
        model = ic.make_model(train_labels.shape[1], args.pretrained)
        optimizer = ic.make_optimizer(model)
        run = efficio.Run(checkpoint_dir=ckpt_dir)
        model, optimizer = run.prepare(model, optimizer)
        return run, model, optimizer

    times, data_wait_s, train_s = [], 0.0, 0.0
    with GpuUtilSampler() as sampler:
        run, model, optimizer = new_run()
        times += train_phase(run, model, optimizer, train_loader, args.epochs, augment,
                             loss_fn, stop_mid_epoch=resume_test)
        data_wait_s += run.profiler.data_wait_s
        train_s += run.profiler.train_s
        stopped_at = (run.state.epoch, run.state.batch_in_epoch, run.state.step)
        if resume_test:
            run.report()
            print(f"--- budget stop at epoch {stopped_at[0] + 1}, batch {stopped_at[1]}; "
                  "resuming with a fresh Run ---")
            run, model, optimizer = new_run()
            times += train_phase(run, model, optimizer, train_loader, args.epochs, augment,
                                 loss_fn, stop_mid_epoch=False)
            data_wait_s += run.profiler.data_wait_s
            train_s += run.profiler.train_s

    expected_steps = args.epochs * len(train_loader)
    if resume_test:
        resume = {"tested": True, "stopped_at_epoch_batch_step": list(stopped_at),
                  "resumed_from": list(run._resumed_from or ()), "final_step": run.state.step,
                  "expected_step": expected_steps, "ok": run.state.step == expected_steps}
    else:                                    # no budget stop: nothing to report
        resume = {"tested": False, "stopped_at_epoch_batch_step": [], "resumed_from": [],
                  "final_step": None, "expected_step": None, "ok": None}

    scores, targets = [], []
    with run.evaluate_mode():
        for x, y in val_loader:
            x = for_eval(x.to(run.device, non_blocking=True))
            scores.append(torch.sigmoid(model(x).float()).cpu())
            targets.append(y)
    scores, targets = torch.cat(scores), torch.cat(targets)
    run.report()
    complete = [t for _, t, ok in times if ok]
    if resume_test:
        print(f"resume check: final step {run.state.step}, expected {expected_steps} -> "
              f"{'OK' if resume['ok'] else 'MISMATCH'}")
    save_results("imagefolder-efficio", complete, len(train_loader) * args.batch_size,
                 sampler.average, ic.label_accuracy(scores, targets),
                 extra={"benchmark": ic.BENCHMARK, "precision": run.hw.precision,
                        "batch_size": args.batch_size, "image_size": s,
                        "num_workers": ic.NUM_WORKERS,
                        "train_images": len(train_paths), "fake_data": args.fake,
                        "data_wait_pct": round(100 * data_wait_s / train_s, 1) if train_s else 0.0,
                        "val_auroc": round(ic.mean_auroc(scores, targets), 4),
                        "cache_build_s": round(cache_build_s, 1),
                        "cache_build_stats": train_cache.build_stats,   # None when reused
                        "epochs_all_s": [[e + 1, round(t, 3), ok] for e, t, ok in times],
                        "resume": resume, "optimizations": run.optimizations})


if __name__ == "__main__":          # needed for DataLoader workers on Windows/macOS
    main()
