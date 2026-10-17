"""Rebuild a DataLoader so the GPU waits less for data. Phase 2 (Person B)."""

import multiprocessing
import os
import sys

import torch
from torch.utils.data import DataLoader, IterableDataset

MAX_WORKERS = 8
PREFETCH_FACTOR = 4


def _cpu_count():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _can_add_workers():
    """Adding workers is only safe with the "fork" start method (Linux: Colab, Kaggle).
    With "spawn" (Windows, macOS) the dataset must be picklable and scripts need an
    `if __name__ == "__main__":` guard, so we don't switch it on behind the user's back."""
    method = multiprocessing.get_start_method(allow_none=True)
    if method is None:
        method = "fork" if sys.platform.startswith("linux") else "spawn"
    return method == "fork"


def optimize_loader(loader, hw):
    """Returns (loader, changes). `changes` is a list of short descriptions. Entries that
    start with "unchanged", "already" or "note" describe something that was not changed."""
    if not isinstance(loader, DataLoader):
        return loader, ["unchanged: not a torch DataLoader"]
    if isinstance(loader.dataset, IterableDataset):
        return loader, ["unchanged: IterableDataset (more workers would duplicate samples)"]
    if loader.batch_size is None and loader.batch_sampler is not None:
        return loader, ["unchanged: custom batch_sampler"]

    old_workers = loader.num_workers
    if old_workers > 0:
        workers = old_workers                       # the user chose this, keep it
    elif _can_add_workers():
        workers = min(_cpu_count(), MAX_WORKERS)
    else:
        workers = 0
    pin = hw.device == "cuda"

    kwargs = dict(
        batch_size=loader.batch_size,
        sampler=loader.sampler,                     # same sampler -> same shuffle behavior
        num_workers=workers,
        collate_fn=loader.collate_fn,
        pin_memory=pin,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        generator=loader.generator,
    )
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=PREFETCH_FACTOR,
                      multiprocessing_context=loader.multiprocessing_context)
    new = DataLoader(loader.dataset, **kwargs)

    changes = []
    if workers != old_workers:
        changes.append(f"num_workers {old_workers} -> {workers}")
    elif old_workers == 0 and not _can_add_workers():
        changes.append("note: kept num_workers 0 (set num_workers yourself on Windows/macOS)")
    if pin and not loader.pin_memory:
        changes.append("pin_memory on")
    if workers > 0 and not loader.persistent_workers:
        changes.append("persistent_workers on")
    if workers > 0 and loader.prefetch_factor != PREFETCH_FACTOR:
        changes.append(f"prefetch_factor {PREFETCH_FACTOR}")
    if not any(not c.startswith("note") for c in changes):
        changes.insert(0, "already optimal")
    return new, changes
