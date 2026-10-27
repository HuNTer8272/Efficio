"""Rebuild a DataLoader so the GPU waits less for data."""

import multiprocessing
import os
import sys

import torch
from torch.utils.data import DataLoader, IterableDataset, RandomSampler

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


def _rebuild(loader, **changes):
    """A new DataLoader with the same settings as `loader`, except `changes`."""
    kwargs = dict(
        batch_size=loader.batch_size,
        sampler=loader.sampler,                     # same sampler -> same shuffle behavior
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        generator=loader.generator,
    )
    if loader.num_workers > 0:
        kwargs.update(persistent_workers=loader.persistent_workers,
                      prefetch_factor=loader.prefetch_factor,
                      multiprocessing_context=loader.multiprocessing_context)
    kwargs.update(changes)
    if kwargs["num_workers"] == 0:
        for key in ("persistent_workers", "prefetch_factor", "multiprocessing_context"):
            kwargs.pop(key, None)
    return DataLoader(loader.dataset, **kwargs)


def default_shuffle_seed(loader):
    """Base seed for seeded_shuffle: the user's DataLoader generator seed if they gave one,
    else the global torch seed (set by torch.manual_seed). Draws nothing from any RNG."""
    gen = getattr(loader.sampler, "generator", None) or loader.generator
    return (gen.initial_seed() if gen is not None else torch.initial_seed()) % 2**63


def seeded_shuffle(loader, base_seed=None):
    """Make a shuffling DataLoader's order depend only on base_seed and the epoch, so a resumed
    epoch is shuffled exactly like the interrupted one (the same idea as GPULoader).

    Returns (loader, set_epoch). Call set_epoch(epoch) before iterating each epoch; the seed
    in use is set_epoch.base_seed (base_seed=None picks default_shuffle_seed). Returns
    (loader, None) when the loader does not shuffle with a plain RandomSampler.

    The sampler gets its own generator (seed + epoch) for the order, and the loader gets a
    second one (seed + epoch + offset) for the worker seeds, so neither draws from the global
    RNG. Random augmentation inside DataLoader workers is still not reproduced exactly after
    a resume: persistent workers keep their own RNG from one epoch to the next."""
    if not isinstance(loader, DataLoader) or type(loader.sampler) is not RandomSampler:
        return loader, None
    if base_seed is None:
        base_seed = default_shuffle_seed(loader)
    old = loader.sampler
    order_gen, worker_gen = torch.Generator(), torch.Generator()
    sampler = RandomSampler(old.data_source, replacement=old.replacement,
                            num_samples=getattr(old, "_num_samples", None), generator=order_gen)
    new = _rebuild(loader, sampler=sampler, generator=worker_gen)

    def set_epoch(epoch):
        order_gen.manual_seed((base_seed + epoch) % 2**63)
        worker_gen.manual_seed((base_seed + epoch + 0x9E3779B97F4A7C15) % 2**63)

    set_epoch.base_seed = base_seed
    return new, set_epoch


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

    settings = dict(num_workers=workers, pin_memory=pin)
    if workers > 0:
        settings.update(persistent_workers=True, prefetch_factor=PREFETCH_FACTOR,
                        multiprocessing_context=loader.multiprocessing_context)
    new = _rebuild(loader, **settings)

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
