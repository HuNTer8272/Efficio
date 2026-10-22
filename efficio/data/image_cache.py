"""Decode and resize an image dataset once, then train from a fast uint8 array. Phase 2 (Person B).

On Colab and Kaggle, 2-4 CPU cores decoding and resizing PNG/JPEG files cannot keep a GPU
busy. ImageCache.build does that work once, with a process pool, and stores the result as
one uint8 array file (N, H, W, C) next to the labels. CachedImageDataset then reads samples
straight from that file, so training does no image decoding at all. Do the float conversion
and augmentation on the GPU with run.loop(..., gpu_transform=...).

    ImageCache.build(paths, labels, "/content/cache/train", size=(224, 224), grayscale=True)
    loader = DataLoader(CachedImageDataset("/content/cache/train"), batch_size=64, shuffle=True)

The build is resumable: it writes the array in chunks and records finished chunks, so an
interrupted build (quota, disconnect) continues where it stopped. Images are resized to
`size` exactly (no aspect-preserving crop).

This module does not import torch at import time, so TensorFlow/Keras support can reuse
the cache builder.
"""

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..env.detect import detect_platform

FORMAT_VERSION = 1
IMAGES_FILE, LABELS_FILE, META_FILE, PROGRESS_FILE = (
    "images.npy", "labels.npy", "meta.json", "progress.json")


def _load_chunk(paths, size, grayscale):
    """Decode and resize a list of images. Runs in a worker process. Returns (n, H, W, C) uint8."""
    from PIL import Image
    h, w = size
    out = np.empty((len(paths), h, w, 1 if grayscale else 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as img:
                img = img.convert("L" if grayscale else "RGB")
                img = img.resize((w, h), Image.BILINEAR)
                out[i] = np.asarray(img, dtype=np.uint8).reshape(h, w, -1)
        except Exception as e:
            raise RuntimeError(f"ImageCache: could not read image {path}: {e}") from e
    return out


def _paths_hash(paths):
    return hashlib.sha1("\n".join(str(p) for p in paths).encode("utf-8")).hexdigest()


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)          # atomic: an interrupted write never leaves a broken file


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _size(nbytes):
    return f"{nbytes / 1024**3:.2f} GB" if nbytes >= 1024**3 else f"{nbytes / 1024**2:.1f} MB"


def _storage_tip(out_dir):
    platform = detect_platform()
    if platform == "colab":
        if os.path.abspath(out_dir).startswith("/content/drive"):
            return "It is on Google Drive, so it survives the session."
        return ("To keep it for later sessions, copy the folder to Google Drive (for example "
                "!cp -r <cache> /content/drive/MyDrive/). Copying one finished folder is much "
                "faster than building on Drive directly.")
    if platform == "kaggle":
        return ("To keep it for later sessions, save it as notebook output (it must be under "
                "/kaggle/working, then Save Version) or upload it as a private dataset, and "
                "attach that in later notebooks.")
    return "Keep it on a fast local disk (an SSD) for the best read speed."


class ImageCache:
    """A finished cache folder. Create one with ImageCache.build(...)."""

    def __init__(self, out_dir):
        self.out_dir = str(out_dir)
        self.meta = _read_json(os.path.join(self.out_dir, META_FILE))
        if not self.meta or not self.meta.get("complete"):
            raise FileNotFoundError(f"No finished image cache in {self.out_dir}. "
                                    "Run ImageCache.build(...) first.")

    def __len__(self):
        return self.meta["count"]

    @property
    def size_on_disk(self):
        return sum(os.path.getsize(os.path.join(self.out_dir, f))
                   for f in (IMAGES_FILE, LABELS_FILE, META_FILE))

    @classmethod
    def build(cls, paths, labels, out_dir, size=(224, 224), grayscale=False, workers=None,
              chunk_size=256, verbose=True):
        """Decode and resize every image in `paths` once into `out_dir`.

        labels: N ints (single label) or an (N, K) array of 0/1 floats (multi-label).
        workers: processes for decoding (default: all CPU cores; 0 or 1 = no pool). With
            workers > 1 on Windows/macOS, call this under `if __name__ == "__main__":`.
        Returns immediately if a finished cache with the same settings and paths exists.
        """
        paths = [str(p) for p in paths]
        labels = np.asarray(labels)
        if len(paths) == 0 or len(labels) != len(paths):
            raise ValueError(f"need the same, non-zero number of paths and labels "
                             f"(got {len(paths)} and {len(labels)})")
        if labels.ndim == 1:
            labels = labels.astype(np.int64)
        elif labels.ndim == 2:
            labels = labels.astype(np.float32)
        else:
            raise ValueError("labels must be 1-D (class ids) or 2-D (multi-hot)")
        h, w = (size, size) if isinstance(size, int) else tuple(size)
        channels = 1 if grayscale else 3
        out_dir = str(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        meta = {"version": FORMAT_VERSION, "size": [h, w], "channels": channels,
                "count": len(paths), "label_shape": list(labels.shape[1:]),
                "label_dtype": str(labels.dtype), "paths_sha1": _paths_hash(paths),
                "chunk_size": chunk_size}
        meta_path = os.path.join(out_dir, META_FILE)
        progress_path = os.path.join(out_dir, PROGRESS_FILE)
        images_path = os.path.join(out_dir, IMAGES_FILE)

        old = _read_json(meta_path)
        same = old is not None and {k: old.get(k) for k in meta} == meta
        if same and old.get("complete"):
            cache = cls(out_dir)
            if verbose:
                print(f"[efficio] image cache: using the finished cache in {out_dir} "
                      f"({len(cache)} images, {_size(cache.size_on_disk)})")
            return cache

        n_chunks = (len(paths) + chunk_size - 1) // chunk_size
        done = set((_read_json(progress_path) or {}).get("done", [])) if same else set()
        if same and os.path.exists(images_path):
            images = np.load(images_path, mmap_mode="r+")
        else:
            if old is not None and verbose:
                print(f"[efficio] image cache: settings or paths changed, rebuilding {out_dir}")
            done = set()
            images = np.lib.format.open_memmap(images_path, mode="w+", dtype=np.uint8,
                                               shape=(len(paths), h, w, channels))
            np.save(os.path.join(out_dir, LABELS_FILE), labels)
            _write_json(progress_path, {"done": []})
            _write_json(meta_path, {**meta, "complete": False})

        todo = [c for c in range(n_chunks) if c not in done]
        if verbose:
            resumed = f", resuming ({len(done)}/{n_chunks} chunks already done)" if done else ""
            print(f"[efficio] image cache: decoding {len(paths)} images to {h}x{w}x{channels} "
                  f"in {out_dir}{resumed}")

        def chunk_paths(c):
            return paths[c * chunk_size:(c + 1) * chunk_size]

        def finish_chunk(c, arr):
            images[c * chunk_size:c * chunk_size + len(arr)] = arr
            images.flush()                                   # data on disk before it counts
            done.add(c)
            _write_json(progress_path, {"done": sorted(done)})

        progress = _Progress(sum(len(chunk_paths(c)) for c in todo), verbose)
        if workers is None:
            workers = os.cpu_count() or 1
        if workers <= 1:
            for c in todo:
                finish_chunk(c, _load_chunk(chunk_paths(c), (h, w), grayscale))
                progress.add(len(chunk_paths(c)))
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            try:
                pending = {}
                queue = list(todo)
                while queue or pending:
                    while queue and len(pending) < 2 * workers:   # bound memory in flight
                        c = queue.pop(0)
                        pending[pool.submit(_load_chunk, chunk_paths(c), (h, w), grayscale)] = c
                    future = next(iter(pending))                  # finish in submission order
                    c = pending.pop(future)
                    finish_chunk(c, future.result())
                    progress.add(len(chunk_paths(c)))
            finally:
                # On an interrupt, drop queued chunks instead of finishing them first
                pool.shutdown(wait=True, cancel_futures=True)
        progress.close()
        del images
        _write_json(meta_path, {**meta, "complete": True})
        cache = cls(out_dir)
        if verbose:
            print(f"[efficio] image cache: done, {_size(cache.size_on_disk)} on disk in "
                  f"{out_dir}. {_storage_tip(out_dir)}")
        return cache


class _Progress:
    """One progress line with images per second and the estimated time left."""

    def __init__(self, total, verbose):
        self.total, self.verbose = total, verbose
        self.done, self.start, self.last = 0, time.time(), 0.0

    def add(self, n):
        self.done += n
        now = time.time()
        if self.verbose and (now - self.last >= 1.0 or self.done == self.total):
            self.last = now
            rate = self.done / max(now - self.start, 1e-9)
            left = (self.total - self.done) / rate if rate > 0 else 0
            print(f"\r[efficio] image cache: {self.done}/{self.total} images, "
                  f"{rate:.0f} img/s, ~{left / 60:.1f} min left   ", end="", flush=True)

    def close(self):
        if self.verbose and self.total:
            print()


class CachedImageDataset:
    """Map-style dataset over a finished ImageCache. Returns (uint8 tensor (C, H, W), label):
    an int64 class id, or a float32 multi-hot vector. No decoding happens here, so convert and
    augment on the GPU (run.loop(..., gpu_transform=...)).

    The arrays are opened lazily in each process, so DataLoader workers do not copy them."""

    def __init__(self, out_dir):
        self.cache = ImageCache(out_dir)            # raises if the cache is not finished
        self.out_dir = self.cache.out_dir
        self._images = self._labels = None

    def _open(self):
        if self._images is None:
            self._images = np.load(os.path.join(self.out_dir, IMAGES_FILE), mmap_mode="r")
            self._labels = np.load(os.path.join(self.out_dir, LABELS_FILE), mmap_mode="r")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        import torch
        self._open()
        # np.array copies: the memory-mapped file is read-only and must stay unchanged
        image = torch.from_numpy(np.array(self._images[i].transpose(2, 0, 1)))
        label = self._labels[i]
        label = torch.from_numpy(np.array(label)) if label.ndim else torch.tensor(int(label))
        return image, label

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = state["_labels"] = None  # reopen in the worker, don't pickle data
        return state

    @property
    def labels(self):
        self._open()
        return self._labels
