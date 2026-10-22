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

Reading and decoding overlap: a thread pool (io_threads) reads file bytes ahead while worker
processes decode. On network-backed storage such as Kaggle's /kaggle/input each file read has
high latency, so many reads in parallel are much faster than one after another.

Building the cache needs no GPU, so on Colab/Kaggle it can be done in a CPU-only session.

This module does not import torch at import time, so TensorFlow/Keras support can reuse
the cache builder.
"""

import hashlib
import io
import json
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait

import numpy as np

from ..env.detect import detect_platform

FORMAT_VERSION = 1
IMAGES_FILE, LABELS_FILE, META_FILE, PROGRESS_FILE = (
    "images.npy", "labels.npy", "meta.json", "progress.json")
DECODE_GROUP = 16            # images per decode task
READ_TIP_RATIO = 0.5         # reads below this share of decode capacity: storage is the limit
GPU_NOTE = ("Building the cache doesn't need a GPU. To save GPU quota, build it once in a "
            "CPU-only session, save it (Kaggle: Save Version or a private dataset; Colab: "
            "Google Drive), then train in a GPU session.")
_gpu_note_shown = False


def _read_file(path):
    """Read one file's bytes. Runs in an I/O thread."""
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as e:
        raise RuntimeError(f"ImageCache: could not read image {path}: {e}") from e


def _decode_group(blobs, names, size, grayscale):
    """Decode and resize images from their file bytes. Runs in a worker process.
    Returns ((n, H, W, C) uint8 array, seconds spent)."""
    from PIL import Image
    start = time.perf_counter()
    h, w = size
    out = np.empty((len(blobs), h, w, 1 if grayscale else 3), dtype=np.uint8)
    for i, (blob, name) in enumerate(zip(blobs, names)):
        try:
            with Image.open(io.BytesIO(blob)) as img:
                img = img.convert("L" if grayscale else "RGB")
                img = img.resize((w, h), Image.BILINEAR)
                out[i] = np.asarray(img, dtype=np.uint8).reshape(h, w, -1)
        except Exception as e:
            raise RuntimeError(f"ImageCache: could not read image {name}: {e}") from e
    return out, time.perf_counter() - start


def _cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _gpu_note():
    """The GPU quota note, once per process, when building on Colab/Kaggle with a GPU."""
    global _gpu_note_shown
    if _gpu_note_shown or detect_platform() not in ("colab", "kaggle") or not _cuda_available():
        return None
    _gpu_note_shown = True
    return GPU_NOTE


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
        self.build_stats = None
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
              io_threads=32, chunk_size=256, verbose=True):
        """Decode and resize every image in `paths` once into `out_dir`.

        labels: N ints (single label) or an (N, K) array of 0/1 floats (multi-label).
        workers: processes for decoding (default: all CPU cores; 0 or 1 = no pool). With
            workers > 1 on Windows/macOS, call this under `if __name__ == "__main__":`.
        io_threads: threads reading files ahead of decoding. Raise it (e.g. 64) when the
            progress line shows reads far below decoding (slow network storage).
        Returns immediately if a finished cache with the same settings and paths exists.
        The returned cache has `build_stats` (rates, None when nothing was built).
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
        if workers is None:
            workers = os.cpu_count() or 1
        io_threads = max(1, int(io_threads))
        if verbose:
            note = _gpu_note()
            if note:
                print(f"[efficio] note: {note}")
            resumed = f", resuming ({len(done)}/{n_chunks} chunks already done)" if done else ""
            print(f"[efficio] image cache: decoding {len(paths)} images to {h}x{w}x{channels} "
                  f"in {out_dir}{resumed}")

        # Work in small groups of images inside each unfinished chunk
        limit = 4 * io_threads              # files read but not yet written: bounds memory
        group = min(DECODE_GROUP, limit)
        groups, left_in_chunk = deque(), {}
        for c in todo:
            first, end = c * chunk_size, min((c + 1) * chunk_size, len(paths))
            left_in_chunk[c] = end - first
            groups.extend(list(range(i, min(i + group, end))) for i in range(first, end, group))

        def finish_chunk(c):
            images.flush()                                   # data on disk before it counts
            done.add(c)
            _write_json(progress_path, {"done": sorted(done)})

        progress = _Progress(sum(left_in_chunk.values()), verbose, io_threads,
                             workers if workers > 1 else 1)
        readers = ThreadPoolExecutor(max_workers=io_threads)
        decoders = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            reading, decoding, in_flight = [], {}, 0

            def write(g, arr, seconds):
                nonlocal in_flight
                images[g[0]:g[-1] + 1] = arr          # each group has its own rows: any order
                in_flight -= len(g)
                c = g[0] // chunk_size
                left_in_chunk[c] -= len(g)
                if left_in_chunk[c] == 0:            # a chunk counts once all of it is written
                    finish_chunk(c)
                progress.decoded(len(g), seconds)

            while groups or reading or decoding:
                # 1. Read ahead (in file order), as long as the in-flight limit allows
                while groups and in_flight + len(groups[0]) <= limit:
                    g = groups.popleft()
                    reading.append((g, [readers.submit(_read_file, paths[i]) for i in g]))
                    in_flight += len(g)
                    progress.max_in_flight = max(progress.max_in_flight, in_flight)
                moved = False
                # 2. Any group whose files are all read goes to decoding, so one slow file
                #    only holds back its own group
                for item in [r for r in reading if all(f.done() for f in r[1])]:
                    reading.remove(item)
                    g, futures = item
                    blobs = [f.result() for f in futures]        # raises read errors
                    progress.read(len(g))
                    args = (blobs, [paths[i] for i in g], (h, w), grayscale)
                    if decoders is None:
                        write(g, *_decode_group(*args))          # nothing waits unsaved
                    else:
                        decoding[decoders.submit(_decode_group, *args)] = g
                    moved = True
                # 3. Write every finished group, in whatever order they finish
                for future in [f for f in decoding if f.done()]:
                    write(decoding.pop(future), *future.result())
                    moved = True
                if not moved:
                    pending = [f for _, futures in reading for f in futures if not f.done()]
                    wait(pending + list(decoding), return_when=FIRST_COMPLETED)
        finally:
            # On an interrupt, drop queued work instead of finishing it first
            readers.shutdown(wait=True, cancel_futures=True)
            if decoders is not None:
                decoders.shutdown(wait=True, cancel_futures=True)
        progress.close()
        del images
        _write_json(meta_path, {**meta, "complete": True})
        cache = cls(out_dir)
        cache.build_stats = progress.stats()
        if verbose:
            print(f"[efficio] image cache: done, {_size(cache.size_on_disk)} on disk in "
                  f"{out_dir}. {_storage_tip(out_dir)}")
        return cache


class _Progress:
    """One progress line: overall images/s, file reads/s, decode capacity and time left.
    Prints a one-time tip when reading files is the bottleneck."""

    def __init__(self, total, verbose, io_threads, decode_workers):
        self.total, self.verbose = total, verbose
        self.io_threads, self.decode_workers = io_threads, decode_workers
        self.done = self.files_read = 0
        self.decode_s, self.max_in_flight = 0.0, 0
        self.start, self.last, self.tip_shown = time.time(), 0.0, False

    def read(self, n):
        self.files_read += n

    def _rates(self):
        elapsed = max(time.time() - self.start, 1e-9)
        # Decode capacity: images per second of decode work, times the parallel workers
        capacity = self.done / self.decode_s * self.decode_workers if self.decode_s else 0.0
        return elapsed, self.done / elapsed, self.files_read / elapsed, capacity

    def decoded(self, n, seconds):
        self.done += n
        self.decode_s += seconds
        now = time.time()
        if not self.verbose or (now - self.last < 1.0 and self.done < self.total):
            return
        self.last = now
        elapsed, rate, reads, capacity = self._rates()
        left = (self.total - self.done) / rate if rate > 0 else 0
        print(f"\r[efficio] image cache: {self.done}/{self.total} images, {rate:.0f} img/s "
              f"(read {reads:.0f} files/s, decode up to {capacity:.0f} img/s), "
              f"~{left / 60:.1f} min left   ", end="", flush=True)
        if (not self.tip_shown and elapsed >= 10 and self.done >= 50
                and reads < READ_TIP_RATIO * capacity):
            self.tip_shown = True
            print(f"\n[efficio] image cache: reading files ({reads:.0f}/s) is much slower than "
                  f"decoding ({capacity:.0f}/s), so storage is the bottleneck. Try a higher "
                  f"io_threads, e.g. ImageCache.build(..., io_threads={2 * self.io_threads}).")

    def stats(self):
        elapsed, rate, reads, capacity = self._rates()
        return {"seconds": round(elapsed, 2), "images": self.done,
                "images_per_s": round(rate, 1), "read_files_per_s": round(reads, 1),
                "decode_capacity_per_s": round(capacity, 1), "io_threads": self.io_threads,
                "decode_workers": self.decode_workers, "max_in_flight": self.max_in_flight}

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
