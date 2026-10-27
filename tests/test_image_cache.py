"""ImageCache and CachedImageDataset on small generated PNGs (CPU)."""

import json
import os
import threading
import time

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")
torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402

from efficio.data import CachedImageDataset, ImageCache  # noqa: E402
from efficio.data import image_cache as ic  # noqa: E402
from efficio.interfaces import HardwareInfo  # noqa: E402
from efficio.speed.dataloader import optimize_loader  # noqa: E402


def _pngs(folder, n, size=(20, 30), mode="RGB"):
    """n random PNG files of different content, (height, width) = size."""
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        shape = size + ((3,) if mode == "RGB" else ())
        Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8), mode).save(folder / f"{i}.png")
        paths.append(folder / f"{i}.png")
    return paths


def _expected(path, size, grayscale):
    with Image.open(path) as img:
        img = img.convert("L" if grayscale else "RGB").resize(size[::-1], Image.BILINEAR)
        return np.asarray(img).reshape(size[0], size[1], -1)


def test_single_label_rgb(tmp_path):
    paths = _pngs(tmp_path / "img", 10)
    labels = list(range(10))
    cache = ImageCache.build(paths, labels, tmp_path / "cache", size=(16, 12), workers=0,
                             chunk_size=4, verbose=False)
    assert len(cache) == 10 and cache.size_on_disk > 10 * 16 * 12 * 3
    ds = CachedImageDataset(tmp_path / "cache")
    for i in (0, 5, 9):
        x, y = ds[i]
        assert x.shape == (3, 16, 12) and x.dtype == torch.uint8
        assert y.dtype == torch.int64 and y.item() == labels[i]
        assert np.array_equal(x.permute(1, 2, 0).numpy(), _expected(paths[i], (16, 12), False))


def test_multi_label_grayscale(tmp_path):
    paths = _pngs(tmp_path / "img", 7, mode="L")
    labels = np.random.default_rng(1).integers(0, 2, (7, 4)).astype(np.float32)
    ImageCache.build(paths, labels, tmp_path / "cache", size=8, grayscale=True, workers=0,
                     verbose=False)
    ds = CachedImageDataset(tmp_path / "cache")
    x, y = ds[3]
    assert x.shape == (1, 8, 8) and x.dtype == torch.uint8
    assert y.dtype == torch.float32 and torch.equal(y, torch.from_numpy(labels[3]))
    meta = json.loads((tmp_path / "cache" / "meta.json").read_text())
    assert meta["label_shape"] == [4] and meta["channels"] == 1 and meta["complete"]


def test_interrupted_build_resumes_without_redoing_chunks(tmp_path, monkeypatch):
    paths = _pngs(tmp_path / "img", 10)
    real = ic._decode_group
    calls = []

    def flaky(blobs, names, size, grayscale):
        if len(calls) == 2:
            raise KeyboardInterrupt          # like a dropped session after two chunks
        calls.append(names[0])
        return real(blobs, names, size, grayscale)

    monkeypatch.setattr(ic, "_decode_group", flaky)
    with pytest.raises(KeyboardInterrupt):
        ImageCache.build(paths, list(range(10)), tmp_path / "cache", size=8, workers=0,
                         chunk_size=3, verbose=False)
    # Chunks may finish in any order; exactly the two decoded ones are recorded as done
    chunk_of = {str(path): i // 3 for i, path in enumerate(paths)}
    done = json.loads((tmp_path / "cache" / "progress.json").read_text())["done"]
    assert sorted(done) == sorted(chunk_of[n] for n in calls) and len(done) == 2
    with pytest.raises(FileNotFoundError):
        CachedImageDataset(tmp_path / "cache")          # not finished yet

    calls.clear()
    monkeypatch.setattr(ic, "_decode_group", lambda *a: calls.append(a[1][0]) or real(*a))
    ImageCache.build(paths, list(range(10)), tmp_path / "cache", size=8, workers=0,
                     chunk_size=3, verbose=False)
    # Only the two missing chunks (of 4) were decoded
    assert sorted(chunk_of[n] for n in calls) == sorted({0, 1, 2, 3} - set(done))

    clean = ImageCache.build(paths, list(range(10)), tmp_path / "clean", size=8, workers=0,
                             chunk_size=3, verbose=False)
    resumed = np.load(tmp_path / "cache" / "images.npy")
    assert np.array_equal(resumed, np.load(os.path.join(clean.out_dir, "images.npy")))


def test_finished_cache_returns_immediately(tmp_path, monkeypatch, capsys):
    paths = _pngs(tmp_path / "img", 5)
    ImageCache.build(paths, [0] * 5, tmp_path / "cache", size=8, workers=0, verbose=False)
    calls = []
    real = ic._decode_group
    monkeypatch.setattr(ic, "_decode_group", lambda *a: calls.append(1) or real(*a))
    ImageCache.build(paths, [0] * 5, tmp_path / "cache", size=8, workers=0)
    assert calls == [] and "using the finished cache" in capsys.readouterr().out
    # Different settings: rebuilt
    ImageCache.build(paths, [0] * 5, tmp_path / "cache", size=6, workers=0, verbose=False)
    assert calls and CachedImageDataset(tmp_path / "cache")[0][0].shape == (3, 6, 6)


def test_process_pool_build_and_dataloader_with_workers(tmp_path):
    paths = _pngs(tmp_path / "img", 12)
    ImageCache.build(paths, list(range(12)), tmp_path / "cache", size=8, workers=2,
                     chunk_size=5, verbose=False)
    ds = CachedImageDataset(tmp_path / "cache")
    assert np.array_equal(ds[11][0].permute(1, 2, 0).numpy(), _expected(paths[11], (8, 8), False))
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    ys = torch.cat([y for _, y in loader])
    assert sorted(ys.tolist()) == list(range(12))
    # optimize_loader accepts it (it is a normal map-style dataset)
    tuned, changes = optimize_loader(loader, HardwareInfo("local", "cpu"))
    assert isinstance(tuned, DataLoader) and not changes[0].startswith("unchanged")


def test_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        ImageCache.build([tmp_path / "a.png"], [0, 1], tmp_path / "c", workers=0)
    with pytest.raises(RuntimeError, match="could not read image"):
        ImageCache.build([tmp_path / "missing.png"], [0], tmp_path / "c", size=4, workers=0,
                         verbose=False)


# ---------- parallel reads ----------

def test_io_threads_and_workers_give_identical_caches(tmp_path):
    paths = _pngs(tmp_path / "img", 23)
    builds = {}
    for name, kw in {"t1": dict(io_threads=1, workers=0), "t8": dict(io_threads=8, workers=0),
                     "t8_pool": dict(io_threads=8, workers=2)}.items():
        cache = ImageCache.build(paths, list(range(23)), tmp_path / name, size=(10, 14),
                                 chunk_size=5, verbose=False, **kw)
        builds[name] = np.load(os.path.join(cache.out_dir, "images.npy"))
        assert cache.build_stats["images"] == 23
    assert np.array_equal(builds["t1"], builds["t8"])
    assert np.array_equal(builds["t1"], builds["t8_pool"])
    # And identical to decoding straight from the file paths, as before the change
    for i in (0, 11, 22):
        assert np.array_equal(builds["t1"][i], _expected(paths[i], (10, 14), False))


def test_in_flight_limit(tmp_path, monkeypatch):
    paths = _pngs(tmp_path / "img", 40)
    lock, counts, worst = threading.Lock(), {"read": 0, "decoded": 0}, []
    real_read, real_decode = ic._read_file, ic._decode_group

    def slow_read(path):
        with lock:
            counts["read"] += 1
            worst.append(counts["read"] - counts["decoded"])
        time.sleep(0.002)
        return real_read(path)

    def counting_decode(blobs, names, size, grayscale):
        out = real_decode(blobs, names, size, grayscale)
        with lock:
            counts["decoded"] += len(blobs)
        return out

    monkeypatch.setattr(ic, "_read_file", slow_read)
    monkeypatch.setattr(ic, "_decode_group", counting_decode)
    cache = ImageCache.build(paths, [0] * 40, tmp_path / "cache", size=8, workers=0,
                             io_threads=2, chunk_size=16, verbose=False)
    limit = 4 * 2
    assert max(worst) <= limit                              # never more files held than allowed
    assert cache.build_stats["max_in_flight"] == limit     # and it does read ahead up to it
    assert counts == {"read": 40, "decoded": 40}


def test_storage_bottleneck_tip(capsys):
    progress = ic._Progress(1000, True, io_threads=32, decode_workers=4)
    progress.start = time.time() - 20                      # 20 s in
    progress.read(100)                                     # 5 files/s read
    progress.decoded(100, 1.0)                             # decoding could do 400 img/s
    out = capsys.readouterr().out
    assert "read 5 files/s, decode up to 400 img/s" in out
    assert "storage is the bottleneck" in out and "io_threads=64" in out
    progress.decoded(10, 0.1)
    assert "storage is the bottleneck" not in capsys.readouterr().out   # only once

    fast = ic._Progress(1000, True, io_threads=32, decode_workers=4)
    fast.start = time.time() - 20
    fast.read(4000)                                        # 200 files/s read
    fast.decoded(4000, 80.0)                               # decoding could do 200 img/s
    assert "storage is the bottleneck" not in capsys.readouterr().out


# ---------- GPU quota note ----------

@pytest.mark.parametrize("platform,cuda,shown", [("kaggle", True, True), ("colab", True, True),
                                                 ("local", True, False), ("kaggle", False, False)])
def test_gpu_note(tmp_path, monkeypatch, capsys, platform, cuda, shown):
    monkeypatch.setattr(ic, "detect_platform", lambda: platform)
    monkeypatch.setattr(ic, "_cuda_available", lambda: cuda)
    monkeypatch.setattr(ic, "_gpu_note_shown", False)
    paths = _pngs(tmp_path / "img", 3)
    ImageCache.build(paths, [0] * 3, tmp_path / "a", size=4, workers=0)
    ImageCache.build(paths, [0] * 3, tmp_path / "b", size=4, workers=0)   # a second build
    out = capsys.readouterr().out
    assert out.count("doesn't need a GPU") == (1 if shown else 0)        # at most once


def test_out_of_order_completion_gives_identical_cache(tmp_path, monkeypatch):
    """The first files are the slowest to read: later groups finish and are written first,
    and the cache is still identical to a strictly sequential build."""
    paths = _pngs(tmp_path / "img", 40)
    reference = ImageCache.build(paths, list(range(40)), tmp_path / "seq", size=(9, 7),
                                 io_threads=1, workers=0, chunk_size=5, verbose=False)
    real_read, real_decode = ic._read_file, ic._decode_group
    order = []

    def slow_first(path):
        index = int(os.path.basename(path).split(".")[0])
        time.sleep(0.2 if index < 4 else 0.0)            # files 0-3 (group 0) are slow
        return real_read(path)

    def record(blobs, names, size, grayscale):
        order.append(int(os.path.basename(names[0]).split(".")[0]))
        return real_decode(blobs, names, size, grayscale)

    monkeypatch.setattr(ic, "_read_file", slow_first)
    monkeypatch.setattr(ic, "_decode_group", record)
    ImageCache.build(paths, list(range(40)), tmp_path / "ooo", size=(9, 7), io_threads=8,
                     workers=0, chunk_size=5, verbose=False)     # up to 32 files in flight
    assert order != sorted(order) and order[-1] == 0      # group 0 was written last
    assert np.array_equal(np.load(tmp_path / "ooo" / "images.npy"),
                          np.load(os.path.join(reference.out_dir, "images.npy")))
    progress = json.loads((tmp_path / "ooo" / "progress.json").read_text())
    assert sorted(progress["done"]) == list(range(8))
