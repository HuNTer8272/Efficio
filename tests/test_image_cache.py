"""ImageCache and CachedImageDataset on small generated PNGs (CPU)."""

import json
import os

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")
torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader                                  # noqa: E402

from efficio.data import CachedImageDataset, ImageCache                  # noqa: E402
from efficio.data import image_cache as ic                               # noqa: E402
from efficio.interfaces import HardwareInfo                              # noqa: E402
from efficio.speed.dataloader import optimize_loader                     # noqa: E402


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
    real = ic._load_chunk
    calls = []

    def flaky(chunk, size, grayscale):
        if len(calls) == 2:
            raise KeyboardInterrupt          # like a dropped session after two chunks
        calls.append(chunk[0])
        return real(chunk, size, grayscale)

    monkeypatch.setattr(ic, "_load_chunk", flaky)
    with pytest.raises(KeyboardInterrupt):
        ImageCache.build(paths, list(range(10)), tmp_path / "cache", size=8, workers=0,
                         chunk_size=3, verbose=False)
    assert json.loads((tmp_path / "cache" / "progress.json").read_text())["done"] == [0, 1]
    with pytest.raises(FileNotFoundError):
        CachedImageDataset(tmp_path / "cache")          # not finished yet

    calls.clear()
    monkeypatch.setattr(ic, "_load_chunk", lambda *a: calls.append(a[0][0]) or real(*a))
    ImageCache.build(paths, list(range(10)), tmp_path / "cache", size=8, workers=0,
                     chunk_size=3, verbose=False)
    assert calls == [str(paths[6]), str(paths[9])]      # only chunks 2 and 3 were decoded

    clean = ImageCache.build(paths, list(range(10)), tmp_path / "clean", size=8, workers=0,
                             chunk_size=3, verbose=False)
    resumed = np.load(tmp_path / "cache" / "images.npy")
    assert np.array_equal(resumed, np.load(os.path.join(clean.out_dir, "images.npy")))


def test_finished_cache_returns_immediately(tmp_path, monkeypatch, capsys):
    paths = _pngs(tmp_path / "img", 5)
    ImageCache.build(paths, [0] * 5, tmp_path / "cache", size=8, workers=0, verbose=False)
    calls = []
    real = ic._load_chunk
    monkeypatch.setattr(ic, "_load_chunk", lambda *a: calls.append(1) or real(*a))
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
