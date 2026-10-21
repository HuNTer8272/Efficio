"""GPULoader and GPU augmentation, run on CPU so they work in CI."""

import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn                                                    # noqa: E402

from efficio.interfaces import HardwareInfo                             # noqa: E402
from efficio.speed import gpu_augment as A                              # noqa: E402
from efficio.speed import gpu_data as gd                                # noqa: E402
from efficio.speed.gpu_data import GPULoader                            # noqa: E402

CPU = HardwareInfo("local", "cpu")


def _images(n=50, h=8, w=8, c=3):
    x = torch.randint(0, 256, (n, h, w, c), dtype=torch.uint8)    # NHWC, like CIFAR's .data
    return x, torch.arange(n)


def _loader(n=50, batch_size=8, **kw):
    x, y = _images(n)
    return GPULoader.from_tensors(x, y, batch_size, device="cpu", **kw)


# ---------- GPULoader ----------

def test_batches_shapes_and_length():
    loader = _loader(50, 8)
    batches = list(loader)
    assert len(loader) == len(batches) == 7 and loader.batch_size == 8
    x, y = batches[0]
    assert x.shape == (8, 3, 8, 8) and x.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert y.shape == (8,)
    assert batches[-1][0].shape[0] == 2                     # 50 = 6 * 8 + 2


def test_every_sample_once_per_epoch_and_values_match():
    x, y = _images(50)
    loader = GPULoader.from_tensors(x, y, 8, device="cpu")
    for _ in range(2):
        xs, ys = zip(*loader)
        ys = torch.cat(ys)
        assert sorted(ys.tolist()) == list(range(50))
        # each image is its own source sample, as NCHW floats
        expected = x[ys].permute(0, 3, 1, 2).float() / 255
        assert torch.equal(torch.cat(xs), expected)


def test_drop_last():
    loader = _loader(50, 8, drop_last=True)
    batches = list(loader)
    assert len(loader) == len(batches) == 6
    assert all(b[0].shape[0] == 8 for b in batches)


def test_no_shuffle_is_in_order():
    ys = torch.cat([y for _, y in _loader(20, 8, shuffle=False)])
    assert ys.tolist() == list(range(20))


def test_permutation_reproducible_per_seed_and_epoch():
    a, b = _loader(seed=3), _loader(seed=3)
    assert torch.equal(a.permutation(2), b.permutation(2))
    assert not torch.equal(a.permutation(0), a.permutation(1))
    assert not torch.equal(a.permutation(0), _loader(seed=4).permutation(0))

    a.set_epoch(5)
    first = torch.cat([y for _, y in a])
    assert torch.equal(first, a.permutation(5))
    # Without set_epoch, the next iteration moves on to the next epoch
    assert torch.equal(torch.cat([y for _, y in a]), a.permutation(6))


def test_set_start_batch_skips():
    loader = _loader(50, 8)
    loader.set_epoch(1)
    full = [y for _, y in loader]
    loader.set_epoch(1)
    loader.set_start_batch(4)
    rest = [y for _, y in loader]
    assert len(rest) == 3
    assert all(torch.equal(a, b) for a, b in zip(rest, full[4:]))
    loader.set_epoch(1)
    assert len(list(loader)) == 7                           # one-shot: back to the start


def test_layouts():
    y = torch.arange(4)
    nchw = GPULoader.from_tensors(torch.zeros(4, 3, 8, 8, dtype=torch.uint8), y, 2, device="cpu")
    gray = GPULoader.from_tensors(torch.zeros(4, 28, 28, dtype=torch.uint8), y, 2, device="cpu")
    feats = GPULoader.from_tensors(torch.randn(4, 5), y, 2, device="cpu", shuffle=False)
    assert next(iter(nchw))[0].shape == (2, 3, 8, 8)
    assert next(iter(gray))[0].shape == (2, 1, 28, 28)
    xb = next(iter(feats))[0]
    assert torch.equal(xb, feats.x[:2])                     # non-image data is passed as is


def test_transform_applied():
    loader = _loader(16, 8, transform=lambda x: x * 0 + 7)
    assert all(torch.all(x == 7) for x, _ in loader)


def test_memory_check(monkeypatch):
    monkeypatch.setattr(gd, "_free_gpu_bytes", lambda device: 1000)
    with pytest.raises(ValueError, match="normal torch DataLoader"):
        gd._check_memory(400, torch.device("cuda"))
    gd._check_memory(300, torch.device("cuda"))             # 30% of free: still allowed
    gd._check_memory(10**12, torch.device("cpu"))           # no limit on CPU


def test_bad_inputs():
    with pytest.raises(ValueError):
        GPULoader.from_tensors(torch.zeros(4, 2), torch.arange(3), 2, device="cpu")
    with pytest.raises(ValueError):
        GPULoader.from_tensors(torch.zeros(4, 2), torch.arange(4), 0, device="cpu")


# ---------- from_torchvision ----------

def test_from_torchvision_fake_dataset():
    import numpy as np
    data = np.random.randint(0, 256, (20, 8, 8, 3), dtype=np.uint8)    # CIFAR: numpy NHWC
    fake = types.SimpleNamespace(data=data, targets=list(range(20)), transform=None)
    loader = GPULoader.from_torchvision(fake, batch_size=5, device="cpu", shuffle=False)
    x, y = next(iter(loader))
    assert x.shape == (5, 3, 8, 8) and y.tolist() == [0, 1, 2, 3, 4]
    assert torch.equal(x[0], torch.from_numpy(data[0]).permute(2, 0, 1).float() / 255)


def test_from_torchvision_warns_about_ignored_transform():
    fake = types.SimpleNamespace(data=torch.zeros(4, 8, 8, dtype=torch.uint8),
                                 targets=torch.arange(4), transform=lambda img: img)
    with pytest.warns(UserWarning, match="ignores the dataset's own transform"):
        GPULoader.from_torchvision(fake, batch_size=2, device="cpu")


def test_from_torchvision_needs_data_and_targets():
    with pytest.raises(TypeError, match="from_tensors"):
        GPULoader.from_torchvision(types.SimpleNamespace(images=[1], labels=[1]), batch_size=2)


# ---------- gpu_augment ----------

def test_random_crop_shape_and_content():
    x = torch.rand(6, 3, 8, 8)
    out = A.random_crop(x, 8, padding=2)
    assert out.shape == (6, 3, 8, 8)
    # Every crop is a window of the zero-padded image
    padded = nn.functional.pad(x, (2, 2, 2, 2))
    for n in range(6):
        windows = padded[n].unfold(1, 8, 1).unfold(2, 8, 1)          # (C, 5, 5, 8, 8)
        assert any(torch.equal(windows[:, i, j], out[n]) for i in range(5) for j in range(5))
    assert A.random_crop(x, 4).shape == (6, 3, 4, 4)
    assert torch.equal(A.random_crop(x, 8), x)                       # no padding, full size


def test_random_crop_offsets_per_sample():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 8, 8).expand(64, 3, 8, 8).contiguous()      # identical images
    out = A.random_crop(x, 8, padding=4)
    assert len({tuple(o.flatten().tolist()) for o in out}) > 1


def test_flip():
    x = torch.rand(5, 3, 4, 4)
    assert torch.equal(A.random_horizontal_flip(x, p=1.0), torch.flip(x, dims=[3]))
    assert torch.equal(A.random_horizontal_flip(x, p=0.0), x)
    torch.manual_seed(0)
    out = A.random_horizontal_flip(torch.rand(1, 1, 2, 2).expand(200, 1, 2, 2), p=0.5)
    flipped = (out[:, 0, 0, 0] != out[0, 0, 0, 0]).sum().item()
    assert 0 < flipped < 200                                          # chosen per sample


def test_normalize_and_compose():
    x = torch.rand(2, 3, 4, 4)
    mean, std = (0.1, 0.2, 0.3), (0.5, 0.25, 2.0)
    out = A.normalize(x, mean, std)
    for c in range(3):
        assert torch.allclose(out[:, c], (x[:, c] - mean[c]) / std[c])
    pipeline = A.Compose([A.RandomHorizontalFlip(p=1.0), A.Normalize(mean, std)])
    assert torch.allclose(pipeline(x), A.normalize(torch.flip(x, dims=[3]), mean, std))
    assert A.RandomCrop(4, padding=1)(x).shape == x.shape


# ---------- resume through Run.loop ----------

@pytest.fixture
def cpu_only(monkeypatch):
    import efficio.run
    monkeypatch.setattr(efficio.run, "detect_hardware", lambda: CPU)


def _train(ckpt_dir, epochs, stop_after=None):
    """Train through Run with a shuffled, augmented GPULoader. With stop_after, save and
    quit early like a dying session."""
    import efficio
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    x = torch.randint(0, 256, (60, 8, 8, 3), dtype=torch.uint8, generator=gen)
    y = torch.randint(0, 4, (60,), generator=gen)
    loader = GPULoader.from_tensors(
        x, y, batch_size=8, shuffle=True, device="cpu", seed=0,
        transform=A.Compose([A.RandomCrop(8, padding=2), A.RandomHorizontalFlip()]))
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    run = efficio.Run(checkpoint_dir=str(ckpt_dir), verbose=False)
    model, opt = run.prepare(model, opt)
    for n, (xb, yb) in enumerate(run.loop(loader, epochs=epochs)):
        run.backward(nn.functional.cross_entropy(model(xb), yb))
        run.step()
        if stop_after is not None and n + 1 == stop_after:
            run.save()
            return run, model
    return run, model


def test_resume_with_gpu_loader_matches_uninterrupted(tmp_path, cpu_only):
    full_run, full_model = _train(tmp_path / "full", epochs=3)

    _train(tmp_path / "split", epochs=3, stop_after=11)      # stops in the middle of epoch 2
    resumed, model = _train(tmp_path / "split", epochs=3)

    assert resumed._resumed_from == (1, 3)
    assert resumed.state.step == full_run.state.step == 24
    assert "GPULoader (data on the GPU)" in resumed.optimizations
    # Same permutation and the restored RNG give the same augmentations: identical weights
    for a, b in zip(model.parameters(), full_model.parameters()):
        assert torch.equal(a, b)
