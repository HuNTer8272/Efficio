"""GPU augmentation ops, gpu_transform in Run.loop, clip_grad_norm and evaluate_mode (CPU)."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import efficio  # noqa: E402
from efficio.interfaces import HardwareInfo  # noqa: E402
from efficio.speed import gpu_augment as A  # noqa: E402

CPU = HardwareInfo("local", "cpu")


@pytest.fixture
def cpu_only(monkeypatch):
    import efficio.run
    monkeypatch.setattr(efficio.run, "detect_hardware", lambda: CPU)


# ---------- GPU augmentation ops ----------

RANDOM_OPS = {
    "random_resized_crop": A.RandomResizedCrop(12, scale=(0.3, 1.0)),
    "random_affine": A.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    "random_brightness_contrast": A.RandomBrightnessContrast(0.3, 0.3),
    "random_crop": A.RandomCrop(16, padding=2),
    "random_horizontal_flip": A.RandomHorizontalFlip(),
}


@pytest.mark.parametrize("name", RANDOM_OPS)
def test_random_ops_shape_dtype_and_seed(name):
    op = RANDOM_OPS[name]
    x = torch.rand(6, 3, 16, 16)
    torch.manual_seed(0)
    a = op(x)
    torch.manual_seed(0)
    b = op(x)
    size = 12 if name == "random_resized_crop" else 16
    assert a.shape == (6, 3, size, size) and a.dtype == x.dtype
    assert torch.equal(a, b)                                     # same seed, same result
    torch.manual_seed(1)
    assert not torch.equal(op(x), a)                             # other seed, other result
    same = x[:1].expand(6, -1, -1, -1)
    torch.manual_seed(0)
    out = op(same)
    assert len({tuple(o.flatten().tolist()) for o in out}) > 1   # randomness per sample


def test_ops_keep_float_dtype():
    x = torch.rand(2, 3, 16, 16, dtype=torch.float64)            # not float32: dtype is kept
    for op in RANDOM_OPS.values():
        assert op(x).dtype == torch.float64
    assert A.Resize(4)(x).dtype == torch.float64


def test_identity_settings():
    x = torch.rand(4, 3, 10, 10)
    assert torch.allclose(A.random_affine(x, degrees=0), x, atol=1e-5)
    assert torch.allclose(A.random_resized_crop(x, 10, scale=(1, 1), ratio=(1, 1)), x, atol=1e-5)
    assert torch.allclose(A.random_brightness_contrast(x, 0, 0), x, atol=1e-6)


def test_affine_geometry(monkeypatch):
    """With the random draws fixed at their upper bound, check exact pixel positions."""
    monkeypatch.setattr(A, "_uniform", lambda n, low, high, x: torch.full((n,), float(high)))
    dot = torch.zeros(1, 1, 5, 5)
    dot[0, 0, 0, 2] = 1.0                                        # top middle
    turned = A.random_affine(dot, degrees=90)
    assert torch.allclose(turned[0, 0, 2, 4], torch.tensor(1.0), atol=1e-5)   # now right middle
    assert torch.allclose(turned.sum(), torch.tensor(1.0), atol=1e-5)
    x = torch.rand(2, 3, 4, 8)                                   # not square
    assert torch.allclose(A.random_affine(x, degrees=180), x.flip(2, 3), atol=1e-5)
    shifted = A.random_affine(x[:, :, :, :5], degrees=0, translate=(0.2, 0.0))  # 1 pixel
    assert torch.allclose(shifted[..., 1:], x[..., :4], atol=1e-5)
    assert torch.all(shifted[..., 0] == 0)


def test_to_float_repeat_resize():
    u = torch.randint(0, 256, (2, 1, 8, 8), dtype=torch.uint8)
    f = A.ToFloat()(u)
    assert f.dtype == torch.float32 and torch.allclose(f * 255, u.float())
    assert A.ToFloat()(f) is f
    r = A.RepeatChannels(3)(f)
    assert r.shape == (2, 3, 8, 8) and torch.equal(r[:, 0], r[:, 2])
    assert A.Resize((4, 6))(f).shape == (2, 1, 4, 6)
    with pytest.raises(TypeError, match="ToFloat"):
        A.Resize(4)(u)
    with pytest.raises(ValueError):
        A.RepeatChannels(3)(torch.rand(1, 2, 4, 4))


def test_on_inputs():
    double = efficio.on_inputs(lambda x: x * 2)
    x, y = torch.ones(2, 3), torch.tensor([1, 2])
    a, b = double((x, y))
    assert torch.equal(a, x * 2) and b is y
    assert isinstance(double([x, y]), list)
    assert torch.equal(double({"pixel_values": x, "labels": y})["pixel_values"], x * 2)
    assert torch.equal(double({"image": x})["image"], x * 2)
    assert torch.equal(double(x), x * 2)
    with pytest.raises(KeyError):
        double({"text": x})


# ---------- gpu_transform in Run.loop ----------

def _images(n=48):
    g = torch.Generator().manual_seed(3)
    x = torch.randint(0, 256, (n, 1, 8, 8), dtype=torch.uint8, generator=g)
    return TensorDataset(x, torch.randint(0, 3, (n,), generator=g))


AUGMENT = efficio.on_inputs(A.Compose([
    A.ToFloat(), A.RepeatChannels(3), A.RandomResizedCrop(8, scale=(0.5, 1.0)),
    A.RandomAffine(degrees=10, translate=(0.1, 0.1)), A.RandomHorizontalFlip(),
    A.RandomBrightnessContrast(0.2, 0.2), A.Normalize((0.5,) * 3, (0.25,) * 3)]))


def _train(ckpt_dir, epochs, stop_after=None):
    torch.manual_seed(0)
    loader = DataLoader(_images(), batch_size=8, shuffle=True)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    run = efficio.Run(checkpoint_dir=str(ckpt_dir), verbose=False, optimize_data=False)
    model, opt = run.prepare(model, opt)
    for n, (x, y) in enumerate(run.loop(loader, epochs=epochs, gpu_transform=AUGMENT)):
        assert x.dtype == torch.float32 and x.shape == (8, 3, 8, 8)
        run.backward(nn.functional.cross_entropy(model(x), y))
        run.clip_grad_norm(1.0)
        run.step()
        if stop_after is not None and n + 1 == stop_after:
            run.save()
            break
    return run, model


def test_gpu_transform_resume_is_exact(tmp_path, cpu_only):
    full_run, full_model = _train(tmp_path / "full", 3)
    _train(tmp_path / "split", 3, stop_after=9)                 # stops in epoch 2 (6 batches)
    resumed, model = _train(tmp_path / "split", 3)
    assert resumed._resumed_from == (1, 3)
    assert resumed.state.step == full_run.state.step == 18
    for a, b in zip(model.parameters(), full_model.parameters()):
        assert torch.equal(a, b)


# ---------- clip_grad_norm ----------

def _run_with_model(tmp_path, grad_accum_steps=1):
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    run = efficio.Run(checkpoint_dir=str(tmp_path), resume=False, verbose=False,
                      grad_accum_steps=grad_accum_steps)
    run.prepare(model, opt)
    return run, model, opt


def _grad_norm(model):
    return torch.norm(torch.stack([p.grad.norm() for p in model.parameters()]))


def test_clip_grad_norm_disabled_scaler(tmp_path, cpu_only):
    run, model, _ = _run_with_model(tmp_path)
    run.backward((model(torch.full((3, 4), 10.0)) ** 2).sum())
    before = _grad_norm(model)
    total = run.clip_grad_norm(0.5)
    assert before > 0.5 and torch.allclose(total, before)
    assert _grad_norm(model) <= 0.5 + 1e-5


def test_clip_grad_norm_with_accumulation(tmp_path, cpu_only):
    run, model, _ = _run_with_model(tmp_path, grad_accum_steps=2)
    x = torch.full((3, 4), 10.0)
    run.backward((model(x) ** 2).sum())
    first = _grad_norm(model).clone()
    assert run.clip_grad_norm(0.5) is None                       # still accumulating
    assert torch.equal(_grad_norm(model), first)
    run.step()                                                   # micro-batch 1 of 2
    run.backward((model(x) ** 2).sum())
    assert run.clip_grad_norm(0.5) is not None                   # the optimizer steps next
    assert _grad_norm(model) <= 0.5 + 1e-5
    run.step()
    assert run.state.step == 1


def test_clip_grad_norm_unscales_fp16_style_gradients(tmp_path, cpu_only):
    """With an enabled GradScaler, clipping must use real (unscaled) gradient values."""
    x = torch.full((3, 4), 10.0)
    ref_run, ref_model, ref_opt = _run_with_model(tmp_path / "ref")
    (ref_model(x) ** 2).sum().backward()
    ref_total = nn.utils.clip_grad_norm_(ref_model.parameters(), 0.5)
    ref_opt.step()

    run, model, _ = _run_with_model(tmp_path / "scaled")
    run.scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    run.backward((model(x) ** 2).sum())                          # gradients x1024
    total = run.clip_grad_norm(0.5)
    assert torch.allclose(total, ref_total)
    run.step()
    for a, b in zip(model.parameters(), ref_model.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


# ---------- evaluate_mode ----------

def test_evaluate_mode_restores_mode(tmp_path, cpu_only):
    run, model, _ = _run_with_model(tmp_path)
    assert model.training
    with run.evaluate_mode():
        assert not model.training and not torch.is_grad_enabled()
        out = model(torch.ones(1, 4))
        assert not out.requires_grad
    assert model.training and torch.is_grad_enabled()
    model.eval()
    with run.evaluate_mode():
        pass
    assert not model.training                                    # stays as it was


# ---------- ImageCache tip detection ----------

def test_detects_datasets_that_decode_image_files(tmp_path):
    from efficio.run import _decodes_image_files

    class FolderLike(torch.utils.data.Dataset):          # like torchvision ImageFolder
        samples = [("a.png", 0)]

        def __len__(self):
            return 1

        def __getitem__(self, i):
            return torch.zeros(1), 0

    class ChestXrayImages(FolderLike):
        samples = None

    del ChestXrayImages.samples
    assert _decodes_image_files(FolderLike())
    assert _decodes_image_files(torch.utils.data.Subset(FolderLike(), [0]))
    assert _decodes_image_files(ChestXrayImages())          # "image" in the class name
    assert not _decodes_image_files(TensorDataset(torch.zeros(2)))
    assert not _decodes_image_files(None)
    PIL = pytest.importorskip("PIL.Image")
    import numpy as np

    from efficio.data import CachedImageDataset, ImageCache
    PIL.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / "a.png")
    ImageCache.build([tmp_path / "a.png"], [0], tmp_path / "c", size=4, workers=0, verbose=False)
    assert not _decodes_image_files(CachedImageDataset(tmp_path / "c"))
