"""Checkpoint modes, trainable-only saving and background saves. CPU only."""

import os
import random
import threading

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import efficio  # noqa: E402
from efficio.checkpoint import manager as cm  # noqa: E402
from efficio.checkpoint.light import (  # noqa: E402
    decode_optimizer,
    dequantize_blockwise,
    encode_optimizer,
    quantize_blockwise,
)
from efficio.checkpoint.manager import CheckpointManager, CheckpointMismatchError  # noqa: E402
from efficio.interfaces import HardwareInfo, TrainState  # noqa: E402


@pytest.fixture
def cpu_only(monkeypatch):
    """Run-based tests check CPU behaviour on every machine, also on a GPU runner."""
    import efficio.run as run_module
    monkeypatch.setattr(run_module, "detect_hardware", lambda: HardwareInfo("local", "cpu"))


def _adam_model(width=256, steps=3):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        model(torch.randn(8, width)).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    return model, opt


# ---------- 0.1.x compatibility ----------

def _save_like_0_1_1(path, model, opt, step=5):
    """The exact payload Efficio 0.1.1 wrote (format 1, no metadata)."""
    torch.save({
        "efficio_version": 1, "mode": "exact",
        "train_state": TrainState(epoch=1, step=step, batch_in_epoch=2).to_dict(),
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "scheduler": None, "scaler": None, "rng": cm._rng_state(),
        "extra": {"shuffle_seed": 3},
    }, path)


def test_checkpoint_from_0_1_x_loads_and_resumes(tmp_path, cpu_only):
    model, opt = _adam_model()
    _save_like_0_1_1(tmp_path / "ckpt_step000000005.pt", model, opt)
    fresh, fresh_opt = _adam_model(steps=0)
    run = efficio.Run(checkpoint_dir=str(tmp_path), verbose=False)
    fresh, fresh_opt = run.prepare(fresh, fresh_opt)
    assert run.state.step == 5 and run._resumed_from == (1, 2)
    assert run.ckpt.extra == {"shuffle_seed": 3} and run.ckpt.meta == {}
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(opt.state_dict()["state"][0]["exp_avg"],
                       fresh_opt.state_dict()["state"][0]["exp_avg"])


def test_exact_mode_is_bit_identical_and_has_metadata(tmp_path):
    model, opt = _adam_model()
    mgr = CheckpointManager(str(tmp_path))
    path = mgr.save(TrainState(step=1), model, opt)
    meta = torch.load(path, weights_only=False)["meta"]
    assert meta["format_version"] == 2 and meta["mode"] == "exact"
    assert meta["trainable_only"] is False and meta["sizes"]["model_bytes"] > 0
    fresh, fresh_opt = _adam_model(steps=0)
    mgr.load(fresh, fresh_opt)
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(a, b)
    for k in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(opt.state_dict()["state"][0][k], fresh_opt.state_dict()["state"][0][k])


# ---------- 8-bit quantization ----------

@pytest.mark.parametrize("shape", [(1000,), (64, 300), (7,), (128,)])
def test_quantize_error_bound(shape):
    torch.manual_seed(1)
    t = torch.randn(shape) * torch.logspace(-3, 1, shape[-1])
    packed = quantize_blockwise(t)
    assert packed["q"].dtype == torch.int8 and packed["q"].numel() == t.numel()
    back = dequantize_blockwise(packed)
    assert back.shape == t.shape
    flat, err = t.reshape(-1), (back - t).abs().reshape(-1)
    for i in range(0, flat.numel(), packed["block"]):                 # per-block bound
        block = flat[i:i + packed["block"]]
        assert err[i:i + packed["block"]].max() <= block.abs().max() / 254 + 1e-7


def test_quantize_zeros_and_constant_blocks():
    t = torch.zeros(300)
    t[200:] = 5.0
    assert torch.equal(dequantize_blockwise(quantize_blockwise(t)), t)


def test_adam_state_round_trip_is_close_and_safe():
    model, opt = _adam_model()
    sd = opt.state_dict()
    back = decode_optimizer(encode_optimizer(sd))
    for pid, st in sd["state"].items():
        m, v = st["exp_avg"], st["exp_avg_sq"]
        if m.numel() < 1024:                       # small tensors are stored exactly
            assert torch.equal(back["state"][pid]["exp_avg"], m)
            continue
        m2, v2 = back["state"][pid]["exp_avg"], back["state"][pid]["exp_avg_sq"]
        assert torch.equal(back["state"][pid]["step"], st["step"])      # exact
        assert (v2 >= 0).all()
        assert (v2.sqrt() - v.sqrt()).abs().max() <= v.sqrt().max() / 100
        # The Adam update direction m / (sqrt(v) + eps) stays close and bounded
        eps = sd["param_groups"][0]["eps"]
        upd, upd2 = m / (v.sqrt() + eps), m2 / (v2.sqrt() + eps)
        assert (upd2 - upd).abs().max() <= upd.abs().max() / 50
        assert upd2.abs().max() <= upd.abs().max() * 1.05


# ---------- light mode ----------

def test_light_file_is_much_smaller(tmp_path):
    model, opt = _adam_model(width=512)
    exact = CheckpointManager(str(tmp_path / "exact")).save(TrainState(step=1), model, opt)
    light = CheckpointManager(str(tmp_path / "light"), mode="light").save(
        TrainState(step=1), model, opt)
    ratio = os.path.getsize(exact) / os.path.getsize(light)
    assert ratio >= 2.5, f"light file only {ratio:.2f}x smaller"


def _train(ckpt_dir, epochs, mode="exact", stop_after=None, **run_kwargs):
    torch.manual_seed(0)
    data = TensorDataset(torch.randn(256, 64), torch.randint(0, 4, (256,)))
    loader = DataLoader(data, batch_size=32, shuffle=False)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 4))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run = efficio.Run(checkpoint_dir=str(ckpt_dir), verbose=False, checkpoint_mode=mode,
                      optimize_data=False, **run_kwargs)
    model, opt = run.prepare(model, opt)
    for n, (x, y) in enumerate(run.loop(loader, epochs=epochs)):
        loss = nn.functional.cross_entropy(model(x), y)
        run.backward(loss)
        run.step()
        if stop_after is not None and n + 1 == stop_after:
            run.save()
            return run, model, loss.item()
    return run, model, loss.item()


def test_light_mode_resumes_close_with_same_step_count(tmp_path, cpu_only):
    full, full_model, full_loss = _train(tmp_path / "full", epochs=3)
    _train(tmp_path / "light", epochs=3, mode="light", stop_after=13)
    resumed, model, loss = _train(tmp_path / "light", epochs=3, mode="light")
    assert resumed._resumed_from == (1, 5)
    assert resumed.state.step == full.state.step == 24
    diff = max((a - b).abs().max().item()
               for a, b in zip(model.parameters(), full_model.parameters()))
    assert diff < 5e-3, diff
    assert abs(loss - full_loss) < 0.05


def test_fp16_weights_are_upcast_on_load(tmp_path):
    model, opt = _adam_model()
    mgr = CheckpointManager(str(tmp_path), mode="light")
    payload = torch.load(mgr.save(TrainState(step=1), model, opt), weights_only=False)
    assert payload["model"]["0.weight"].dtype == torch.float16
    fresh, fresh_opt = _adam_model(steps=0)
    mgr.load(fresh, fresh_opt)
    assert fresh[0].weight.dtype == torch.float32
    assert (fresh[0].weight - model[0].weight).abs().max() < 1e-3


# ---------- trainable-only ----------

def _frozen_backbone(seed=0):
    torch.manual_seed(seed)
    backbone = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
    model = nn.Sequential(backbone, nn.ReLU(), nn.Linear(256, 4))
    for p in backbone.parameters():
        p.requires_grad_(False)
    torch.manual_seed(1)
    nn.init.normal_(model[2].weight)                                  # same head init
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    return model, opt


def _step(model, opt, n=3):
    torch.manual_seed(2)
    for _ in range(n):
        model(torch.randn(8, 256)).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()


def test_trainable_only_much_smaller_and_resumes(tmp_path):
    model, opt = _frozen_backbone()
    _step(model, opt)
    full = CheckpointManager(str(tmp_path / "full")).save(TrainState(step=3), model, opt)
    mgr = CheckpointManager(str(tmp_path / "head"), trainable_only=True)
    head = mgr.save(TrainState(step=3), model, opt)
    assert os.path.getsize(full) > 10 * os.path.getsize(head)      # 18x for this model
    meta = torch.load(head, weights_only=False)["meta"]
    assert meta["trainable_only"] and set(meta["frozen"]) == {"0.0.weight", "0.0.bias",
                                                              "0.2.weight", "0.2.bias"}
    fresh, fresh_opt = _frozen_backbone()
    assert mgr.load(fresh, fresh_opt)["step"] == 3
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(a, b)


def test_trainable_only_refuses_different_frozen_weights(tmp_path):
    model, opt = _frozen_backbone(seed=0)
    _step(model, opt)
    mgr = CheckpointManager(str(tmp_path), trainable_only=True)
    mgr.save(TrainState(step=3), model, opt)
    other, other_opt = _frozen_backbone(seed=99)                       # another "pretrained" model
    with pytest.raises(CheckpointMismatchError, match="frozen parameters"):
        mgr.load(other, other_opt)


def test_run_auto_trainable_only(tmp_path, cpu_only):
    model, opt = _frozen_backbone()
    run = efficio.Run(checkpoint_dir=str(tmp_path / "a"), verbose=False)
    run.prepare(model, opt)
    assert run.ckpt.trainable_only is True                             # some params frozen
    plain, plain_opt = _adam_model(steps=0)
    run2 = efficio.Run(checkpoint_dir=str(tmp_path / "b"), verbose=False)
    run2.prepare(plain, plain_opt)
    assert run2.ckpt.trainable_only is False                           # nothing frozen
    with pytest.raises(ValueError):
        efficio.Run(checkpoint_dir=str(tmp_path / "c"), save_trainable_only="yes")


# ---------- background saving ----------

def test_background_save_writes_the_same_file(tmp_path):
    model, opt = _adam_model()
    mgr = CheckpointManager(str(tmp_path), async_save=True)
    path = mgr.save(TrainState(step=7), model, opt, blocking=False)
    _step_plain(model, opt)                         # training goes on while it writes
    mgr.wait()
    saved = torch.load(path, weights_only=False)
    assert mgr.stats["background_s"] > 0 and mgr.stats["last_bytes"] == os.path.getsize(path)
    assert saved["train_state"]["step"] == 7
    assert not torch.equal(saved["model"]["0.weight"], model[0].weight.detach())  # snapshot


def _step_plain(model, opt):
    model(torch.randn(8, model[0].in_features)).pow(2).mean().backward()
    opt.step()
    opt.zero_grad()


def test_interrupted_background_write_keeps_latest_checkpoint(tmp_path, monkeypatch):
    model, opt = _adam_model()
    mgr = CheckpointManager(str(tmp_path), async_save=True)
    good = mgr.save(TrainState(step=1), model, opt)
    real_save = torch.save

    def crash_mid_write(obj, f, *args, **kwargs):
        with open(f, "wb") as fh:
            fh.write(b"half a checkpoint")             # partial file, then the "crash"
        raise OSError("disk full")

    monkeypatch.setattr(cm.torch, "save", crash_mid_write)
    mgr.save(TrainState(step=2), model, opt, blocking=False)
    with pytest.raises(RuntimeError, match="disk full"):
        mgr.wait()
    monkeypatch.setattr(cm.torch, "save", real_save)
    assert mgr.latest() == good
    assert not any(f.endswith(".tmp") for f in os.listdir(tmp_path))
    fresh, fresh_opt = _adam_model(steps=0)
    assert mgr.load(fresh, fresh_opt)["step"] == 1


def test_background_error_surfaces_at_next_step(tmp_path, cpu_only, monkeypatch):
    run = efficio.Run(checkpoint_dir=str(tmp_path), verbose=False)
    model, opt = _adam_model(steps=0)
    run.prepare(model, opt)

    def fail(obj, f, *args, **kwargs):
        raise OSError("quota exceeded")

    monkeypatch.setattr(cm.torch, "save", fail)
    run.save(blocking=False)
    run.ckpt._thread.join()
    run.backward(model(torch.randn(2, 256)).sum())
    with pytest.raises(RuntimeError, match="quota exceeded"):
        run.step()


def test_emergency_save_waits_for_pending_background_save(tmp_path, cpu_only, monkeypatch):
    run = efficio.Run(checkpoint_dir=str(tmp_path), verbose=False)
    model, opt = _adam_model(steps=0)
    run.prepare(model, opt)
    release, order = threading.Event(), []
    real_save = torch.save

    def slow_save(obj, f, *args, **kwargs):
        if obj["train_state"]["step"] == 1:
            release.wait(5)                             # background write still running
        order.append(obj["train_state"]["step"])
        real_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(cm.torch, "save", slow_save)
    run.state.step = 1
    run.save(blocking=False)
    assert run.ckpt.pending
    threading.Timer(0.3, release.set).start()
    run.state.step = 2
    run._signal_received = True                         # SIGTERM -> emergency save
    run._maybe_save_or_stop()
    assert order == [1, 2] and not run.ckpt.pending
    assert run.ckpt.latest().endswith("ckpt_step000000002.pt")
    assert run.ckpt.stats["waited_s"] > 0


def test_report_shows_checkpoint_mode_size_and_time(tmp_path, cpu_only, capsys):
    run, _, _ = _train(tmp_path, epochs=1, mode="light")
    run.report()
    out = capsys.readouterr().out
    assert "checkpoint mode : light" in out and "last file" in out
    assert "save time" in out


def test_invalid_mode_is_refused(tmp_path):
    with pytest.raises(ValueError):
        CheckpointManager(str(tmp_path), mode="tiny")
    random.random()
