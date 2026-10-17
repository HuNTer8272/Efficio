"""Phase 2 step 1: dataloader tuning, channels_last, profiler and resume. CPU only."""

import pytest

from efficio.interfaces import HardwareInfo
from efficio.report.profiler import Profiler, tips

torch = pytest.importorskip("torch")
from torch import nn                                                    # noqa: E402
from torch.utils.data import (BatchSampler, DataLoader, IterableDataset,  # noqa: E402
                              RandomSampler, SequentialSampler, TensorDataset)

from efficio.speed import dataloader as dl                              # noqa: E402
from efficio.speed.memory_format import use_channels_last               # noqa: E402

CPU = HardwareInfo("local", "cpu")
GPU = HardwareInfo("colab", "cuda", "Tesla T4", 1, 15.0, (7, 5), False)


@pytest.fixture
def four_cores(monkeypatch):
    monkeypatch.setattr(dl, "_cpu_count", lambda: 4)
    monkeypatch.setattr(dl, "_can_add_workers", lambda: True)


def _data(n=100):
    return TensorDataset(torch.arange(n).float().unsqueeze(1), torch.arange(n))


# ---------- optimize_loader ----------

def test_optimize_loader_preserves_settings(four_cores):
    ds = _data()
    old = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    new, changes = dl.optimize_loader(old, CPU)

    assert new.dataset is ds
    assert len(new.dataset) == len(old.dataset)
    assert len(new) == len(old) == 12
    assert new.batch_size == 8 and new.drop_last is True
    assert new.sampler is old.sampler and isinstance(new.sampler, RandomSampler)
    assert new.num_workers == 4 and new.persistent_workers and new.prefetch_factor == 4
    assert new.pin_memory is False                      # CPU: nothing to pin for
    assert "num_workers 0 -> 4" in changes


def test_optimize_loader_shuffle_behavior(four_cores, monkeypatch):
    monkeypatch.setattr(dl, "_cpu_count", lambda: 2)   # keep worker start-up short
    shuffled, _ = dl.optimize_loader(DataLoader(_data(40), batch_size=8, shuffle=True), CPU)
    first = torch.cat([y for _, y in shuffled])
    second = torch.cat([y for _, y in shuffled])
    assert sorted(first.tolist()) == list(range(40))   # every sample exactly once
    assert first.tolist() != list(range(40)) or second.tolist() != list(range(40))

    ordered, _ = dl.optimize_loader(DataLoader(_data(40), batch_size=8, shuffle=False), CPU)
    assert isinstance(ordered.sampler, SequentialSampler)
    assert torch.cat([y for _, y in ordered]).tolist() == list(range(40))


def test_optimize_loader_keeps_user_workers_and_pins_on_gpu(four_cores):
    old = DataLoader(_data(), batch_size=8, num_workers=3)
    new, changes = dl.optimize_loader(old, GPU)
    assert new.num_workers == 3
    assert new.pin_memory and new.persistent_workers
    assert "pin_memory on" in changes


def test_optimize_loader_no_extra_workers_with_spawn(monkeypatch):
    monkeypatch.setattr(dl, "_can_add_workers", lambda: False)
    new, _ = dl.optimize_loader(DataLoader(_data(), batch_size=8), CPU)
    assert new.num_workers == 0


def test_optimize_loader_leaves_unsupported_loaders(four_cores):
    ds = _data()
    custom = DataLoader(ds, batch_sampler=BatchSampler(SequentialSampler(ds), 8, False))
    assert dl.optimize_loader(custom, CPU)[0] is custom

    class Stream(IterableDataset):
        def __iter__(self):
            return iter(range(10))

    stream = DataLoader(Stream(), batch_size=2)
    assert dl.optimize_loader(stream, CPU)[0] is stream

    plain = [1, 2, 3]
    loader, changes = dl.optimize_loader(plain, CPU)
    assert loader is plain and changes[0].startswith("unchanged")


# ---------- channels_last ----------

def test_channels_last_choice():
    conv = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())
    linear = nn.Linear(4, 2)
    assert use_channels_last("auto", "cpu", conv)[0] is False
    assert use_channels_last(True, "cpu", conv)[0] is False
    assert use_channels_last("auto", "cuda", linear)[0] is False
    assert use_channels_last(False, "cuda", conv)[0] is False
    assert use_channels_last("auto", "cuda", conv)[0] is True
    with pytest.raises(ValueError):
        use_channels_last("yes", "cuda", conv)


def test_run_skips_channels_last_on_cpu(tmp_path):
    import efficio
    run = efficio.Run(checkpoint_dir=str(tmp_path), resume=False, verbose=False)
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Flatten())
    run.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1))
    assert run.channels_last is False
    assert "channels_last" not in run.optimizations


# ---------- profiler ----------

def test_profiler_percentages():
    p = Profiler(clock=lambda: 0.0)       # created at t=0
    p.first_batch(now=20.0)               # 20 s setup
    for _ in range(10):
        p.add_step(data_wait_s=1.5, step_s=6.0)
    p.finish(now=100.0)                   # 80 s training
    assert p.setup_s == 20.0 and p.train_s == 80.0 and p.total_s == 100.0
    assert p.data_wait_pct == pytest.approx(15 / 80 * 100)
    assert p.setup_pct == pytest.approx(20.0)
    assert p.steps == 10


def test_profiler_without_batches():
    p = Profiler(clock=lambda: 5.0)
    p.finish(now=5.0)
    assert p.train_s == 0.0 and p.data_wait_pct == 0.0


def test_tips():
    assert any("waiting for data" in t for t in tips(25.0, 60.0, 1.0))
    assert not any("batch size" in t for t in tips(25.0, 60.0, 1.0))
    assert any("batch size" in t for t in tips(2.0, 60.0, 1.0))
    assert any("data cache" in t for t in tips(2.0, 95.0, 30.0))
    assert tips(2.0, 95.0, 1.0) == []
    assert tips(2.0, None, 1.0) == []     # no GPU numbers: no utilization tip
    assert tips(25.0, 60.0, 1.0, on_gpu=False) == []   # CPU run: no GPU tips
    assert any("data cache" in t for t in tips(25.0, None, 30.0, on_gpu=False))


# ---------- resume (quickstart-style) ----------

def _train(ckpt_dir, epochs, stop_after=None):
    """Train a tiny model through Run. With stop_after, save and quit early like a dying session."""
    import efficio
    torch.manual_seed(0)
    loader = DataLoader(_data(64), batch_size=8, shuffle=False)
    model = nn.Linear(1, 64)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    run = efficio.Run(checkpoint_dir=str(ckpt_dir), verbose=False)
    model, opt = run.prepare(model, opt)
    for n, (x, y) in enumerate(run.loop(loader, epochs=epochs)):
        loss = nn.functional.cross_entropy(model(x), y)
        run.backward(loss)
        run.step()
        if stop_after is not None and n + 1 == stop_after:
            run.save()
            return run, model
    return run, model


def test_resume_mid_epoch_matches_uninterrupted(tmp_path, four_cores, monkeypatch):
    monkeypatch.setattr(dl, "_cpu_count", lambda: 2)    # also exercise the worker loader
    full_run, full_model = _train(tmp_path / "full", epochs=3)

    _train(tmp_path / "split", epochs=3, stop_after=13)  # stops in the middle of epoch 2
    resumed, model = _train(tmp_path / "split", epochs=3)

    assert resumed._resumed_from == (1, 5)
    assert resumed.state.step == full_run.state.step == 24
    for a, b in zip(model.parameters(), full_model.parameters()):
        assert torch.allclose(a, b)
