"""Phase 2 step 1: dataloader tuning, channels_last, profiler and resume. CPU only."""

import pytest

from efficio.interfaces import HardwareInfo
from efficio.report.profiler import Profiler, tips

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import (  # noqa: E402
    BatchSampler,
    DataLoader,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from efficio.speed import dataloader as dl  # noqa: E402
from efficio.speed.memory_format import use_channels_last  # noqa: E402

CPU = HardwareInfo("local", "cpu")
GPU = HardwareInfo("colab", "cuda", "Tesla T4", 1, 15.0, (7, 5), False)


@pytest.fixture
def cpu_only(monkeypatch):
    """Run on CPU even on a GPU machine (Colab/Kaggle), so these tests check the same thing
    everywhere. Without this, Run picks the GPU and turns channels_last on."""
    import efficio.run
    monkeypatch.setattr(efficio.run, "detect_hardware", lambda: CPU)


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


def test_run_skips_channels_last_on_cpu(tmp_path, cpu_only):
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
    assert any("ImageCache" in t for t in tips(2.0, 95.0, 30.0))
    assert tips(2.0, 95.0, 1.0) == []
    assert not any("GPULoader" in t for t in tips(25.0, 60.0, 1.0))
    assert any("GPULoader" in t for t in tips(25.0, 60.0, 1.0, gpu_loader_fits=True))
    assert not any("GPULoader" in t for t in tips(5.0, 60.0, 1.0, gpu_loader_fits=True))
    assert tips(2.0, None, 1.0) == []     # no GPU numbers: no utilization tip
    assert tips(25.0, 60.0, 1.0, on_gpu=False) == []   # CPU run: no GPU tips
    assert any("ImageCache" in t for t in tips(25.0, None, 30.0, on_gpu=False))
    # Image files decoded every epoch: suggest ImageCache, but GPULoader wins when it fits
    assert any("ImageCache.build" in t for t in tips(25.0, 60.0, 1.0, image_files=True))
    assert not any("ImageCache.build" in t
                   for t in tips(25.0, 60.0, 1.0, gpu_loader_fits=True, image_files=True))
    assert not any("ImageCache.build" in t for t in tips(5.0, 95.0, 1.0, image_files=True))


def test_short_run_hides_setup_tip():
    # Setup took 30% of the run: shown after 2 minutes of training, hidden before
    assert any("Setup took" in t for t in tips(2.0, 95.0, 30.0, train_s=120.0))
    assert not any("Setup took" in t for t in tips(2.0, 95.0, 30.0, train_s=119.0))
    assert not any("Setup took" in t for t in tips(2.0, None, 96.0, on_gpu=False, train_s=3.0))


def test_short_run_hides_batch_size_tip():
    # Low utilization with little data-wait: shown after 2 minutes of training, hidden before
    assert any("batch size" in t for t in tips(2.0, 60.0, 1.0, train_s=600.0))
    assert not any("batch size" in t for t in tips(2.0, 60.0, 1.0, train_s=30.0))
    # Data-wait tips still appear in short runs
    assert any("waiting for data" in t for t in tips(25.0, 60.0, 1.0, train_s=30.0))


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


def test_resume_mid_epoch_matches_uninterrupted(tmp_path, four_cores, cpu_only, monkeypatch):
    monkeypatch.setattr(dl, "_cpu_count", lambda: 2)    # also exercise the worker loader
    full_run, full_model = _train(tmp_path / "full", epochs=3)

    _train(tmp_path / "split", epochs=3, stop_after=13)  # stops in the middle of epoch 2
    resumed, model = _train(tmp_path / "split", epochs=3)

    assert resumed._resumed_from == (1, 5)
    assert resumed.state.step == full_run.state.step == 24
    for a, b in zip(model.parameters(), full_model.parameters()):
        assert torch.allclose(a, b)


# ---------- Colab/Kaggle robustness ----------

def test_sigterm_handler_restored_after_loop(tmp_path, cpu_only):
    """After training, SIGTERM must stop the process (or kernel) again as before."""
    import signal

    import efficio
    before = signal.getsignal(signal.SIGTERM)
    run = efficio.Run(checkpoint_dir=str(tmp_path), resume=False, verbose=False)
    model = nn.Linear(1, 4)
    run.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1))
    for x, y in run.loop(DataLoader(_data(16), batch_size=8), epochs=1):
        assert signal.getsignal(signal.SIGTERM) is not before     # ours while training
        run.backward(nn.functional.cross_entropy(model(x), y % 4))
        run.step()
    assert signal.getsignal(signal.SIGTERM) == before


def test_report_after_resuming_a_finished_run(tmp_path, cpu_only, capsys):
    """Resumed at the very end: no batches, no steps, and the report must not divide by zero."""
    import efficio
    for _ in range(2):
        run = efficio.Run(checkpoint_dir=str(tmp_path), verbose=False)
        model = nn.Linear(1, 4)
        run.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1))
        for x, y in run.loop(DataLoader(_data(16), batch_size=8), epochs=1):
            run.backward(nn.functional.cross_entropy(model(x), y % 4))
            run.step()
        run.report()
    assert run._resumed_from == (1, 0) and run.profiler.steps == 0
    assert "data-wait       : 0%" in capsys.readouterr().out


def test_loop_twice_does_not_repeat_optimizations(tmp_path, cpu_only, four_cores, monkeypatch):
    import efficio
    monkeypatch.setattr(dl, "_cpu_count", lambda: 1)
    run = efficio.Run(checkpoint_dir=str(tmp_path), resume=False, verbose=False)
    model = nn.Linear(1, 4)
    run.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1))
    loader = DataLoader(_data(16), batch_size=8)
    for epochs in (1, 2):
        for x, y in run.loop(loader, epochs=epochs):
            run.backward(nn.functional.cross_entropy(model(x), y % 4))
            run.step()
    assert run.state.step == 4
    assert sum(o.startswith("dataloader") for o in run.optimizations) == 1


def test_gpu_sampler_gives_up_without_nvidia_smi():
    from efficio.report.profiler import GpuUtilSampler
    s = GpuUtilSampler(interval=0.01)
    s.CMD = ["efficio-no-such-nvidia-smi"]
    s.start()
    s._thread.join(timeout=5)
    assert not s._thread.is_alive() and s.available is False and s.average is None
    s.start()                       # no new thread once we know it is missing
    assert not s._thread.is_alive()
    s.stop()


# ---------- data-exact resume with a shuffled DataLoader ----------

class _NoisyData(torch.utils.data.Dataset):
    """Adds random noise in __getitem__, like a CPU augmentation, using the global torch RNG."""

    def __init__(self, n=64):
        self.base = _data(n)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x + 0.1 * torch.rand(1), y


def _train_shuffled(ckpt_dir, epochs, dataset, optimize, stop_after=None):
    import efficio
    torch.manual_seed(0)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = nn.Linear(1, 64)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    run = efficio.Run(checkpoint_dir=str(ckpt_dir), verbose=False, optimize_data=optimize)
    model, opt = run.prepare(model, opt)
    seen = []
    for n, (x, y) in enumerate(run.loop(loader, epochs=epochs)):
        seen.append(y.tolist())
        run.backward(nn.functional.cross_entropy(model(x), y))
        run.step()
        if stop_after is not None and n + 1 == stop_after:
            run.save()
            break
    return run, model, seen


@pytest.mark.parametrize("optimize,noisy", [(False, True), (True, False)],
                         ids=["no-workers+cpu-augment", "optimized-2-workers"])
def test_resume_shuffled_loader_is_data_exact(tmp_path, cpu_only, four_cores, monkeypatch,
                                               optimize, noisy):
    monkeypatch.setattr(dl, "_cpu_count", lambda: 2)
    make = (lambda: _NoisyData(64)) if noisy else (lambda: _data(64))
    full_run, full_model, full_seen = _train_shuffled(tmp_path / "full", 3, make(), optimize)

    _, _, first = _train_shuffled(tmp_path / "split", 3, make(), optimize, stop_after=13)
    resumed, model, rest = _train_shuffled(tmp_path / "split", 3, make(), optimize)

    assert resumed._resumed_from == (1, 5)
    assert first + rest == full_seen                     # same batches in the same order
    assert resumed.state.step == full_run.state.step == 24
    for a, b in zip(model.parameters(), full_model.parameters()):
        assert torch.equal(a, b)


def test_seeded_shuffle_order(four_cores):
    loader = DataLoader(_data(40), batch_size=8, shuffle=True)
    a, set_a = dl.seeded_shuffle(loader, base_seed=7)
    b, set_b = dl.seeded_shuffle(loader, base_seed=7)
    orders = []
    for epoch in (0, 0, 1):
        set_a(epoch)
        orders.append(torch.cat([y for _, y in a]).tolist())
    set_b(1)
    assert orders[0] == orders[1] != orders[2]
    assert torch.cat([y for _, y in b]).tolist() == orders[2]
    assert sorted(orders[0]) == list(range(40))
    assert set_a.base_seed == 7
    # Unshuffled loaders are left alone
    plain = DataLoader(_data(8), batch_size=4)
    assert dl.seeded_shuffle(plain, 1) == (plain, None)


def test_old_checkpoint_without_extra_still_loads(tmp_path):
    from efficio.checkpoint import CheckpointManager
    from efficio.interfaces import TrainState
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    mgr = CheckpointManager(str(tmp_path))
    path = mgr.save(TrainState(step=1), model, opt)
    payload = torch.load(path, weights_only=False)
    del payload["extra"]                                  # what older checkpoints look like
    torch.save(payload, path)
    assert mgr.load(model, opt)["step"] == 1 and mgr.extra == {}
