"""CheckpointSession on its own, without efficio.Run: resume choice, saving and uploading,
the stop decision, the SIGTERM flag, auto-shutdown gating and the report lines. CPU only;
storage is a local folder."""

import os
import signal
import time

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from efficio.checkpoint.manager import (  # noqa: E402
    CheckpointManager,
    DifferentRunError,
    StorageInUseError,
    run_fingerprint,
)
from efficio.checkpoint.session import CheckpointSession, open_remote, size_text  # noqa: E402
from efficio.interfaces import Budget, TrainState  # noqa: E402


def _model(width=8, optimizer=torch.optim.SGD):
    torch.manual_seed(0)
    model = nn.Linear(width, width)
    return model, optimizer(model.parameters(), lr=0.1)


def _session(tmp_path, name="local", storage=None, model=None, opt=None, **kwargs):
    """A session over tmp_path/name, with the fingerprint of (model, opt) set."""
    ckpt = CheckpointManager(str(tmp_path / name))
    remote = (
        open_remote(str(storage), 3, ckpt.directory, "local", lambda m: None)
        if storage is not None
        else None
    )
    kwargs.setdefault("budget", Budget())
    session = CheckpointSession(ckpt, remote, log=lambda m: None, **kwargs)
    if model is not None:
        ckpt.fingerprint = run_fingerprint(model, opt)
    return session


# ---------- stop decision and signal ----------


def test_poll_saves_periodically_and_stops_on_budget_or_signal(tmp_path):
    session = _session(tmp_path, save_every=60)
    assert session.poll() is None and not session.should_stop
    session.last_save = time.time() - 61
    assert session.poll() == "save" and session.stop_reason is None
    session.budget = Budget(total_seconds=1.0, safety_margin_seconds=10.0)  # nearly used
    assert session.poll() == "stop" and session.stop_reason == "time budget nearly used"
    assert session.should_stop
    signalled = _session(tmp_path, name="sig", budget=Budget(1.0, 10.0))
    signalled.signal_received = True  # the signal wins over the budget
    assert signalled.poll() == "stop" and signalled.stop_reason == "shutdown signal"


def test_finished_is_not_an_early_stop(tmp_path):
    session = _session(tmp_path)
    session.stop_reason = "finished"
    assert not session.should_stop


def test_sigterm_handler_sets_the_flag_and_is_restored(tmp_path):
    session = _session(tmp_path)
    before = signal.getsignal(signal.SIGTERM)
    old = session.install_signal_handler()
    try:
        assert old == before
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)  # what a SIGTERM would call
        assert session.signal_received
    finally:
        CheckpointSession.restore_signal_handler(old)
    assert signal.getsignal(signal.SIGTERM) == before


# ---------- saving and resuming without storage ----------


def test_save_then_load_latest_resumes(tmp_path):
    model, opt = _model()
    session = _session(tmp_path, model=model, opt=opt)
    assert session.load_latest(model, opt) is None  # nothing saved yet
    started = session.last_save
    path = session.save(TrainState(epoch=1, step=4), model, opt, extra={"shuffle_seed": 2})
    assert os.path.basename(path) == "ckpt_step000000004.pt"
    assert session.saves == 1 and session.last_save >= started
    assert session.final_upload is None  # no storage
    fresh, fresh_opt = _model()
    again = _session(tmp_path, model=fresh, opt=fresh_opt)
    saved = again.load_latest(fresh, fresh_opt)
    assert saved["step"] == 4 and again.ckpt.extra == {"shuffle_seed": 2}


def test_different_run_moves_aside_with_auto_and_raises_with_true(tmp_path, capsys):
    model, opt = _model()
    _session(tmp_path, model=model, opt=opt).save(TrainState(step=3), model, opt)
    other, other_opt = _model(optimizer=torch.optim.Adam)  # another optimizer type
    strict = _session(tmp_path, model=other, opt=other_opt, resume=True)
    with pytest.raises(DifferentRunError):
        strict.load_latest(other, other_opt)
    assert strict.ckpt.latest() is not None  # nothing moved
    session = _session(tmp_path, model=other, opt=other_opt, resume="auto")
    assert session.load_latest(other, other_opt) is None
    out = capsys.readouterr().out
    assert "not resuming" in out and "moved 1 checkpoint file(s)" in out
    assert session.ckpt.latest() is None
    aside = [d for d in os.listdir(tmp_path / "local") if d.startswith("other_run_")]
    assert len(aside) == 1


# ---------- storage ----------


def test_blocking_save_uploads_and_a_new_machine_downloads_it(tmp_path):
    model, opt = _model()
    store = tmp_path / "store"
    session = _session(tmp_path, storage=store, model=model, opt=opt)
    session.check_storage_owner()  # empty storage: fine
    path = session.save(TrainState(epoch=2, step=7), model, opt)
    assert session.final_upload == (path, "ok")
    entry = session.remote.latest()
    assert entry["step"] == 7 and entry["fingerprint"] == session.ckpt.fingerprint
    fresh, fresh_opt = _model()
    elsewhere = _session(tmp_path, name="elsewhere", storage=store, model=fresh, opt=fresh_opt)
    resumed = elsewhere.resume_file()
    assert os.path.dirname(resumed) == elsewhere.ckpt.directory  # downloaded here
    assert elsewhere.load_latest(fresh, fresh_opt)["step"] == 7


def test_background_save_uploads_later(tmp_path):
    model, opt = _model()
    session = _session(tmp_path, storage=tmp_path / "store", model=model, opt=opt)
    session.save(TrainState(step=2), model, opt, blocking=False)
    session.remote.wait()
    assert session.remote.latest()["step"] == 2
    assert session.final_upload is None  # only blocking saves record it


def test_storage_owner_check_refuses_or_overwrites_another_run(tmp_path, capsys):
    model, opt = _model()
    store = tmp_path / "store"
    _session(tmp_path, storage=store, model=model, opt=opt).save(TrainState(step=1), model, opt)
    other, other_opt = _model(optimizer=torch.optim.Adam)
    refused = _session(tmp_path, name="b", storage=store, model=other, opt=other_opt)
    with pytest.raises(StorageInUseError):
        refused.check_storage_owner()
    allowed = _session(
        tmp_path, name="c", storage=store, model=other, opt=other_opt, overwrite_storage=True
    )
    allowed.check_storage_owner()
    assert "overwrite_storage=True" in capsys.readouterr().out
    assert allowed.remote.replace_fingerprints == {run_fingerprint(model, opt)["hash"]}
    assert allowed.resume_file() is None  # the other run's checkpoint is never resumed


def test_open_remote_refuses_the_checkpoint_folder(tmp_path):
    with pytest.raises(ValueError, match="different folder"):
        open_remote(str(tmp_path), 3, str(tmp_path), "local", lambda m: None)


# ---------- end of the run ----------


def test_shutdown_needs_colab_an_ended_run_and_a_confirmed_upload(tmp_path, capsys):
    model, opt = _model()
    logged = []
    session = _session(tmp_path, storage=tmp_path / "store", model=model, opt=opt)
    session.log = logged.append
    session.maybe_shut_down("colab", enabled=False)
    session.maybe_shut_down("kaggle", enabled=True)
    assert len(logged) == 1  # Kaggle: only a note
    session.maybe_shut_down("colab", enabled=True)
    assert "has not ended" in capsys.readouterr().out
    session.stop_reason = "finished"
    session.save(TrainState(step=1), model, opt, blocking=False)  # not waited for
    session.remote.wait()
    session.maybe_shut_down("colab", enabled=True)
    assert "NOT shutting down" in capsys.readouterr().out


def test_report_lines_cover_checkpoints_and_storage(tmp_path):
    model, opt = _model()
    session = _session(tmp_path, storage=tmp_path / "store", model=model, opt=opt)
    session.save(TrainState(step=5), model, opt)
    text = "\n".join(session.report_lines())
    assert "checkpoint mode : exact, last file" in text
    assert "storage         : " in text and "last uploaded   : ckpt_step000000005.pt" in text
    assert _session(tmp_path, name="plain").report_lines() == ["  checkpoint mode : exact"]


def test_size_text():
    assert size_text(512) == "512 B"
    assert size_text(1536) == "1.5 KB"
    assert size_text(3 * 1024**3) == "3.0 GB"
