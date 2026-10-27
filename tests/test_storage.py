"""Checkpoint storage: uploads, manifest, retention, verified downloads, resume on a new
machine, tokens and auto-shutdown. CPU only, no network: a fake backend, a temp folder and a
fake huggingface_hub module stand in for the real services."""

import hashlib
import json
import logging
import os
import pickle
import sys
import time
import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import efficio  # noqa: E402
from efficio.interfaces import HardwareInfo  # noqa: E402
from efficio.storage import (  # noqa: E402
    MANIFEST,
    HuggingFaceHub,
    LocalFolder,
    Storage,
    StorageError,
    open_storage,
)
from efficio.storage.hf_token import find_hf_token  # noqa: E402

SECRET = "hf_" + "QzT9secretTokenValue0123456789"


class FakeStorage(Storage):
    """In-memory storage. fail_uploads: that many checkpoint uploads fail (with partial,
    half the file is left behind under its name first); delay: seconds per upload."""

    def __init__(self):
        self.files, self.attempts = {}, []
        self.fail_uploads, self.partial, self.delay = 0, False, 0.0

    def describe(self):
        return "fake://bucket"

    def upload(self, local_path, name):
        time.sleep(self.delay)
        with open(local_path, "rb") as f:
            data = f.read()
        self.attempts.append(name)
        if self.fail_uploads and name != MANIFEST:
            self.fail_uploads -= 1
            if self.partial:
                self.files[name] = data[:len(data) // 2]
            raise ConnectionError("network down")
        self.files[name] = data

    def download(self, name, local_path):
        with open(local_path, "wb") as f:
            f.write(self.files[name])

    def list(self):
        return sorted(self.files)

    def delete(self, name):
        self.files.pop(name, None)


@pytest.fixture
def hardware(monkeypatch):
    """Pretend to run on a platform ("local" by default), always on the CPU."""
    import efficio.run as run_module

    def use(platform="local"):
        monkeypatch.setattr(run_module, "detect_hardware", lambda: HardwareInfo(platform, "cpu"))
    use()
    return use


def _data(n=64):
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(n, 8, generator=gen)
    return TensorDataset(x, (x.sum(1) > 0).long())


def _model(frozen=False):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))
    if frozen:
        model[0].requires_grad_(False)
    return model


def _train(ckpt_dir, storage=None, epochs=2, stop_at=None, frozen=False, **kwargs):
    """Train 4 steps per epoch. stop_at: the time budget runs out after that step (the run
    stops and saves at the next one)."""
    model = _model(frozen)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    loader = DataLoader(_data(), batch_size=16, shuffle=True)
    kwargs.setdefault("verbose", False)
    run = efficio.Run(checkpoint_dir=str(ckpt_dir), storage=storage, **kwargs)
    if run.remote is not None:
        run.remote.backoff = 0.0
    model, opt = run.prepare(model, opt)
    for x, y in run.loop(loader, epochs=epochs):
        run.backward(nn.functional.cross_entropy(model(x), y))
        run.step()
        if stop_at is not None and run.state.step == stop_at:
            run.budget.safety_margin_seconds = 0
            run.budget.total_seconds = max(run.budget.elapsed() - 1.0, 1e-6)
    return run, model


def _manifest(store):
    if isinstance(store, FakeStorage):
        return json.loads(store.files[MANIFEST])
    return store.read_manifest()


def _ckpts(store):
    return [n for n in store.list() if n.startswith("ckpt_step")]


# ---------- uploads and the manifest ----------

def test_upload_download_round_trip_and_manifest(tmp_path, hardware):
    store = FakeStorage()
    run, _ = _train(tmp_path / "local", store, epochs=1)
    entry, = _manifest(store)["checkpoints"]
    local = run.ckpt.stats["last_path"]
    with open(local, "rb") as f:
        data = f.read()
    assert entry["file"] == os.path.basename(local) == "ckpt_step000000004.pt"
    assert store.files[entry["file"]] == data
    assert entry["sha256"] == hashlib.sha256(data).hexdigest() and entry["size"] == len(data)
    assert (entry["step"], entry["epoch"], entry["batch_in_epoch"]) == (4, 1, 0)
    assert entry["mode"] == "exact" and entry["trainable_only"] is False
    assert entry["platform"] == "local" and entry["efficio_version"]
    assert time.strptime(entry["time"], "%Y-%m-%dT%H:%M:%SZ")

    path = run.remote.download(entry, str(tmp_path / "elsewhere"))
    with open(path, "rb") as f:
        assert f.read() == data


def test_local_folder_backend_copies_atomically(tmp_path, hardware):
    remote = tmp_path / "drive_folder"
    run, _ = _train(tmp_path / "local", str(remote), epochs=1)
    names = sorted(os.listdir(remote))
    assert names == ["ckpt_step000000004.pt", MANIFEST]          # no .partial files left
    assert run.remote.storage.read_manifest()["checkpoints"][0]["file"] == names[0]


def test_retention_keeps_the_newest_remote_checkpoints(tmp_path, hardware):
    store = FakeStorage()
    run, _ = _train(tmp_path / "local", store, epochs=2, keep_remote=2, save_every_minutes=0)
    entries = _manifest(store)["checkpoints"]
    assert [e["step"] for e in entries] == [7, 8]
    assert _ckpts(store) == [e["file"] for e in entries]
    assert run.remote.stats["uploads"] >= 8


def test_half_upload_never_becomes_latest(tmp_path, hardware, capsys):
    store = FakeStorage()
    _train(tmp_path / "a", store, epochs=1)                         # step 4 uploaded
    store.fail_uploads, store.partial = 99, True
    run, _ = _train(tmp_path / "a", store, epochs=2)                # step 8: upload breaks
    assert run.state.step == 8 and "UPLOAD FAILED" in capsys.readouterr().out
    assert [e["step"] for e in _manifest(store)["checkpoints"]] == [4]
    assert "ckpt_step000000008.pt" in store.files                   # the half file is there
    store.fail_uploads = 0
    fresh, _ = _train(tmp_path / "b", store, epochs=1)              # a new machine
    assert fresh.remote.stats["downloaded"]["step"] == 4
    assert fresh._resumed_from == (1, 0)


def test_sha256_mismatch_is_refused(tmp_path, hardware):
    store = FakeStorage()
    _train(tmp_path / "a", store, epochs=1)
    name = _manifest(store)["checkpoints"][-1]["file"]
    store.files[name] = store.files[name][:-10] + b"x" * 10           # damaged in storage
    with pytest.raises(StorageError, match="sha256"):
        _train(tmp_path / "b", store, epochs=2)
    assert os.listdir(tmp_path / "b") == []                         # nothing kept or loaded


# ---------- resuming from storage ----------

@pytest.mark.parametrize("mode,frozen", [("exact", False), ("light", False),
                                         ("smallest", False), ("exact", True),
                                         ("light", True)])
def test_resume_from_storage_on_an_empty_machine(tmp_path, hardware, mode, frozen):
    remote = str(tmp_path / "remote")
    stopped, _ = _train(tmp_path / "kaggle", remote, stop_at=5, frozen=frozen,
                        checkpoint_mode=mode)
    assert stopped.should_stop and stopped.state.step == 6
    entry = stopped.remote.latest()
    assert entry["step"] == 6 and entry["mode"] == mode and entry["trainable_only"] is frozen

    resumed, model = _train(tmp_path / "colab", remote, frozen=frozen, checkpoint_mode=mode)
    assert resumed.remote.stats["downloaded"]["file"] == entry["file"]
    assert resumed._resumed_from == (1, 2)
    assert resumed.state.step == 8

    reference, ref_model = _train(tmp_path / "reference", frozen=frozen)
    assert reference.state.step == 8
    if mode == "exact":
        for a, b in zip(model.parameters(), ref_model.parameters()):
            assert torch.equal(a, b)


def test_newer_remote_checkpoint_beats_older_local_one(tmp_path, hardware, capsys):
    remote = str(tmp_path / "remote")
    _train(tmp_path / "home", remote, epochs=1)                     # step 4 at home
    _train(tmp_path / "colab", remote, epochs=2)                    # step 8 in the cloud
    run, _ = _train(tmp_path / "home", remote, epochs=3, verbose=True)
    out = capsys.readouterr().out
    assert run.remote.stats["downloaded"]["step"] == 8 and "newer checkpoint" in out
    assert run.state.step == 12


def test_local_checkpoint_used_when_not_older(tmp_path, hardware):
    remote = str(tmp_path / "remote")
    _train(tmp_path / "home", remote, epochs=1)
    run, _ = _train(tmp_path / "home", remote, epochs=2)
    assert run.remote.stats["downloaded"] is None and run._resumed_from == (1, 0)


# ---------- errors, waiting and retries ----------

def test_upload_error_surfaces_at_next_step(tmp_path, hardware):
    store = FakeStorage()
    store.fail_uploads = 99
    with pytest.raises(RuntimeError, match="Uploading a checkpoint to fake://bucket failed"):
        _train(tmp_path / "local", store, epochs=2, save_every_minutes=0)
    assert os.listdir(tmp_path / "local")                           # the local save is fine


def test_budget_stop_waits_for_the_upload(tmp_path, hardware, capsys):
    store = FakeStorage()
    store.delay = 0.3
    run, _ = _train(tmp_path / "local", store, stop_at=2, verbose=True)
    assert not run.remote.pending                       # finished before training returned
    assert _manifest(store)["checkpoints"][-1]["step"] == run.state.step == 3
    assert "upload complete: ckpt_step000000003.pt" in capsys.readouterr().out


def test_final_save_waits_for_the_upload(tmp_path, hardware):
    store = FakeStorage()
    store.delay = 0.3
    run, _ = _train(tmp_path / "local", store, epochs=1)
    assert not run.remote.pending and run._final_upload[1] == "ok"
    assert _manifest(store)["checkpoints"][-1]["step"] == 4


def test_retries_with_backoff_then_success(tmp_path, hardware):
    store = FakeStorage()
    store.fail_uploads = 2
    run, _ = _train(tmp_path / "local", store, epochs=1)
    assert store.attempts.count("ckpt_step000000004.pt") == 3
    assert run._final_upload[1] == "ok" and run.remote.stats["last_ok"]


def test_slow_upload_is_reported_as_not_confirmed(tmp_path, hardware, capsys):
    store = FakeStorage()
    store.delay = 1.0
    run, _ = _train(tmp_path / "local", store, epochs=1, upload_timeout=0.05)
    assert run._final_upload[1] == "running"
    assert "upload NOT confirmed" in capsys.readouterr().out
    run.remote.wait()


def test_report_lines(tmp_path, hardware, capsys):
    run, _ = _train(tmp_path / "local", FakeStorage(), epochs=1)
    run.report()
    out = capsys.readouterr().out
    assert "storage         : fake://bucket (keeps the last 3)" in out
    assert "last uploaded   : ckpt_step000000004.pt" in out
    assert "s in the background (" in out and "hidden)" in out


def test_without_storage_nothing_changes(tmp_path, hardware, capsys):
    run, _ = _train(tmp_path / "local", epochs=1)
    run.report()
    out = capsys.readouterr().out
    assert run.remote is None and "  storage " not in out and "upload" not in out


# ---------- backends and settings ----------

def test_open_storage_parses_targets(tmp_path):
    hub = open_storage("hf://alice/my-run/exp1")
    assert isinstance(hub, HuggingFaceHub) and hub.describe() == "hf://alice/my-run/exp1"
    assert isinstance(open_storage(str(tmp_path)), LocalFolder)
    store = FakeStorage()
    assert open_storage(store) is store
    for bad in ("hf://alice", "s3://bucket/x"):
        with pytest.raises(ValueError):
            open_storage(bad)
    with pytest.raises(TypeError):
        open_storage(3)


@pytest.mark.skipif(os.path.isdir("/content/drive/MyDrive"), reason="Drive is mounted here")
def test_unmounted_google_drive_is_refused():
    with pytest.raises(StorageError, match=r"drive\.mount\('/content/drive'\)"):
        LocalFolder("/content/drive/MyDrive/efficio_run").connect()


def test_storage_folder_must_differ_from_checkpoint_dir(tmp_path, hardware):
    with pytest.raises(ValueError, match="different folder"):
        efficio.Run(checkpoint_dir=str(tmp_path), storage=str(tmp_path), verbose=False)


def test_remote_checkpoints_need_keep_of_at_least_one(tmp_path, hardware):
    with pytest.raises(ValueError, match="keep_remote"):
        efficio.Run(checkpoint_dir=str(tmp_path / "a"), storage=FakeStorage(), keep_remote=0)


# ---------- Hugging Face Hub (fake huggingface_hub module) ----------

@pytest.fixture
def fake_hub(monkeypatch):
    """A stand-in huggingface_hub module that keeps repos in memory, records the tokens it
    was given and can fail like a library that puts the token in its error message."""
    hub = types.ModuleType("huggingface_hub")
    hub.repos, hub.tokens, hub.calls, hub.fail_uploads = {}, [], [], 0
    hub.public = False

    class HfApi:
        def __init__(self, token=None):
            self.token = token
            hub.tokens.append(token)

        def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False):
            hub.calls.append(("create_repo", repo_id, private, exist_ok))
            hub.repos.setdefault(repo_id, {})

        def repo_info(self, repo_id, repo_type=None):
            return types.SimpleNamespace(private=not hub.public)

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None,
                        commit_message=None):
            if hub.fail_uploads:
                hub.fail_uploads -= 1
                raise RuntimeError(f"401 Client Error for token {self.token}")
            with open(path_or_fileobj, "rb") as f:
                hub.repos[repo_id][path_in_repo] = f.read()

        def list_repo_files(self, repo_id, repo_type=None):
            return sorted(hub.repos[repo_id])

        def delete_file(self, path_in_repo, repo_id, repo_type=None, commit_message=None):
            del hub.repos[repo_id][path_in_repo]

    def hf_hub_download(repo_id, filename, repo_type=None, token=None, local_dir=None,
                        force_download=False):
        hub.tokens.append(token)
        path = os.path.join(local_dir, *filename.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(hub.repos[repo_id][filename])
        return path

    hub.HfApi, hub.hf_hub_download = HfApi, hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return hub


def test_hub_creates_a_private_repo_and_resumes_elsewhere(tmp_path, hardware, fake_hub,
                                                          monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    hardware("kaggle")
    stopped, _ = _train(tmp_path / "kaggle", "hf://alice/run1/exp", stop_at=5)
    assert ("create_repo", "alice/run1", True, True) in fake_hub.calls
    assert sorted(fake_hub.repos["alice/run1"]) == ["exp/ckpt_step000000006.pt",
                                                    f"exp/{MANIFEST}"]
    hardware("colab")
    resumed, _ = _train(tmp_path / "colab", "hf://alice/run1/exp")
    assert resumed.remote.stats["downloaded"]["platform"] == "kaggle"
    assert resumed.state.step == 8
    assert set(fake_hub.tokens) == {SECRET}                  # the token reached the library


def test_hub_warns_about_a_public_repo(fake_hub, capsys):
    fake_hub.public = True
    HuggingFaceHub("alice/open", token="x").connect()
    assert "PUBLIC repo" in capsys.readouterr().out


def test_hub_needs_the_optional_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match=r'efficio\[hub\]'):
        HuggingFaceHub("alice/run1", token="x").connect()


@pytest.mark.parametrize("given", ["env", "argument"])
def test_token_never_leaks(tmp_path, hardware, fake_hub, monkeypatch, capsys, caplog, given):
    secret = SECRET if given == "env" else "plain-secret-without-hf-prefix-123"
    if given == "env":
        monkeypatch.setenv("HF_TOKEN", secret)
        storage = "hf://alice/run1"
    else:
        storage = HuggingFaceHub("alice/run1", token=secret)
    caplog.set_level(logging.DEBUG)
    fake_hub.fail_uploads = 2                                # retried, then it works
    run, _ = _train(tmp_path / "local", storage, epochs=1, verbose=True)
    run.report()
    fake_hub.fail_uploads = 99                               # now every upload fails
    failed, _ = _train(tmp_path / "local", run.remote.storage, epochs=2, verbose=True)
    failed.report()
    with pytest.raises(StorageError) as error:
        failed.remote._retry("x", lambda: (_ for _ in ()).throw(RuntimeError(secret)))
    assert secret in fake_hub.tokens                          # it was used...

    out = capsys.readouterr()
    texts = [out.out, out.err, caplog.text, str(error.value), repr(run.remote.storage),
             json.dumps(run.remote.stats), json.dumps(failed.remote.stats),
             pickle.dumps(run.remote.storage).decode("latin-1")]
    assert "UPLOAD FAILED" in out.out and "<hidden>" in out.out
    for name in os.listdir(tmp_path / "local"):              # local checkpoints
        path = tmp_path / "local" / name
        texts.append(path.read_bytes().decode("latin-1"))
        texts.append(repr(torch.load(path, weights_only=False)["meta"]))
    for data in fake_hub.repos["alice/run1"].values():       # remote files and the manifest
        texts.append(data.decode("latin-1"))
    assert all(secret not in text for text in texts)         # ...but never written or shown


def test_token_lookup_order(monkeypatch):
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda name: "from-colab")
    google.colab = colab
    kaggle = types.ModuleType("kaggle_secrets")
    kaggle.UserSecretsClient = lambda: types.SimpleNamespace(get_secret=lambda n: "from-kaggle")
    monkeypatch.setitem(sys.modules, "kaggle_secrets", kaggle)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert find_hf_token()[0] == "from-kaggle"
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    assert find_hf_token() == ("from-colab", "Colab secrets (HF_TOKEN)")
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert find_hf_token()[0] == "from-env"
    assert find_hf_token("explicit") == ("explicit", "the token argument")


# ---------- auto-shutdown ----------

@pytest.fixture
def colab_runtime(monkeypatch):
    calls = []
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    runtime = types.ModuleType("google.colab.runtime")
    runtime.unassign = lambda: calls.append("unassign")
    google.colab, colab.runtime = colab, runtime
    for name, module in (("google", google), ("google.colab", colab),
                         ("google.colab.runtime", runtime)):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_shutdown_after_a_confirmed_final_upload(tmp_path, hardware, colab_runtime):
    hardware("colab")
    run, _ = _train(tmp_path / "local", FakeStorage(), epochs=1, shutdown_when_done=True)
    assert colab_runtime == []                              # not before report()
    run.report()
    assert colab_runtime == ["unassign"]


def test_no_shutdown_when_the_upload_failed(tmp_path, hardware, colab_runtime, capsys):
    hardware("colab")
    store = FakeStorage()
    store.fail_uploads = 99
    run, _ = _train(tmp_path / "local", store, epochs=1, shutdown_when_done=True)
    run.report()
    assert colab_runtime == [] and "NOT shutting down" in capsys.readouterr().out


def test_no_shutdown_without_storage_or_when_not_asked(tmp_path, hardware, colab_runtime):
    hardware("colab")
    run, _ = _train(tmp_path / "a", epochs=1, shutdown_when_done=True)
    run.report()
    run, _ = _train(tmp_path / "b", FakeStorage(), epochs=1)
    run.report()
    assert colab_runtime == []


@pytest.mark.parametrize("platform,text", [("kaggle", "can't be stopped from code"),
                                           ("local", "nothing to shut down")])
def test_shutdown_not_supported_off_colab(tmp_path, hardware, colab_runtime, capsys,
                                          platform, text):
    hardware(platform)
    run, _ = _train(tmp_path / "local", FakeStorage(), epochs=1, shutdown_when_done=True,
                    verbose=True)
    run.report()
    assert colab_runtime == [] and text in capsys.readouterr().out
