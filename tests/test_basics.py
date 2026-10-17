import os

import pytest

from efficio.interfaces import Budget, HardwareInfo
from efficio.env.detect import detect_platform


def test_budget_parse():
    assert Budget.parse("2h").total_seconds == 7200
    assert Budget.parse("90m").total_seconds == 5400
    assert Budget.parse("1h30m").total_seconds == 5400
    assert Budget.parse("45s").total_seconds == 45
    assert Budget.parse(None).remaining() == float("inf")
    with pytest.raises(ValueError):
        Budget.parse("two hours")


def test_precision_choice():
    assert HardwareInfo("local", "cpu").precision == "fp32"
    assert HardwareInfo("colab", "cuda", "Tesla T4", 1, 15.0, (7, 5), False).precision == "fp16"
    assert HardwareInfo("local", "cuda", "RTX 4090", 1, 24.0, (8, 9), True).precision == "bf16"


def test_detect_kaggle(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert detect_platform() == "kaggle"


def test_resume_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from efficio.checkpoint import CheckpointManager
    from efficio.interfaces import TrainState

    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters())
    model(torch.randn(3, 4)).sum().backward()
    opt.step()

    mgr = CheckpointManager(str(tmp_path), keep_last=2)
    for step in (1, 2, 3):
        mgr.save(TrainState(epoch=1, step=step), model, opt)
    assert len(os.listdir(tmp_path)) == 2            # old ones cleaned up

    model2 = torch.nn.Linear(4, 2)
    opt2 = torch.optim.AdamW(model2.parameters())
    state = mgr.load(model2, opt2)
    assert state["step"] == 3
    assert torch.equal(model.weight, model2.weight)
