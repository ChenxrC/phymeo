"""Backend selection is explicit; importing a backend is not a GPU result."""

import sys
from types import ModuleType

import pytest

from phymeo.backends.mujoco_runtime import MuJoCoRuntime


@pytest.mark.parametrize(
    "mode,implementation,expected",
    [
        ("cuda", None, "tyx"),
        ("cuda", "tyx", "tyx"),
        ("interpret", None, "tyx"),
        ("cuda", "warp", "warp"),
    ],
)
def test_explicit_backend_selection(monkeypatch, mode, implementation, expected):
    class FakeTyx(MuJoCoRuntime):
        def __init__(self, compilation, **options):
            self.selected = "tyx"

    class FakeWarp(MuJoCoRuntime):
        def __init__(self, compilation, **options):
            self.selected = "warp"

    for suffix, name, cls in (
        ("tyx", "MuJoCoTyxRuntime", FakeTyx),
        ("warp", "MuJoCoWarpRuntime", FakeWarp),
    ):
        module = ModuleType(f"phymeo.backends._mujoco_{suffix}_runtime")
        setattr(module, name, cls)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime = MuJoCoRuntime(object(), mode=mode, implementation=implementation)
    assert runtime.selected == expected


@pytest.mark.parametrize(
    "mode,implementation",
    [
        ("cpu", "tyx"),
        ("cpu", "warp"),
        ("cuda", "native"),
        ("interpret", "warp"),
        ("cuda", "auto"),
        ("invalid", None),
    ],
)
def test_invalid_combination_never_falls_back(mode, implementation):
    with pytest.raises(ValueError):
        MuJoCoRuntime(object(), mode=mode, implementation=implementation)
