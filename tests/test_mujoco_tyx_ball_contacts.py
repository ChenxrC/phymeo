"""BALL-C fixed source matrix. All numeric work is parent-executed."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location("ball_contact_consumer", Path(__file__).parent / "installed" / "check_ball_contacts.py")
case = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(case)

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
CASES = ("c1-loaded", "c1-inactive", "c2-birth", "c3-early", "c3-late", "c3-disabled", "c4-two-ball")

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", CASES)
def test_ball_contact_continuous(name, mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "ball-contact.jsonl.gz")
    try:
        case.continuous(name, mode, journal)
    finally:
        journal.close()

NEGATIVES = (
    "ball-spring", "ball-limit", "scalar-limit", "dry", "ball-motor",
    "free-ball", "same-body", "elliptic", "condim4", "condim6",
    "pair-overflow", "slot-overflow", "explicit-pair", "exclude",
)

@pytest.mark.parametrize("negative", NEGATIVES)
def test_ball_contact_domain_rejected_before_device(negative, monkeypatch, tmp_path):
    mj = pytest.importorskip("mujoco")
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    import phymeo.backends._mujoco_tyx_runtime as runtime_module
    xml = case.negative_source(negative)
    # Native compilation establishes that this is a valid native declaration.
    model = mj.MjModel.from_xml_string(xml)
    compiled = compile_mujoco_xml(xml, require_asset=False)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("device allocation reached for unsupported domain")
    monkeypatch.setattr(runtime_module.MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises((ValueError, NotImplementedError)) as caught:
        MuJoCoRuntime(compiled, mode="interpret", implementation="tyx")
    assert not calls
    (tmp_path / "rejection.txt").write_text(repr(caught.value), encoding="utf-8")

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("family", ("pairs", "slots"))
def test_ball_contact_exact_static_capacity(family, mode, tmp_path):
    mj = pytest.importorskip("mujoco")
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    xml = case.capacity_source(family, False)
    compiled = compile_mujoco_xml(xml, require_asset=False)
    model = mj.MjModel.from_xml_string(xml)
    with MuJoCoRuntime(compiled, mode=mode, implementation="tyx") as runtime:
        journal = case.Journal(tmp_path / "capacity.jsonl.gz")
        try:
            journal.emit("capacity", xml=xml, dimensions=(model.nq, model.nv), capabilities=runtime.capabilities,
                         initial=runtime.integration_snapshot())
            assert runtime.capabilities["domain"] == "fixed-ball-scalar-normal-contact-v1"
        finally:
            journal.close()
