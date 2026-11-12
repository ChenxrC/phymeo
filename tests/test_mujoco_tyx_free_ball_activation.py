"""R3-A fixed FREE+BALL FILTEREXACT qualification.

Tests-first AUTHORING_ONLY checkpoint. The parent alone executes this matrix.
The standalone consumer owns literal sources and independent native references.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "free_ball_activation_consumer",
    Path(__file__).parent / "installed" / "check_free_ball_activation.py",
)
case = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(case)
MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
CASES = (
    "no-contact-late", "no-contact-early", "feet-normal", "feet-pyramidal",
    "siblings-normal", "siblings-pyramidal",
)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", CASES)
def test_free_ball_activation_complete_native_24_steps(name, mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "continuous.jsonl.gz")
    try:
        case.continuous(name, mode, journal)
    finally:
        journal.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ("r2", "r3-normal", "r3-pyramidal"))
def test_original_activation_rejection_inputs_now_take_two_steps(name, mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "original-input.jsonl.gz")
    try:
        case.original_activation(name, mode, journal)
    finally:
        journal.close()


NEGATIVES = {
    "integrator-dynamics": "activation-dynamics",
    "filter-dynamics": "activation-dynamics",
    "muscle-dynamics": "activation-dynamics",
    "free-target": "activation-transmission",
    "parent-transmission": "activation-transmission",
    "ball-affine": "activation-ball-bias",
    "condim4": "free-ball-contact",
    "condim6": "free-ball-contact",
    "elliptic": "constraint-cone",
    "ball-limit": "free-ball-constraint",
    "scalar-limit": "free-ball-constraint",
    "dry": "free-ball-constraint",
    "spring": "free-ball-spring",
    "explicit-pair": "explicit-pairs",
    "exclude": "explicit-excludes",
    "cylinder": "contact-domain",
    "pair-overflow": "contact-layout-capacity",
    "slot-overflow": "contact-layout-capacity",
    "geom-overflow": "contact-layout-capacity",
    "actuator-overflow": "activation-capacity",
    "nv-overflow": "joint-domain",
}


@pytest.mark.parametrize("negative", tuple(NEGATIVES))
def test_activation_domain_rejected_before_device(negative, monkeypatch, tmp_path):
    mj = pytest.importorskip("mujoco")
    import phymeo.backends._mujoco_tyx_runtime as runtime_module
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    xml = case.negative_source(negative)
    model = mj.MjModel.from_xml_string(xml)
    assert model.nq > model.nv >= 9
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unsupported R3-A source reached device allocation")

    monkeypatch.setattr(runtime_module.MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises(MuJoCoTyxUnsupported) as caught:
        MuJoCoRuntime(compile_mujoco_xml(xml, require_asset=False), mode="interpret", implementation="tyx")
    assert calls == []
    assert NEGATIVES[negative] in {issue.code for issue in caught.value.issues}
    (tmp_path / "source.xml").write_text(xml, encoding="utf-8")
    (tmp_path / "rejection.txt").write_text(repr(caught.value), encoding="utf-8")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("bad", ("root-zero", "ball-nonunit", "act-shape", "act-nan"))
def test_initial_state_rejected_before_device(bad, mode, monkeypatch, tmp_path):
    mj = pytest.importorskip("mujoco")
    import phymeo.backends._mujoco_tyx_runtime as runtime_module
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState, MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    xml = case.source("no-contact-late")
    model = mj.MjModel.from_xml_string(xml)
    start = case.initial(mj, model, "no-contact-late", 0)
    q, act = start.q.copy(), start.act.copy()
    if bad == "root-zero":
        q[3:7] = 0
    elif bad == "ball-nonunit":
        qa = int(model.jnt_qposadr[1])
        q[qa : qa + 4] = (1.1, 0, 0, 0)
    elif bad == "act-shape":
        act = np.append(act, 0)
    else:
        act[0] = np.nan
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid activation/quaternion reached device allocation")

    monkeypatch.setattr(runtime_module.MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises(ValueError):
        states = {k: MuJoCoInitialState(q, start.v, act=act, time=start.time) for k in case.KEYS}
        MuJoCoRuntime(compile_mujoco_xml(xml, require_asset=False), mode=mode,
                      implementation="tyx", world_keys=case.KEYS, initial_states=states)
    assert calls == []
    journal = case.Journal(tmp_path / "invalid-initial.jsonl.gz")
    try:
        journal.emit("rejected", source=xml, q=q, v=start.v, act=act, bad=bad)
    finally:
        journal.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("capacity", ("pairs", "slots", "activation"))
def test_original_capacity_boundary_takes_real_step(capacity, mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "capacity.jsonl.gz")
    try:
        case.capacity(capacity, mode, journal)
    finally:
        journal.close()
