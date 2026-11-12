"""R3-P fixed FREE+BALL pyramidal sources and independent native qualification.

Tests-first authoring; only the parent may execute this matrix.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "free_ball_pyramidal_consumer",
    Path(__file__).parent / "installed" / "check_free_ball_pyramidal.py",
)
case = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(case)
MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
CASES = ("feet", "siblings")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", CASES)
def test_free_ball_pyramidal_complete_native_24_steps(name, mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "continuous.jsonl.gz")
    try:
        case.continuous(name, mode, journal)
    finally:
        journal.close()


NEGATIVES = {
    "activation": "free-ball-activation",
    "condim4": "free-ball-contact",
    "condim6": "free-ball-contact",
    "elliptic": "constraint-cone",
    "ball-limit": "free-ball-constraint",
    "scalar-limit": "free-ball-constraint",
    "dry": "free-ball-constraint",
    "spring": "free-ball-spring",
    "free-target": "actuator-domain",
    "ball-affine": "actuator-domain",
    "parent-transmission": "actuator-domain",
    "explicit-pair": "explicit-pairs",
    "exclude": "explicit-excludes",
    "cylinder": "contact-domain",
    "pair-overflow": "contact-layout-capacity",
    "slot-overflow": "contact-layout-capacity",
    "geom-overflow": "contact-layout-capacity",
}


@pytest.mark.parametrize("negative", tuple(NEGATIVES))
def test_free_ball_pyramidal_unsupported_precedes_device(negative, monkeypatch, tmp_path):
    mj = pytest.importorskip("mujoco")
    import phymeo.backends._mujoco_tyx_runtime as runtime_module
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    xml = case.negative_source(negative)
    model = mj.MjModel.from_xml_string(xml)
    assert model.nq > model.nv >= 9
    compiled = compile_mujoco_xml(xml, require_asset=False)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("R3-P unsupported source reached device allocation")

    monkeypatch.setattr(runtime_module.MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises(MuJoCoTyxUnsupported) as caught:
        MuJoCoRuntime(compiled, mode="interpret", implementation="tyx")
    assert calls == []
    assert NEGATIVES[negative] in {issue.code for issue in caught.value.issues}
    (tmp_path / "source.xml").write_text(xml, encoding="utf-8")
    (tmp_path / "rejection.txt").write_text(repr(caught.value), encoding="utf-8")


@pytest.mark.parametrize("negative", ("condim4", "condim6", "elliptic", "activation", "ball-limit", "dry"))
def test_free_ball_pyramidal_direct_compiler_flags_cannot_bypass_domain(negative):
    mj = pytest.importorskip("mujoco")
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported
    from phymeo.backends.mujoco_tyx._floating_model import compile_floating_model

    model = mj.MjModel.from_xml_string(case.negative_source(negative))
    with pytest.raises(MuJoCoTyxUnsupported) as caught:
        compile_floating_model(
            mj, model, allow_normal_contacts=True, allow_constraints=True, allow_activation=True
        )
    assert NEGATIVES[negative] in {issue.code for issue in caught.value.issues}


@pytest.mark.parametrize("capacity", ("pairs", "slots"))
def test_free_ball_pyramidal_original_static_capacity_boundary(capacity):
    mj = pytest.importorskip("mujoco")
    from phymeo.backends.mujoco_tyx._model import compile_model
    from phymeo.backends.mujoco_tyx._normal_model import possible_pairs

    model = mj.MjModel.from_xml_string(case.capacity_source(capacity, overflow=False))
    plan = compile_model(mj, model)
    assert plan.constraint_contact and plan.base.kind == 3 and plan.native_joint_layout
    assert len(possible_pairs(model)) == (16 if capacity == "pairs" else 4)
    assert len(plan.slot_ids) == (16 if capacity == "pairs" else 32)
    assert plan.nq == model.nq and plan.nv == model.nv
    assert plan.joint_transmission.moment.shape == (3,)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("location", ("root", "child"))
@pytest.mark.parametrize("bad", ("zero", "nonunit"))
def test_free_ball_pyramidal_invalid_quaternion_before_device(
    mode, location, bad, monkeypatch, tmp_path
):
    mj = pytest.importorskip("mujoco")
    import phymeo.backends._mujoco_tyx_runtime as runtime_module
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState, MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    xml = case.source("feet")
    model = mj.MjModel.from_xml_string(xml)
    start = case.initial(mj, model, "feet", 0)
    q = start.q.copy()
    joint = 0 if location == "root" else 1
    adr = int(model.jnt_qposadr[joint]) + (3 if location == "root" else 0)
    q[adr : adr + 4] = np.zeros(4) if bad == "zero" else (1.1, 0, 0, 0)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid quaternion reached device allocation")

    monkeypatch.setattr(runtime_module.MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises(ValueError):
        MuJoCoRuntime(
            compile_mujoco_xml(xml, require_asset=False),
            mode=mode,
            implementation="tyx",
            world_keys=case.KEYS,
            initial_states={
                key: MuJoCoInitialState(q, start.v, time=start.time) for key in case.KEYS
            },
        )
    assert calls == []
    journal = case.Journal(tmp_path / "bad-quaternion.jsonl.gz")
    try:
        journal.emit("rejected", source=xml, q=q, v=start.v, location=location, bad=bad)
    finally:
        journal.close()


@pytest.mark.parametrize("cone", ("pyramidal", "elliptic"))
@pytest.mark.parametrize("mixed_three", (False, True))
def test_free_ball_pyramidal_priority_mixing_controls_route(cone, mixed_three):
    """Source priorities, not authored default condim, decide the route."""
    mj = pytest.importorskip("mujoco")
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported
    from phymeo.backends.mujoco_tyx._model import compile_model

    xml = case.source("feet").replace('cone="pyramidal"', f'cone="{cone}"')
    xml = xml.replace('name="ground"', 'name="ground" condim="1" priority="1"')
    if mixed_three:
        xml = xml.replace('name="toe0"', 'name="toe0" condim="3" priority="2"')
    model = mj.MjModel.from_xml_string(xml)
    if mixed_three and cone == "elliptic":
        with pytest.raises(MuJoCoTyxUnsupported) as caught:
            compile_model(mj, model)
        assert "constraint-cone" in {issue.code for issue in caught.value.issues}
    else:
        plan = compile_model(mj, model)
        assert plan.base.kind == 3 and plan.native_joint_layout
        assert bool(getattr(plan, "constraint_contact", False)) == mixed_three
        if mixed_three:
            assert set(map(int, plan.pair_condim)) == {1, 3}
        else:
            assert plan.normal_contact


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("capacity", ("pairs", "slots"))
def test_free_ball_pyramidal_exact_static_capacity_real_step(capacity, mode, tmp_path):
    """The original 16-pair/32-slot boundary is actually consumed, not just admitted."""
    mj = pytest.importorskip("mujoco")
    from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    xml = case.capacity_source(capacity, overflow=False)
    model = mj.MjModel.from_xml_string(xml)
    journal = case.Journal(tmp_path / "capacity.jsonl.gz")
    try:
        with MuJoCoRuntime(
            compile_mujoco_xml(xml, require_asset=False),
            mode=mode,
            implementation="tyx",
            world_keys=case.KEYS,
        ) as runtime:
            states = runtime.integration_snapshot()
            packets = {
                k: MuJoCoControl(
                    np.zeros(model.nu),
                    qfrc_applied=np.zeros(model.nv),
                    xfrc_applied=np.zeros((model.nbody, 6)),
                )
                for k in case.KEYS
            }
            journal.emit("capacity-input", source=xml, state=states, capacity=capacity)
            case.advance(runtime, mj, model, packets, case.references(mj, model, states),
                         journal, "exact-static-capacity")
            assert runtime.counters["accepted_world_steps"] == len(case.KEYS)
    finally:
        journal.close()