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
    "no-contact-late",
    "no-contact-early",
    "feet-normal",
    "feet-pyramidal",
    "siblings-normal",
    "siblings-pyramidal",
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
@pytest.mark.parametrize("name", ("r2", "r3-normal", "r3-pyramidal", "r1-free"))
def test_original_activation_rejection_inputs_now_take_two_steps(name, mode, tmp_path):
    pytest.importorskip("mujoco")
    if name == "r2":
        from test_mujoco_tyx_free_ball import negative_source as old_source

        expected = old_source("activation")
    elif name == "r1-free":
        from test_mujoco_tyx_ball_activation import case as old

        expected = old.negative_source("free")
    else:
        module = (
            "test_mujoco_tyx_free_ball_normal"
            if name == "r3-normal"
            else "test_mujoco_tyx_free_ball_pyramidal"
        )
        old = __import__(module, fromlist=["case"]).case
        expected = old.negative_source("activation")
    assert case.original_source(name) == expected
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


@pytest.mark.parametrize(
    "negative", tuple(name for name in NEGATIVES if name not in ("scalar-limit", "dry"))
)
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
        MuJoCoRuntime(
            compile_mujoco_xml(xml, require_asset=False), mode="interpret", implementation="tyx"
        )
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
        MuJoCoRuntime(
            compile_mujoco_xml(xml, require_asset=False),
            mode=mode,
            implementation="tyx",
            world_keys=case.KEYS,
            initial_states=states,
        )
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


@pytest.mark.parametrize(
    "corruption",
    (
        "duplicate-address",
        "missing-address",
        "none-address",
        "wrong-count",
        "tau-zero",
        "tau-nan",
        "gain-inf",
        "ctrl-reversed",
        "act-reversed",
    ),
)
def test_native_activation_layout_and_parameters_refused(corruption, tmp_path):
    mj = pytest.importorskip("mujoco")
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported
    from phymeo.backends.mujoco_tyx._model import compile_model

    model = mj.MjModel.from_xml_string(case.source("no-contact-late"))
    active = np.flatnonzero(model.actuator_actadr >= 0)
    a, b = map(int, active[:2])
    expected = "activation-layout"
    if corruption == "duplicate-address":
        model.actuator_actadr[b] = model.actuator_actadr[a]
    elif corruption == "missing-address":
        model.actuator_actadr[a] = -1
    elif corruption == "none-address":
        model.actuator_actadr[np.flatnonzero(model.actuator_actadr < 0)[0]] = 0
    elif corruption == "wrong-count":
        model.actuator_actnum[a] = 2
    elif corruption in ("tau-zero", "tau-nan"):
        model.actuator_dynprm[a, 0] = 0 if corruption == "tau-zero" else np.nan
        expected = "activation-time-constant"
    elif corruption == "gain-inf":
        model.actuator_gainprm[a, 0] = np.inf
        expected = "activation-parameters"
    elif corruption == "ctrl-reversed":
        model.actuator_ctrlrange[a] = (1, -1)
        expected = "activation-range"
    else:
        model.actuator_actrange[a] = (1, -1)
        expected = "activation-range"
    journal = case.Journal(tmp_path / "mutated-native-layout.jsonl.gz")
    try:
        journal.emit(
            "input",
            corruption=corruption,
            actadr=model.actuator_actadr.copy(),
            actnum=model.actuator_actnum.copy(),
            dynprm=model.actuator_dynprm.copy(),
            gain=model.actuator_gainprm.copy(),
            ctrlrange=model.actuator_ctrlrange.copy(),
            actrange=model.actuator_actrange.copy(),
        )
        with pytest.raises(MuJoCoTyxUnsupported) as caught:
            compile_model(mj, model)
        journal.emit("rejected", type=type(caught.value).__name__, reason=str(caught.value))
        assert expected in {issue.code for issue in caught.value.issues}
    finally:
        journal.close()


@pytest.mark.parametrize("negative", ("condim4", "condim6", "elliptic", "ball-limit"))
def test_allow_flags_do_not_bypass_free_ball_structure(negative):
    mj = pytest.importorskip("mujoco")
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported
    from phymeo.backends.mujoco_tyx._floating_model import compile_floating_model

    model = mj.MjModel.from_xml_string(case.negative_source(negative))
    with pytest.raises(MuJoCoTyxUnsupported) as caught:
        compile_floating_model(
            mj, model, allow_normal_contacts=True, allow_constraints=True, allow_activation=True
        )
    assert NEGATIVES[negative] in {issue.code for issue in caught.value.issues}


@pytest.mark.parametrize("name", CASES)
def test_source_mechanics_and_original_entry_control_are_unchanged(name):
    """Compare original declarations, not production-derived expectations."""
    import xml.etree.ElementTree as ET

    mj = pytest.importorskip("mujoco")
    if name.startswith("no-contact"):
        from test_mujoco_tyx_free_ball import control as old_control
        from test_mujoco_tyx_free_ball import initial as old_initial
        from test_mujoco_tyx_free_ball import source as old_source

        original = old_source("mixed")
    else:
        module = (
            "test_mujoco_tyx_free_ball_normal"
            if name.endswith("-normal")
            else "test_mujoco_tyx_free_ball_pyramidal"
        )
        old = __import__(module, fromlist=["case"]).case
        original = old.source(name.split("-")[0])
    root, previous = ET.fromstring(case.source(name)), ET.fromstring(original)
    for tag in ("compiler", "worldbody", "default"):
        left, right = root.find(tag), previous.find(tag)
        assert (None if left is None else ET.tostring(left)) == (
            None if right is None else ET.tostring(right)
        )
    options = dict(root.find("option").attrib)
    if name.startswith("siblings"):
        assert options.pop("actuatorgroupdisable") == "1"
    assert options == previous.find("option").attrib
    model = mj.MjModel.from_xml_string(case.source(name))
    for slot in range(2):
        actual = case.initial(mj, model, name, slot)
        expected = (
            old_initial(mj, model, slot)
            if name.startswith("no-contact")
            else old.initial(mj, model, name.split("-")[0], slot)
        )
        np.testing.assert_array_equal(actual.q, expected.q)
        np.testing.assert_array_equal(actual.v, expected.v)
        assert actual.time == expected.time
        for tick in (0, 8, 16):
            command = (
                old_control(model, tick, slot)
                if name.startswith("no-contact")
                else old.control(model, tick, slot)
            )
            case.exact(case.control(model, tick, slot), command, "original three-channel control")
