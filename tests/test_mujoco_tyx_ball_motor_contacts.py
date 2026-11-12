"""BALL-MC fixed sources and independent native force/motion qualification."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ball_motor_contact_consumer",
    Path(__file__).parent / "installed" / "check_ball_motor_contacts.py",
)
case = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(case)

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
CASES = (
    "mc1-normal",
    "mc2-friction",
    "mc3-mixed",
    "mc4-shared",
    "mc5-two-ball",
    "mc6-disabled",
)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", CASES)
def test_ball_motor_contact_continuous(name, mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "ball-motor-contact.jsonl.gz")
    try:
        case.continuous(name, mode, journal)
    finally:
        journal.close()


NEGATIVES = (
    "activation",
    "scalar-activation",
    "free",
    "ball-limit",
    "scalar-limit",
    "dry",
    "ball-spring",
    "elliptic",
    "condim4",
    "condim6",
    "affine-ball",
    "parent-transmission",
    "actor-overflow",
    "pair-overflow",
    "slot-overflow",
)


@pytest.mark.parametrize("negative", NEGATIVES)
def test_ball_motor_contact_rejected_before_device(negative, monkeypatch, tmp_path):
    mj = pytest.importorskip("mujoco")
    import phymeo.backends._mujoco_tyx_runtime as runtime_module
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported

    xml = case.negative_source(negative)
    model = mj.MjModel.from_xml_string(xml)
    assert model.nq >= 4 and model.nv >= 3
    compiled = compile_mujoco_xml(xml, require_asset=False)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unsupported BALL-MC reached device allocation")

    monkeypatch.setattr(runtime_module.MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises(MuJoCoTyxUnsupported) as caught:
        MuJoCoRuntime(compiled, mode="interpret", implementation="tyx")
    assert not calls
    expected_codes = {
        "activation": "joint-motor-activation",
        "scalar-activation": "joint-motor-activation",
        "free": "joint-domain",
        "ball-limit": "ball-constraint-domain",
        "scalar-limit": "ball-constraint-domain",
        "dry": "ball-constraint-domain",
        "ball-spring": "ball-spring",
        "elliptic": "ball-contact-cone",
        "condim4": "ball-contact-dimension",
        "condim6": "ball-contact-dimension",
        "affine-ball": "ball-actuator-domain",
        "parent-transmission": "ball-actuator-domain",
        "actor-overflow": "ball-actuator-capacity",
        "pair-overflow": "ball-contact-capacity",
        "slot-overflow": "ball-contact-capacity",
    }
    assert expected_codes[negative] in {issue.code for issue in caught.value.issues}
    (tmp_path / "source.xml").write_text(xml, encoding="utf-8")
    (tmp_path / "rejection.txt").write_text(repr(caught.value), encoding="utf-8")


@pytest.mark.parametrize("mode", MODES)
def test_previous_exact_motor_contact_negative_now_supported(mode, tmp_path):
    # This is the exact original M1 "contact" XML, including native defaults.
    # It is separated at default entry: qualification must not invent a load.
    pytest.importorskip("mujoco")
    from test_mujoco_tyx_ball_joint_motor import negative_source as old_source

    xml = old_source("contact")
    journal = case.Journal(tmp_path / "legacy-motor-contact.jsonl.gz")
    try:
        case.legacy_default(xml, mode, journal)
    finally:
        journal.close()
