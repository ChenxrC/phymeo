"""R1 fixed sources and original native activation/force/motion gates; tests first."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ball_activation_consumer",
    Path(__file__).parent / "installed" / "check_ball_activation.py",
)
case = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(case)

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
CASES = (
    "r1-late",
    "r2-early",
    "r3-normal",
    "r4-pyramid",
    "r5-mixed",
    "r6-shared",
    "r7-two-ball",
    "r8-disabled",
)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", CASES)
def test_ball_activation_continuous(name, mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "ball-activation.jsonl.gz")
    try:
        case.continuous(name, mode, journal)
    finally:
        journal.close()


NEGATIVES = {
    "free": "free-ball-activation",
    "ball-limit": "ball-constraint-domain",
    "scalar-limit": "ball-constraint-domain",
    "dry": "ball-constraint-domain",
    "ball-spring": "ball-spring",
    "elliptic": "ball-contact-cone",
    "condim4": "ball-contact-dimension",
    "condim6": "ball-contact-dimension",
    "affine-ball": "activation-ball-bias",
    "parent-transmission": "activation-transmission",
    "filter": "activation-dynamics",
    "actor-overflow": "activation-capacity",
}


@pytest.mark.parametrize("negative", NEGATIVES)
def test_ball_activation_rejected_before_device(negative, monkeypatch, tmp_path):
    mj = pytest.importorskip("mujoco")
    import phymeo.backends._mujoco_tyx_runtime as runtime_module
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    xml = case.negative_source(negative)
    model = mj.MjModel.from_xml_string(xml)
    assert model.na > 0
    compiled = compile_mujoco_xml(xml, require_asset=False)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unsupported R1 reached device allocation")

    monkeypatch.setattr(runtime_module.MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises(MuJoCoTyxUnsupported) as caught:
        MuJoCoRuntime(compiled, mode="interpret", implementation="tyx")
    assert not calls
    assert NEGATIVES[negative] in {issue.code for issue in caught.value.issues}
    (tmp_path / "source.xml").write_text(xml, encoding="utf-8")
    (tmp_path / "rejection.txt").write_text(repr(caught.value), encoding="utf-8")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("old_negative", ("joint-motor", "activation", "scalar-activation"))
def test_exact_previous_motor_contact_activation_rejection_now_supported(
    old_negative, mode, tmp_path
):
    # Exact previous MC XML and native default entry, not another changed fixture.
    pytest.importorskip("mujoco")
    if old_negative == "joint-motor":
        from test_mujoco_tyx_ball_joint_motor import negative_source

        xml = negative_source("activation")
    else:
        from test_mujoco_tyx_ball_motor_contacts import case as previous

        xml = previous.negative_source(old_negative)
    journal = case.Journal(tmp_path / "legacy-activation.jsonl.gz")
    try:
        case.legacy_default(xml, mode, journal)
    finally:
        journal.close()
