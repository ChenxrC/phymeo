"""BALL-C fixed source matrix. All numeric work is parent-executed."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ball_contact_consumer", Path(__file__).parent / "installed" / "check_ball_contacts.py"
)
case = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(case)

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
CASES = (
    "c1-loaded",
    "c1-inactive",
    "c2-birth",
    "c3-early",
    "c3-late",
    "c3-disabled",
    "c4-two-ball",
)


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
    "ball-spring",
    "ball-limit",
    "scalar-limit",
    "dry",
    "ball-motor",
    "free-ball",
    "same-body",
    "elliptic",
    "condim4",
    "condim6",
    "pair-overflow",
    "slot-overflow",
    "explicit-pair",
    "exclude",
)


@pytest.mark.parametrize("negative", tuple(name for name in NEGATIVES if name not in ("scalar-limit", "dry")))
def test_ball_contact_domain_rejected_before_device(negative, monkeypatch, tmp_path):
    mj = pytest.importorskip("mujoco")
    import phymeo.backends._mujoco_tyx_runtime as runtime_module
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    xml = case.negative_source(negative)
    # Native compilation establishes that this is a valid native declaration.
    model = mj.MjModel.from_xml_string(xml)
    assert model.nq >= 4 and model.nv >= 3
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
            journal.emit(
                "capacity",
                xml=xml,
                dimensions=(model.nq, model.nv),
                capabilities=runtime.capabilities,
                initial=runtime.integration_snapshot(),
            )
            assert runtime.capabilities["domain"] == "fixed-ball-scalar-normal-contact-v1"
            expected_slots = 16 if family == "pairs" else 32
            assert runtime.capabilities["normal_candidate_slots"] == expected_slots
            entry = runtime.integration_snapshot()
            refs = case.references(mj, model, entry)
            packets = {k: case.control(model, 0, i) for i, k in enumerate(runtime.world_keys)}
            result = case.advance(runtime, mj, model, packets, refs, journal, "exact-capacity-step")
            assert len(result[0].last_step.raw_contact_fields["slot_ids"]) == expected_slots
        finally:
            journal.close()
