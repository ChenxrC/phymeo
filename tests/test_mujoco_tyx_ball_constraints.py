"""BALL-S five-family qualification; fixed inputs before first execution."""

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ball_constraints_consumer", Path(__file__).parent / "installed" / "check_ball_constraints.py"
)
case = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(case)
MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
PROMOTIONS = (
    ("ball", "dry"), ("ball", "scalar-limit"), ("ball_joint_motor", "dry"),
    ("ball_contacts", "scalar-limit"), ("ball_contacts", "dry"),
    ("ball_motor_contacts", "scalar-limit"), ("ball_motor_contacts", "dry"),
    ("ball_scalar_actuation", "scalar-limit"), ("ball_scalar_actuation", "scalar-dry"),
    ("ball_activation", "scalar-limit"), ("ball_activation", "dry"),
    ("free_ball", "dry"), ("free_ball_normal", "scalar-limit"), ("free_ball_normal", "dry"),
    ("free_ball_pyramidal", "scalar-limit"), ("free_ball_pyramidal", "dry"),
    ("free_ball_activation", "scalar-limit"), ("free_ball_activation", "dry"),
)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", case.CASES)
def test_ball_scalar_complete_mixed_rows_and_continuous_native(name, mode, tmp_path):
    with case.saved(tmp_path / (name + ".jsonl.gz")) as journal:
        case.continuous(name, mode, journal)


def old_source(module, label):
    previous = importlib.import_module("test_mujoco_tyx_" + module)
    if module == "ball":
        return dict(previous.NEGATIVE_XML)[label]
    provider = previous.case if hasattr(previous, "case") else previous
    return provider.negative_source(label)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("module,label", PROMOTIONS, ids=lambda value: value)
def test_exact_retired_scalar_or_dry_negative_has_real_successor(module, label, mode, tmp_path):
    xml = old_source(module, label)
    with case.saved(tmp_path / "exact-original.jsonl.gz") as journal:
        case.original(xml, mode, journal, label=module + ":" + label)


@pytest.mark.parametrize("module", ("free_ball_normal", "free_ball_pyramidal", "free_ball_activation"))
def test_exact_direct_compiler_dry_admission_keeps_explicit_opt_in(module):
    import mujoco as mj
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported
    from phymeo.backends.mujoco_tyx._floating_model import compile_floating_model

    model = mj.MjModel.from_xml_string(old_source(module, "dry"))
    with pytest.raises(MuJoCoTyxUnsupported):
        compile_floating_model(mj, model, allow_normal_contacts=True, allow_activation=True)
    accepted = compile_floating_model(mj, model, allow_normal_contacts=True,
                                     allow_constraints=True, allow_activation=True)
    assert accepted.nv == model.nv and accepted.nq == model.nq


@pytest.mark.parametrize("bad", ("ball-limit", "free-target", "spring", "elliptic", "condim4",
                                 "condim6", "cylinder", "exclude", "explicit-pair", "multiple-free"))
def test_new_joint_route_does_not_bypass_original_structural_rejections(bad, monkeypatch, tmp_path):
    import mujoco as mj
    import phymeo.backends._mujoco_tyx_runtime as implementation
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    xml = case.negative_source(bad)
    model = mj.MjModel.from_xml_string(xml)
    assert model.nq > 0
    reached = []

    def forbidden(*args, **kwargs):
        reached.append(True)
        raise AssertionError("unsupported structural source allocated device")

    monkeypatch.setattr(implementation.MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises((ValueError, NotImplementedError)) as caught:
        MuJoCoRuntime(compile_mujoco_xml(xml, require_asset=False), mode="interpret")
    assert reached == []
    (tmp_path / "source.xml").write_text(xml, encoding="utf-8")
    (tmp_path / "rejection.txt").write_text(repr(caught.value), encoding="utf-8")


@pytest.mark.parametrize("free", (False, True))
@pytest.mark.parametrize("bad", ("negative", "nan", "inf"))
def test_nonfinite_or_negative_native_dry_parameter_rejected_before_selection(free, bad):
    import mujoco as mj
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported
    from phymeo.backends.mujoco_tyx._model import compile_model

    model = mj.MjModel.from_xml_string(case.source("free-filter-limits" if free else "dry-positive"))
    model.dof_frictionloss[-1] = {"negative": -.1, "nan": np.nan, "inf": np.inf}[bad]
    with pytest.raises(MuJoCoTyxUnsupported) as caught:
        compile_model(mj, model)
    assert "constraint-parameters" in {issue.code for issue in caught.value.issues}


def test_independent_small_box_oracle_fixed_signed_and_bound_faces():
    # A standalone algebra guard for the independent oracle, not product physics.
    matrix = np.array(((2., 1.), (1., 2.)))
    np.testing.assert_allclose(case.tiny_box_qp(matrix, np.array((-3., -1.)),
                                             np.zeros(2), np.full(2, np.inf)), (1.5, 0.), rtol=0, atol=1e-12)
    np.testing.assert_allclose(case.tiny_box_qp(np.eye(2), np.array((-.1, .1)),
                                             np.full(2, -.2), np.full(2, .2)), (.1, -.1), rtol=0, atol=1e-12)


@pytest.mark.parametrize("bad", ("range", "law", "margin"))
def test_invalid_scalar_limit_parameters_are_not_silently_ignored(bad):
    import mujoco as mj
    from phymeo.backends.mujoco_tyx import MuJoCoTyxUnsupported
    from phymeo.backends.mujoco_tyx._model import compile_model

    model = mj.MjModel.from_xml_string(case.source("fixed-lower"))
    joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "hinge")
    if bad == "range":
        model.jnt_range[joint] = (.25, -.25)
    elif bad == "law":
        model.jnt_solref[joint, 0] = np.nan
    else:
        model.jnt_margin[joint] = -.1
    with pytest.raises((ValueError, MuJoCoTyxUnsupported)):
        compile_model(mj, model)