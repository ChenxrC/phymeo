"""T1 fixed-tree resident dynamics: frozen inputs, authoring only / UNEXECUTED.

The public RED constructs a valid existing RobotModel/DrivenRobotProgram first.
No expected FK, mass, bias, or load projection comes from Phymeo mechanics.
"""

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from functools import lru_cache
from math import pi

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends import tyx_world_batch as resident

RTOL = 5e-5
ATOL = 5e-6
WINDOW_ATOL = 2e-4


def _json(value):
    if isinstance(value, ph.DrivenRobotProgram):
        return _json(value.problem_signature())
    if is_dataclass(value):
        return {f.name: _json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


def _save(tmp_path, name, value):
    (tmp_path / (name + ".json")).write_text(
        json.dumps(_json(value), indent=2, allow_nan=False), encoding="utf-8"
    )


def _program(model, *, q=None, v=None, kp=None, kd=None, cap=None, delay=0):
    n = model.dof_count
    program = ph.DrivenRobotProgram(
        model,
        kp=np.zeros(n) if kp is None else np.asarray(kp, dtype=float),
        kd=np.zeros(n) if kd is None else np.asarray(kd, dtype=float),
        effort_limit=np.full(n, 8.0) if cap is None else np.asarray(cap, dtype=float),
        delay_steps=delay,
    )
    initial = program.initial_state(
        ph.RobotState(
            np.zeros(n) if q is None else np.asarray(q, dtype=float),
            np.zeros(n) if v is None else np.asarray(v, dtype=float),
        )
    )
    return program, initial


def _rr_model():
    return ph.RobotModel(
        np.array([-1, 0]),
        ("revolute", "revolute"),
        np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        np.tile([0.0, 0.0, 1.0], (2, 1)),
        np.ones(2),
        np.tile([0.5, 0.0, 0.0], (2, 1)),
        np.tile([1 / 24, 1 / 24, 1 / 12], (2, 1)),
        np.zeros(3),
    )


def _action(program, *, target=None, rate=None, ff=None, effort=None, force=None, torque=None):
    n = program.model.dof_count

    def vector(value):
        return np.zeros(n) if value is None else np.asarray(value, dtype=float)

    def field(value):
        return np.zeros((n, 3)) if value is None else np.asarray(value, dtype=float)

    return program.inputs(
        vector(target),
        vector(rate),
        feedforward=vector(ff),
        loads=ph.RobotInputs(vector(effort), field(force), field(torque)),
    )


def _tree(program, initial, *, iterations=512):
    return resident.FixedRobotTree(program, initial, control_max_iterations=iterations)


def _close(actual, expected, *, atol=ATOL):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=atol)


@pytest.mark.tyx
def test_t01_public_tree_step_has_real_coupled_two_revolute_response(tyx_mode, tmp_path):
    model = _rr_model()
    assert model.dof_count == 2 and tuple(model.parents) == (-1, 0)
    program, initial = _program(model)
    assert initial.domains[program.domain].q.tolist() == [0.0, 0.0]
    command = _action(program, effort=(1, 1 / 6))
    _save(
        tmp_path,
        "t01-public-input",
        {"model": model, "initial": initial, "command": command, "h_s": 1 / 128},
    )
    # First missing capability occurs here, not during collection/import.
    world = _tree(program, initial)
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: command})
        accepted = batch.step(1 / 128)
        actual = batch.snapshot()[7]
        _save(
            tmp_path,
            f"t01-public-{tyx_mode}",
            {
                "accepted": accepted,
                "actual": actual,
                "stats": batch.stats(),
                "oracle_mass": [[8 / 3, 5 / 6], [5 / 6, 1 / 3]],
                "oracle_acceleration": [1, -2],
            },
        )
        assert accepted == {7: True} and actual.tick == 1
        _close(actual.velocity, [1 / 128, -2 / 128])
        _close(actual.position, [1 / 128**2, -2 / 128**2])
        np.testing.assert_array_equal(actual.applied_effort, [0, 0])
        _close(actual.generalized_applied, [1, 1 / 6])
        read = batch.tree_kinematics([7])
        assert tuple(read) == (7,) and read[7].tick == 1
        assert read[7].joint_types == ("revolute", "revolute")
        _close(read[7].position, actual.position)
        _close(read[7].velocity, actual.velocity)


# Independent authored-model reference: import only its NumPy oracle at test time.
# Its CLI imports Newton separately and never imports Phymeo.
@lru_cache(maxsize=1)
def _reference():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "reference_newton/robot/resident_tree_reference.py"
    spec = importlib.util.spec_from_file_location("resident_tree_frozen_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ordinary_model(data):
    assert "fixed" not in data["kinds"]
    return ph.RobotModel(
        data["parents"],
        data["kinds"],
        data["offsets"],
        data["orientations"],
        data["axes"],
        data["mass"],
        data["com"],
        data["inertia"],
        data["gravity"],
    )


def _asset(tmp_path, data, *, reorder=False):
    import xml.etree.ElementTree as ET

    text = _reference().urdf_text(data)
    if reorder:
        root = ET.fromstring(text)
        root[:] = list(reversed(list(root)))
        text = ET.tostring(root, encoding="unicode")
    path = tmp_path / "frozen-tree.urdf"
    path.write_text(text, encoding="utf-8")
    return ph.load_urdf(path, gravity=data["gravity"])


def _checked_diagnostic(batch, key, data, entry, actual, h, effort, force, torque, *, folded=False):
    ref = _reference()
    expected = ref.oracle(data, entry.position, entry.velocity)
    diag = batch.tree_diagnostics([key])[key]
    assert diag is not None
    assert (diag.entry_tick, diag.exit_tick, diag.dt_s) == (entry.tick, actual.tick, h)
    _close(diag.entry_position, entry.position)
    _close(diag.entry_velocity, entry.velocity)
    _close(diag.mass, expected["mass"])
    _close(diag.bias, expected["bias"])
    load = ref.external(expected, effort, force, torque)
    _close(diag.external, load)
    _close(diag.drive, actual.applied_effort)
    _close(actual.generalized_applied, load + np.asarray(actual.applied_effort))
    _close(
        diag.acceleration,
        np.linalg.solve(
            expected["mass"], load + np.asarray(actual.applied_effort) - expected["bias"]
        ),
    )
    if not folded:
        _close(diag.linear_jacobian, expected["linear_jacobian"])
        _close(diag.angular_jacobian, expected["angular_jacobian"])
        velocity = np.asarray(entry.velocity)
        power = np.sum(force * np.einsum("bij,j->bi", diag.linear_jacobian, velocity))
        power += np.sum(torque * np.einsum("bij,j->bi", diag.angular_jacobian, velocity))
        _close(power, np.dot(load - effort, velocity))
    for name in (
        "entry_position",
        "entry_velocity",
        "mass",
        "bias",
        "linear_jacobian",
        "angular_jacobian",
    ):
        value = getattr(diag, name)
        assert not value.flags.writeable
    assert diag.equation_residual <= 2e-5
    assert np.isfinite(diag.control_residual)
    return diag, expected


def _check_end_fk(batch, key, data, actual, *, folded=False):
    truth = _reference().oracle(data, actual.position, actual.velocity)
    read = batch.tree_kinematics([key])[key]
    assert read.tick == actual.tick
    _close(read.position, actual.position)
    _close(read.velocity, actual.velocity)
    if not folded:
        assert read.joint_types == data["kinds"]
        _close(read.link_position_m, truth["position"])
        _close(read.link_rotation, truth["rotation"])
        _close(read.com_position_m, truth["com_position"])
    summary = batch.summary()[key]
    assert "q_m" not in summary and "q_rad" not in summary
    _close(
        summary["kinetic_energy_j"],
        0.5 * np.asarray(actual.velocity) @ truth["mass"] @ actual.velocity,
    )
    return read, truth


@pytest.mark.tyx
@pytest.mark.parametrize(
    "q,v,cycled",
    (
        ((0.37, pi / 2), (0.5, -0.25), False),
        ((0.37, -0.51), (0.5, -0.25), False),
        ((0.37, -0.51), (0.5, -0.25), True),
    ),
)
def test_t01_nonzero_coriolis_full_jacobians_and_common_transform(q, v, cycled, tyx_mode, tmp_path):
    ref = _reference()
    data, _ = ref.transformed(
        ref.model_data("T01"),
        cycled=cycled,
        translation=(0.5, -0.25, 0.75) if cycled else (0, 0, 0),
    )
    program, initial = _program(_ordinary_model(data), q=q, v=v)
    command = _action(program)
    with resident.TyxWorldBatch(
        [(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({7: command})
        entry = batch.snapshot()[7]
        mass, bias = ref.rr_terms(entry.position, entry.velocity)
        independent = ref.oracle(data, entry.position, entry.velocity)
        np.testing.assert_allclose(independent["mass"], mass, atol=1e-12)
        np.testing.assert_allclose(independent["bias"], bias, atol=1e-12)
        if q[1] == pi / 2:
            _close(bias, [3 / 32, 1 / 8])
            _close(np.linalg.solve(mass, -bias), [3 / 128, -51 / 128])
        assert batch.step(1 / 1024) == {7: True}
        actual = batch.snapshot()[7]
        _close(actual.velocity, np.asarray(entry.velocity) - np.linalg.solve(mass, bias) / 1024)
        _close(actual.position, np.asarray(entry.position) + np.asarray(actual.velocity) / 1024)
        diag, _ = _checked_diagnostic(
            batch, 7, data, entry, actual, 1 / 1024, np.zeros(2), np.zeros((2, 3)), np.zeros((2, 3))
        )
        read, _ = _check_end_fk(batch, 7, data, actual)
        _save(
            tmp_path,
            f"t01-pose-{cycled}-{tyx_mode}",
            {
                "input": data,
                "entry": entry,
                "actual": actual,
                "diagnostic": diag,
                "kinematics": read,
                "stats": batch.stats(),
            },
        )


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("T01", "T02", "T03", "T04", "N9", "N16"))
def test_t01_t04_t06_independent_forty_step_windows(name, tyx_mode, tmp_path):
    ref = _reference()
    data, q, v, effort, force, torque = ref.case_input(name)
    folded = name == "T03"
    asset = _asset(tmp_path, data) if folded else None
    model = asset.model if folded else _ordinary_model(data)
    program, initial = _program(model, q=q, v=v)
    # T03 has zero COM wrenches, so its three independent inertial pieces require
    # no ambiguous conversion to the two aggregate runtime COMs.
    command = _action(
        program, effort=effort, force=None if folded else force, torque=None if folded else torque
    )
    report = {
        "case": name,
        "input": data,
        "initial_q": q,
        "initial_v": v,
        "effort": effort,
        "force": force,
        "torque": torque,
        "h_s": 1 / 1024,
        "trace": [],
    }
    with resident.TyxWorldBatch(
        [(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({7: command})
        start = batch.snapshot()[7]
        oq, ov = np.asarray(start.position), np.asarray(start.velocity)
        previous = start
        try:
            for k in range(40):
                accepted = batch.step(1 / 1024)
                actual = batch.snapshot()[7]
                oq, ov, terms, load, acceleration = ref.reference_step(
                    data, oq, ov, effort, force, torque, 1 / 1024
                )
                diag = batch.tree_diagnostics([7])[7]
                read = batch.tree_kinematics([7])[7]
                report["trace"].append(
                    {
                        "accepted": accepted,
                        "entry": previous,
                        "actual": actual,
                        "diagnostic": diag,
                        "kinematics": read,
                        "oracle_q": oq.copy(),
                        "oracle_v": ov.copy(),
                        "oracle_mass": terms["mass"],
                        "oracle_bias": terms["bias"],
                        "oracle_load": load,
                        "oracle_acceleration": acceleration,
                        "stats": batch.stats(),
                    }
                )
                assert accepted == {7: True} and actual.tick == k + 1
                _close(actual.position, oq, atol=WINDOW_ATOL)
                _close(actual.velocity, ov, atol=WINDOW_ATOL)
                if name in ("T01", "T02"):
                    closed = ref.rr_terms if name == "T01" else ref.branch_terms
                    cm, cb = closed(previous.position, previous.velocity)
                    np.testing.assert_allclose(
                        ref.oracle(data, previous.position, previous.velocity)["mass"],
                        cm,
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        ref.oracle(data, previous.position, previous.velocity)["bias"],
                        cb,
                        atol=1e-12,
                    )
                _checked_diagnostic(
                    batch, 7, data, previous, actual, 1 / 1024, effort, force, torque, folded=folded
                )
                _check_end_fk(batch, 7, data, actual, folded=folded)
                previous = actual
            assert {item["stats"]["current_buffer"] for item in report["trace"]} == {0, 1}
        finally:
            report["final_stats"] = batch.stats()
            _save(tmp_path, f"t-window-{name}-{tyx_mode}", report)


@pytest.mark.tyx
@pytest.mark.parametrize("variant", ("zero", "moving", "permuted", "negative_axis"))
def test_t02_branch_undriven_axis_reaction_and_coordinate_controls(variant, tyx_mode, tmp_path):
    ref = _reference()
    data = ref.model_data("T02")
    q = np.zeros(3) if variant == "zero" else np.array([0.37, -1 / 16, 3 / 16])
    v = np.zeros(3) if variant == "zero" else np.array([0.5, 0.25, -0.125])
    effort = np.array([0.0, 1.0, 0.0])
    expected = ref.oracle(data, q, v)
    if variant == "zero":
        np.testing.assert_allclose(
            np.linalg.solve(expected["mass"], effort), [-2, 3, -2], atol=1e-12
        )
    else:
        np.testing.assert_allclose(expected["mass"][0, 0], 325 / 128, atol=1e-12)
        np.testing.assert_allclose(expected["bias"], [-5 / 128, 1 / 64, -3 / 64], atol=1e-12)
    if variant == "permuted":
        order = [0, 2, 1]
        for field in ("offsets", "orientations", "axes", "mass", "com", "inertia"):
            data[field] = data[field][order].copy()
        q, v, effort = q[order], v[order], effort[order]
        expected = {"mass": expected["mass"][np.ix_(order, order)], "bias": expected["bias"][order]}
    elif variant == "negative_axis":
        sign = np.array([1, -1, 1])
        data["axes"][1] *= -1
        q, v, effort = sign * q, sign * v, sign * effort
        expected = {
            "mass": sign[:, None] * expected["mass"] * sign[None, :],
            "bias": sign * expected["bias"],
        }
    independent = ref.oracle(data, q, v)
    np.testing.assert_allclose(independent["mass"], expected["mass"], atol=1e-12)
    np.testing.assert_allclose(independent["bias"], expected["bias"], atol=1e-12)
    program, initial = _program(_ordinary_model(data), q=q, v=v)
    with resident.TyxWorldBatch(
        [(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({7: _action(program, effort=effort)})
        entry = batch.snapshot()[7]
        assert batch.step(1 / 128) == {7: True}
        actual = batch.snapshot()[7]
        diag, truth = _checked_diagnostic(
            batch, 7, data, entry, actual, 1 / 128, effort, np.zeros((3, 3)), np.zeros((3, 3))
        )
        _close(
            actual.velocity,
            np.asarray(entry.velocity)
            + np.linalg.solve(truth["mass"], effort - truth["bias"]) / 128,
        )
        _save(
            tmp_path,
            f"t02-{variant}-{tyx_mode}",
            {"input": data, "entry": entry, "actual": actual, "diagnostic": diag},
        )


@pytest.mark.tyx
@pytest.mark.parametrize("pose", ((0.0, 0.0), (0.37, -0.51), (-0.51, 0.37)))
@pytest.mark.parametrize("reordered", (False, True))
def test_t03_fixed_tool_mass_tensor_and_source_order(pose, reordered, tyx_mode, tmp_path):
    ref = _reference()
    data = ref.model_data("T03")
    if reordered:
        data, _ = ref.transformed(data, cycled=True, translation=(0.5, -0.25, 0.75))
    asset = _asset(tmp_path, data, reorder=reordered)
    assert asset.model.dof_count == 2
    _close(asset.model.mass, [1, 1.5])
    # Check physical tensor independently in the authored link1 frame; no product FK.
    com = np.array([7 / 12, 1 / 12, 0])
    tensor = np.array([[3 / 32, -1 / 48, 0], [-1 / 48, 5 / 64, 0], [0, 0, 21 / 128]])
    parts = [
        (1.0, np.array([0.5, 0, 0]), np.diag([1 / 24, 1 / 24, 1 / 12])),
        (0.5, np.array([0.75, 0.25, 0]), np.diag([1 / 32, 1 / 64, 5 / 128])),
    ]
    aggregate = sum(
        (
            inertia
            + mass
            * (np.dot(point - com, point - com) * np.eye(3) - np.outer(point - com, point - com))
            for mass, point, inertia in parts
        ),
        np.zeros((3, 3)),
    )
    np.testing.assert_allclose(aggregate, tensor, atol=1e-15)
    if pose == (0.0, 0.0):
        plain = ref.model_data("T01")
        added = ref.oracle(data, pose, [0, 0])["mass"] - ref.oracle(plain, pose, [0, 0])["mass"]
        np.testing.assert_allclose(added, np.array([[205, 93], [93, 45]]) / 128, atol=1e-12)
    program, initial = _program(asset.model, q=pose, v=(0.25, -0.125))
    with resident.TyxWorldBatch(
        [(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({7: _action(program, effort=(1, 1 / 6))})
        entry = batch.snapshot()[7]
        assert batch.step(1 / 1024) == {7: True}
        actual = batch.snapshot()[7]
        diag, _ = _checked_diagnostic(
            batch,
            7,
            data,
            entry,
            actual,
            1 / 1024,
            np.array([1, 1 / 6]),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            folded=True,
        )
        _save(
            tmp_path,
            f"t03-{pose}-{reordered}-{tyx_mode}",
            {
                "authored": data,
                "model": asset.model,
                "entry": entry,
                "actual": actual,
                "diagnostic": diag,
            },
        )


@pytest.mark.tyx
@pytest.mark.parametrize("cycled,gravity_on", ((False, True), (True, True), (False, False)))
def test_t04_nonprincipal_axes_current_com_wrenches_gravity_virtual_power(
    cycled, gravity_on, tyx_mode, tmp_path
):
    ref = _reference()
    data, q, v, effort, force, torque = ref.case_input("T04")
    if not gravity_on:
        data["gravity"][:] = 0
    data, rotation = ref.transformed(
        data, cycled=cycled, translation=(0.5, -0.25, 0.75) if cycled else (0, 0, 0)
    )
    force, torque = force @ rotation.T, torque @ rotation.T
    program, initial = _program(_ordinary_model(data), q=q, v=v)
    with resident.TyxWorldBatch(
        [(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({7: _action(program, effort=effort, force=force, torque=torque)})
        entry = batch.snapshot()[7]
        assert batch.step(1 / 1024) == {7: True}
        actual = batch.snapshot()[7]
        diag, truth = _checked_diagnostic(
            batch, 7, data, entry, actual, 1 / 1024, effort, force, torque
        )
        _close(actual.velocity, np.asarray(entry.velocity) + diag.acceleration / 1024)
        _check_end_fk(batch, 7, data, actual)
        _save(
            tmp_path,
            f"t04-{cycled}-{gravity_on}-{tyx_mode}",
            {
                "input": data,
                "force": force,
                "torque": torque,
                "entry": entry,
                "actual": actual,
                "oracle": truth,
                "diagnostic": diag,
            },
        )


def _pd_gate(program, entry, actual, *, h, target, rate, ff):
    q, u, tau = (
        np.asarray(entry.position),
        np.asarray(actual.velocity),
        np.asarray(actual.applied_effort),
    )
    expected = np.clip(
        program.kp * (np.asarray(target) - q - h * u)
        + program.kd * (np.asarray(rate) - u)
        + np.asarray(ff),
        -program.effort_limit,
        program.effort_limit,
    )
    residual = np.abs(tau - expected)
    allowed = 5e-5 * np.maximum(1, np.maximum(np.abs(tau), np.abs(expected)))
    assert np.all(np.isfinite(expected)) and np.all(residual <= allowed)
    return float(np.max(residual))


@pytest.mark.tyx
def test_t05_coupled_partial_saturation_uses_original_effort_residual(tyx_mode, tmp_path):
    program, initial = _program(_rr_model(), kp=(4, 8), kd=(1, 2), cap=(0.25, 0.5))
    command = _action(program, ff=(1, -75 / 448))
    with resident.TyxWorldBatch(
        [(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({7: command})
        entry = batch.snapshot()[7]
        assert batch.step(1 / 16) == {7: True}
        actual = batch.snapshot()[7]
        _close(actual.applied_effort, [0.25, 0.0])
        _close(actual.velocity, [3 / 112, -15 / 224])
        _close(actual.position, np.array([3 / 112, -15 / 224]) / 16)
        residual = _pd_gate(
            program, entry, actual, h=1 / 16, target=[0, 0], rate=[0, 0], ff=[1, -75 / 448]
        )
        diag, _ = _checked_diagnostic(
            batch,
            7,
            _reference().model_data("T01"),
            entry,
            actual,
            1 / 16,
            np.zeros(2),
            np.zeros((2, 3)),
            np.zeros((2, 3)),
        )
        _close(diag.control_residual, residual)
        _save(
            tmp_path,
            f"t05-partial-{tyx_mode}",
            {"entry": entry, "actual": actual, "diagnostic": diag},
        )


@pytest.mark.tyx
@pytest.mark.parametrize("cancellation", (False, True))
def test_t05_wide_caps_do_not_hide_small_drive_or_large_term_cancellation(
    cancellation, tyx_mode, tmp_path
):
    kp = [1, 1] if cancellation else [0, 0]
    program, initial = _program(_rr_model(), kp=kp, cap=[2**20, 2**20])
    small = np.array([1 / 8, -1 / 16])
    target = np.array([2**15, -(2**15)], dtype=float) if cancellation else np.zeros(2)
    ff = -target + small
    effort = -small if cancellation else np.zeros(2)
    with resident.TyxWorldBatch(
        [(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({7: _action(program, target=target, ff=ff, effort=effort)})
        entry = batch.snapshot()[7]
        assert batch.step(1 / 16) == {7: True}
        actual = batch.snapshot()[7]
        _close(actual.applied_effort, small)
        expected_v = (
            np.zeros(2)
            if cancellation
            else np.linalg.solve(np.array([[8 / 3, 5 / 6], [5 / 6, 1 / 3]]), small) / 16
        )
        _close(actual.velocity, expected_v)
        _close(actual.generalized_applied, small + effort)
        _pd_gate(program, entry, actual, h=1 / 16, target=target, rate=[0, 0], ff=ff)
        _save(
            tmp_path,
            f"t05-widecap-{cancellation}-{tyx_mode}",
            {
                "entry": entry,
                "actual": actual,
                "target": target,
                "feedforward": ff,
                "caps": program.effort_limit,
                "diagnostic": batch.tree_diagnostics([7])[7],
            },
        )


@pytest.mark.tyx
def test_t05_delay_only_target_rate_feedforward_not_current_external(tyx_mode, tmp_path):
    data = _reference().model_data("T01")
    program, initial = _program(_rr_model(), delay=1)
    ff1, ff2 = np.array([1 / 8, -1 / 16]), np.array([-1 / 4, 1 / 8])
    force = np.zeros((2, 3))
    with resident.TyxWorldBatch(
        [(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        assert batch.tree_diagnostics([7]) == {7: None}
        batch.set_controls({7: _action(program, ff=ff1, effort=(1, 1 / 6))})
        entry = batch.snapshot()[7]
        assert batch.step(1 / 16) == {7: True}
        first = batch.snapshot()[7]
        _close(first.velocity, [1 / 16, -2 / 16])
        _close(first.applied_effort, [0, 0])
        _close(first.history, np.column_stack((np.zeros(2), np.zeros(2), ff1)))
        _checked_diagnostic(
            batch, 7, data, entry, first, 1 / 16, np.array([1, 1 / 6]), force, force
        )
        batch.set_controls({7: _action(program, ff=ff2, effort=(0, 1 / 4))})
        second_entry = batch.snapshot()[7]
        assert batch.step(1 / 16) == {7: True}
        second = batch.snapshot()[7]
        _close(second.applied_effort, ff1)
        _close(second.generalized_applied, ff1 + [0, 1 / 4])
        _close(second.history, np.column_stack((np.zeros(2), np.zeros(2), ff2)))
        _checked_diagnostic(
            batch, 7, data, second_entry, second, 1 / 16, np.array([0, 1 / 4]), force, force
        )
        assert np.asarray(second.control).shape == (2, 10)
        _close(np.asarray(second.control)[:, 2], ff2)
        _close(np.asarray(second.control)[:, 3], [0, 1 / 4])
        _save(
            tmp_path,
            f"t05-delay-{tyx_mode}",
            {
                "entry": entry,
                "first": first,
                "second": second,
                "diagnostic": batch.tree_diagnostics([7])[7],
            },
        )


@pytest.mark.tyx
def test_t05_iteration_budget_failure_keeps_whole_tree_and_neighbor_advances(tyx_mode, tmp_path):
    program, initial = _program(_rr_model(), kp=(4, 8), kd=(1, 2), cap=(10, 10), delay=0)
    worlds = [
        (7, _tree(program, initial, iterations=1)),
        (9, _tree(program, initial)),
        (23, resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0))),
    ]
    with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
        packet = _action(program, ff=(0.25, 0.25))
        batch.set_controls({7: packet, 9: packet, 23: (2, 0, 0)})
        entry = batch.snapshot()
        assert batch.step(1 / 16) == {7: False, 9: True, 23: True}
        actual = batch.snapshot()
        assert actual[7] == entry[7]
        assert actual[9].tick == actual[23].tick == 1
        assert batch.tree_diagnostics([7])[7] is None
        good = batch.tree_diagnostics([9])[9]
        assert good is not None and good.iterations > 1
        _pd_gate(
            program, entry[9], actual[9], h=1 / 16, target=[0, 0], rate=[0, 0], ff=[0.25, 0.25]
        )
        _save(
            tmp_path,
            f"t05-budget-{tyx_mode}",
            {
                "entry": entry,
                "actual": actual,
                "diagnostic": good,
                "stats": batch.stats(),
                "status": {7: False, 9: True, 23: True},
            },
        )


@pytest.mark.tyx
@pytest.mark.parametrize("kind", ("revolute", "prismatic"))
def test_t06_one_row_tree_has_actual_inertia_and_units(kind, tyx_mode, tmp_path):
    ref = _reference()
    data = ref._model(
        [-1],
        [kind],
        [[0.5, -0.25, 0.75]],
        [[0, 0, 1]],
        [2],
        [[0.25, 0, 0]],
        [[1 / 8, 1 / 8, 1 / 8]],
    )
    program, initial = _program(_ordinary_model(data))
    with resident.TyxWorldBatch(
        [(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({7: _action(program, effort=[1 / 4])})
        assert batch.step(1 / 128) == {7: True}
        actual = batch.snapshot()[7]
        acceleration = 1.0 if kind == "revolute" else 1 / 8
        _close(actual.velocity, [acceleration / 128])
        _close(actual.position, [acceleration / 128**2])
        read, _ = _check_end_fk(batch, 7, data, actual)
        assert read.joint_types == (kind,)
        _save(
            tmp_path,
            f"t06-one-{kind}-{tyx_mode}",
            {"input": data, "actual": actual, "kinematics": read},
        )


@pytest.mark.parametrize("problem", ("n17", "delay2", "stops", "iterations0", "iterations4097"))
def test_t06_capability_rejection_precedes_runtime_allocation(problem, tmp_path, monkeypatch):
    data = _reference().model_data("N17" if problem == "n17" else "T01")
    model = _ordinary_model(data)
    program, initial = _program(model, delay=2 if problem == "delay2" else 0)
    if problem == "stops":
        limits = ph.JointLimits(np.full(2, -1.0), np.ones(2), 2.0, 0.0)
        program = ph.DrivenRobotProgram(
            model, kp=np.zeros(2), kd=np.zeros(2), effort_limit=np.full(2, 8.0), limits=limits
        )
        initial = program.initial_state(ph.RobotState(np.zeros(2), np.zeros(2)))

    # An invalid public world must fail before allocating device buffers. A failure
    # from this sentinel would be an infrastructure failure, not the expected gate.
    def allocation_forbidden(*args, **kwargs):
        raise AssertionError("invalid tree reached runtime allocation")

    monkeypatch.setattr(resident.TyxWorldBatch, "_buffer", allocation_forbidden)
    iterations = 0 if problem == "iterations0" else 4097 if problem == "iterations4097" else 512
    with pytest.raises(
        (ValueError, TypeError), match="16|count|capacity|delay|stop|limit|iteration"
    ):
        with resident.TyxWorldBatch(
            [(7, _tree(program, initial, iterations=iterations))], "interpret", cache_root=tmp_path
        ):
            pass


@pytest.mark.tyx
@pytest.mark.parametrize("joint,sign", ((0, 1), (0, -1), (1, 1), (1, -1)))
def test_t06_stored_coordinate_domain_and_entire_tree_rejection(joint, sign, tyx_mode, tmp_path):
    data = _reference().model_data("T04")
    data["gravity"][:] = 0
    q, v = np.zeros(2), np.zeros(2)
    q[joint], v[joint] = sign * 4096.0, sign * 1.0
    program, initial = _program(_ordinary_model(data), q=q, v=v, delay=1)
    with resident.TyxWorldBatch(
        [(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({7: _action(program, ff=(0.25, -0.125))})
        entry = batch.snapshot()[7]
        before_fk = batch.tree_kinematics([7])[7]
        assert batch.step(1 / 128) == {7: False}
        assert batch.snapshot()[7] == entry and batch.tree_diagnostics([7])[7] is None
        _close(batch.tree_kinematics([7])[7].link_position_m, before_fk.link_position_m)
        outside = list(entry.position)
        outside[joint] = sign * float(np.nextafter(np.float32(4096), np.float32(np.inf)))
        with pytest.raises(ValueError, match="coordinate|angle|domain|4096"):
            batch.restore({7: replace(entry, position=tuple(outside))})
        assert batch.snapshot()[7] == entry
        batch.restore({7: replace(entry, position=(0.0, 0.0), velocity=(0.0, 0.0))})
        assert batch.step(1 / 128) == {7: True}
        _save(
            tmp_path,
            f"t06-domain-{joint}-{sign}-{tyx_mode}",
            {"rejected_entry": entry, "retried": batch.snapshot()[7], "stats": batch.stats()},
        )


@pytest.mark.tyx
def test_t06_mixed_restore_full_control_history_diagnostics_and_failure_retry(
    tyx_mode, tmp_path, monkeypatch
):
    program, initial = _program(_rr_model(), delay=1)
    data2, q2, v2, _, _, _ = _reference().case_input("T02")
    second, second_initial = _program(_ordinary_model(data2), q=q2, v=v2)
    worlds = [
        (7, _tree(program, initial)),
        (23, resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0))),
        (91, _tree(second, second_initial)),
    ]
    with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
        initial_all = batch.snapshot()
        assert batch.tree_diagnostics([91, 7]) == {91: None, 7: None}
        batch.set_controls(
            {
                7: _action(program, ff=(0.25, -0.125), effort=(1, 1 / 6)),
                23: (2, 0, 0),
                91: _action(second, effort=(0, 1, 0)),
            }
        )
        assert batch.step(1 / 1024) == {7: True, 23: True, 91: True}
        first = batch.snapshot()
        owned_diag = batch.tree_diagnostics([7])[7]
        owned_q = owned_diag.entry_position.copy()
        old_stats = batch.stats()
        with monkeypatch.context() as patch:

            def fail_status():
                raise RuntimeError("injected complete tree device attempt")

            patch.setattr(batch, "_read_status", fail_status)
            with pytest.raises(RuntimeError, match="complete tree"):
                batch.step(1 / 1024)
        assert batch.snapshot() == first
        assert batch.stats()["state_epoch"] == old_stats["state_epoch"]
        assert batch.tree_diagnostics([7, 91]) == {7: None, 91: None}
        np.testing.assert_array_equal(owned_diag.entry_position, owned_q)
        assert batch.step(1 / 1024) == {7: True, 23: True, 91: True}
        second_state = batch.snapshot()
        diagnostic_before_validation = _json(batch.tree_diagnostics([7])[7])
        with pytest.raises((KeyError, ValueError)):
            batch.set_controls({7: _action(program, ff=(7, 7)), 999: (0, 0, 0)})
        assert batch.snapshot() == second_state
        assert _json(batch.tree_diagnostics([7])[7]) == diagnostic_before_validation
        with pytest.raises(ValueError):
            batch.restore({7: replace(first[7], control=((0.0,) * 4,) * 2), 23: first[23]})
        assert batch.snapshot() == second_state
        # Full held n*10, delayed n*3, actual drive and generalized load roundtrip.
        batch.restore({7: first[7], 23: first[23]})
        assert batch.snapshot()[7] == first[7] and batch.snapshot()[23] == first[23]
        assert batch.snapshot()[91] == second_state[91]
        assert batch.tree_diagnostics([7])[7] is None
        batch.step(1 / 1024)
        replayed = batch.snapshot()
        assert replayed[7] == second_state[7] and replayed[23] == second_state[23]
        observer = batch.tree_kinematics([91, 7])
        assert tuple(observer) == (91, 7)
        with pytest.raises((ValueError, TypeError)):
            observer[7].position[0] = 99
        keep = batch.snapshot()[91]
        batch.reset([7])
        assert batch.snapshot()[7] == initial_all[7] and batch.snapshot()[91] == keep
        assert batch.tree_diagnostics([7])[7] is None
        _save(
            tmp_path,
            f"t06-transactions-{tyx_mode}",
            {
                "initial": initial_all,
                "first": first,
                "second": second_state,
                "replayed": replayed,
                "observer": observer,
                "final": batch.snapshot(),
                "stats": batch.stats(),
            },
        )


@pytest.mark.tyx
def test_t06_topology_identity_owned_inputs_and_scalar_reader_fail_closed(tyx_mode, tmp_path):
    program, initial = _program(_rr_model())
    foreign_model = replace(program.model, com=np.array([[0.25, 0, 0], [0.5, 0, 0]]))
    foreign, other_initial = _program(foreign_model)
    with resident.TyxWorldBatch(
        [(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path / "first"
    ) as batch:
        start = batch.snapshot()[7]
        with resident.TyxWorldBatch(
            [(7, _tree(foreign, other_initial))], tyx_mode, cache_root=tmp_path / "other"
        ) as other:
            assert start.topology != other.snapshot()[7].topology
            with pytest.raises(ValueError, match="topology|model"):
                batch.restore({7: other.snapshot()[7]})
        command = _action(program, effort=(1, 1 / 6))
        batch.set_controls({7: command})
        command.domains[program.domain].effort[:] = 77
        # Source arrays may be user-owned and writable; compiled data must be owned.
        program.model.mass.setflags(write=True)
        program.model.mass[:] = 999
        initial.domains[program.domain].q[:] = 77
        assert batch.step(1 / 128) == {7: True}
        _close(batch.snapshot()[7].velocity, [1 / 128, -2 / 128])
        with pytest.raises((ValueError, TypeError), match="tree|scalar|kind|supported"):
            batch.prepare_readback([7])
        old = batch.snapshot()[7]
        with pytest.raises(ValueError):
            batch.restore({7: replace(old, velocity=(float("nan"), 0.0))})
        assert batch.snapshot()[7] == old
        _save(tmp_path, f"t06-identity-{tyx_mode}", {"actual": old, "stats": batch.stats()})


@pytest.mark.tyx
def test_t05_original_partial_saturation_packet_is_delayed_one_physical_step(tyx_mode, tmp_path):
    program, initial = _program(_rr_model(), kp=(4, 8), kd=(1, 2), cap=(0.25, 0.5), delay=1)
    with resident.TyxWorldBatch(
        [(7, _tree(program, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({7: _action(program, ff=(1, -75 / 448))})
        assert batch.step(1 / 16) == {7: True}
        first = batch.snapshot()[7]
        np.testing.assert_array_equal(first.position, [0, 0])
        np.testing.assert_array_equal(first.velocity, [0, 0])
        assert batch.step(1 / 16) == {7: True}
        second = batch.snapshot()[7]
        _close(second.applied_effort, [0.25, 0])
        _close(second.velocity, [3 / 112, -15 / 224])
        _pd_gate(program, first, second, h=1 / 16, target=[0, 0], rate=[0, 0], ff=[1, -75 / 448])
        _save(
            tmp_path,
            f"t05-original-delay-{tyx_mode}",
            {"first": first, "second": second, "diagnostic": batch.tree_diagnostics([7])[7]},
        )


@pytest.mark.tyx
def test_t06_finite_huge_external_load_rejects_and_valid_retry_keeps_history(tyx_mode, tmp_path):
    program, initial = _program(_rr_model(), delay=1)
    with resident.TyxWorldBatch(
        [(7, _tree(program, initial)), (91, resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0)))],
        tyx_mode,
        cache_root=tmp_path,
    ) as batch:
        batch.set_controls(
            {7: _action(program, ff=(0.25, -0.125), effort=(1e30, 0)), 91: (2, 0, 0)}
        )
        entry = batch.snapshot()
        assert batch.step(1 / 1024) == {7: False, 91: True}
        assert batch.snapshot()[7] == entry[7]
        assert batch.tree_diagnostics([7])[7] is None
        rejected = batch.snapshot()
        # Bad HOST inputs cannot partially replace either world's held control.
        wrong_count = ph.SceneInputs({program.domain: ph.RobotInputs.zeros(1)})
        wrong_domain = ph.SceneInputs({"not_the_tree": _action(program).domains[program.domain]})
        nonfinite = _action(program)
        nonfinite.domains[program.domain].force[1, 2] = float("nan")
        for packet in (wrong_count, wrong_domain, nonfinite):
            with pytest.raises((ValueError, TypeError)):
                batch.set_controls({91: (8, 0, 0), 7: packet})
            assert batch.snapshot() == rejected
        batch.set_controls({7: _action(program, ff=(0.25, -0.125), effort=(1, 1 / 6))})
        assert batch.step(1 / 1024) == {7: True, 91: True}
        accepted = batch.snapshot()[7]
        _close(accepted.velocity, [1 / 1024, -2 / 1024])
        _close(accepted.applied_effort, [0, 0])
        _close(accepted.history, [[0, 0, 0.25], [0, 0, -0.125]])
        _save(
            tmp_path,
            f"t06-overflow-retry-{tyx_mode}",
            {"entry": entry, "rejected": rejected, "accepted": accepted, "stats": batch.stats()},
        )


@pytest.mark.parametrize("problem", ("parents", "axis", "inertia"))
def test_t06_existing_robot_model_validation_is_preserved(problem):
    data = _reference().model_data("T01")
    if problem == "parents":
        data["parents"][:] = [1, 0]
    elif problem == "axis":
        data["axes"][0] = [0, 0, 2]
    else:
        data["inertia"][0] = [1, 1, 3]
    with pytest.raises(ValueError):
        _ordinary_model(data)


def test_t06_real_mimic_asset_and_floating_state_are_explicitly_outside_tree_world(tmp_path):
    import xml.etree.ElementTree as ET

    authored = ET.fromstring(_reference().urdf_text(_reference().model_data("T01")))
    follower = next(j for j in authored.findall("joint") if j.get("name") == "joint1")
    ET.SubElement(follower, "mimic", joint="joint0", multiplier="-1", offset="0.125")
    path = tmp_path / "actual_mimic.urdf"
    path.write_text(ET.tostring(authored, encoding="unicode"), encoding="utf-8")
    mimic = ph.load_mimic_urdf(path, gravity=(0, 0, 0))
    assert mimic.tree.model.dof_count == 2 and mimic.matrix.shape == (2, 1)
    with pytest.raises((TypeError, ValueError), match="program|ordinary|DrivenRobot|mimic|fixed"):
        with resident.TyxWorldBatch(
            [(7, _tree(mimic, ph.RobotState(np.zeros(1), np.zeros(1))))],
            "interpret",
            cache_root=tmp_path / "mimic",
        ):
            pass
    program, _ = _program(_rr_model())
    floating = ph.FloatingRobotState(
        np.array([0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=float), np.zeros(8)
    )
    with pytest.raises((TypeError, ValueError), match="state|SceneState|fixed|floating"):
        with resident.TyxWorldBatch(
            [(7, _tree(program, floating))], "interpret", cache_root=tmp_path / "floating"
        ):
            pass
