"""Independent finite-gripper oracles; no production state is prescribed by them."""

import json
import os
import uuid
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

from phymeo.core.scene import fingerprint
from phymeo.experiments.parallel_jaw_cloth import (
    GripperClothRun,
    build_gripper_case,
    run_gripper_case,
    save_gripper_run,
)
from phymeo.runtime.program import SceneStepReport, SceneStepResult, SceneSubstepReport


def observe(case, mode, h, duration, **configuration):
    run = run_gripper_case(case, mode=mode, h=h, duration_s=duration, **configuration)
    root = os.environ.get("PHY_MEO_G2_EVIDENCE_DIR")
    if root:
        save_gripper_run(
            run,
            Path(root) / f"{case}-{mode}-{uuid.uuid4().hex}",
            source={
                "snapshot": os.environ.get("P65_SNAPSHOT", "unspecified"),
                "collection": "same-run full records; external manifest binds source",
            },
        )
    return run


def independent_oracle(case, h, step):
    h, t = F(h), step * F(h)
    answer = dict(
        z=F(0),
        vz=F(0),
        qz=F(0),
        rvz=F(0),
        qc=F(0),
        vc=F(0),
        p=2 * h / 3,
        eta=F(0),
        tz=F(0),
        tauz=F(0),
        dx=F(0),
        vx=F(0),
    )
    if case == "hold":
        answer["tz"] = h / 2
    elif case == "slip":
        answer.update(
            z=-(t * t + t * h) / 3, vz=-2 * t / 3, qz=(t * t + t * h) / 6, rvz=t / 3, tz=h / 6
        )
    elif case == "capture":
        delta = F(1, 16)
        contact_step = 1
        while h * h * contact_step * (contact_step + 1) / 2 < delta:
            contact_step += 1
        event = contact_step * h

        def d(s):
            return s * (s + h) / 2

        if step < contact_step:
            answer.update(z=-d(t), vz=-t, qz=d(t), rvz=t, qc=d(t), vc=t, p=F(0))
        else:
            elapsed = t - event
            answer.update(
                z=-d(event) + event * elapsed / 3 + d(elapsed) / 3,
                vz=t / 3,
                qz=d(event) + event * elapsed / 3 + d(elapsed) / 3,
                rvz=t / 3,
                qc=delta,
                tz=2 * h / 3,
            )
            if step == contact_step:
                u = (delta - d(event - h)) / h
                answer.update(p=2 * (event - u) / 3, eta=2 * u / 3, tz=F(0), tauz=2 * event / 3)
    elif case == "material":
        multiplier = -F(1, 64) / (2 + F(1, 16) / (h * h))
        answer.update(dx=-multiplier, vx=-multiplier / h)
    else:
        raise AssertionError("unknown oracle case")
    return answer


def assert_run_physics(run):
    assert run.accepted, [r.report.reason for r in run.records]
    tol = run.settings.tolerance
    scale = run.parameters["mass_scale"]
    case = run.parameters["case"]
    origin = run.initial.domains["nodes"].position
    shift = run.parameters["vertical_shift_m"]
    previous = run.initial
    cumulative_work = 0.0
    for index, result in enumerate(run.records, 1):
        expected = independent_oracle(case, run.h, index)
        nodes = result.state.domains["nodes"]
        robot = result.state.domains["robot"]
        x = origin + np.array([0.0, 0.0, float(expected["z"])])
        x = x.copy()
        x[0, 0] += float(expected["dx"])
        x[2, 0] -= float(expected["dx"])
        v = np.tile([0.0, 0.0, float(expected["vz"])], (3, 1))
        v[0, 0], v[2, 0] = float(expected["vx"]), -float(expected["vx"])
        np.testing.assert_allclose(nodes.position, x, rtol=0, atol=tol)
        np.testing.assert_allclose(nodes.velocity, v, rtol=0, atol=tol)
        np.testing.assert_allclose(
            robot.q,
            [shift + float(expected["qz"]), float(expected["qc"]), float(expected["qc"])],
            rtol=0,
            atol=tol,
        )
        np.testing.assert_allclose(
            robot.v,
            [float(expected["rvz"]), float(expected["vc"]), float(expected["vc"])],
            rtol=0,
            atol=tol,
        )
        assert len(result.substep_states) == 1
        sub = result.report.substeps[0]
        data = sub.diagnostics["dynamic_robot_thick_contact"]
        p, e = data["held_impulses"], data["terminal_impulses"]
        if expected["p"] or expected["eta"]:
            assert p.shape == e.shape == (6,)
        np.testing.assert_allclose(p, float(expected["p"]) * scale, rtol=0, atol=tol)
        np.testing.assert_allclose(e, float(expected["eta"]) * scale, rtol=0, atol=tol)
        for key, oracle_key in (("held_tangent", "tz"), ("terminal_tangent", "tauz")):
            target = np.tile([0.0, 0.0, float(expected[oracle_key]) * scale], (len(p), 1))
            np.testing.assert_allclose(data[key], target, rtol=0, atol=tol)
        assert data["terminal_ledger"]["applications"] == 1
        assert all(c["status"] == "certified" for c in data["ccd"])
        assert len(data["ccd"]) == 2
        for c in data["ccd"]:
            segments = c["segments"]
            assert F(segments[0][0]) == 0 and F(segments[-1][1]) == 1
            assert all(F(a[1]) == F(b[0]) for a, b in zip(segments, segments[1:]))
        physical = data["physical_budgets"]
        assert (
            physical["frozen_model_energy"]["positive_gain_upper_j"]
            <= run.parameters["max_energy_gain_j"]
        )
        assert (
            physical["material_angular_defect"]["l1_sum_nms"]
            <= run.parameters["max_material_angular_defect_nms"]
        )
        # Independent net force balance, including the massive shared carriage.
        momentum_z = scale * np.sum(nodes.velocity[:, 2]) + 6 * scale * robot.v[0]
        force_z = 0.0 if case == "material" else run.parameters["lift_effort_N"] - 9 * scale
        np.testing.assert_allclose(momentum_z, force_z * index * run.h, rtol=0, atol=tol)
        kinetic = (
            scale * np.sum(nodes.velocity**2) / 2
            + 3 * scale * robot.v[0] ** 2
            + scale * np.sum(robot.v[1:] ** 2)
        )
        dq = robot.q - previous.domains["robot"].q
        cumulative_work += float(run.inputs.domains["robot"].effort @ dq)
        if case == "material":
            extension = np.linalg.norm(nodes.position[2] - nodes.position[0]) - (2 - 1 / 64)
            delta_energy = (
                kinetic + extension**2 / (2 / (16 * scale)) - (1 / 64) ** 2 / (2 / (16 * scale))
            )
            exact = -(F(1, 64) ** 2) / (F(1, 16) * (2 + F(1, 16) / F(run.h) ** 2))
            np.testing.assert_allclose(delta_energy, float(exact) * scale, rtol=0, atol=tol)
            assert data["material_reports"]["nodes"]["max_residual"] <= tol
        else:
            potential = scale * np.sum(nodes.position[:, 2] - origin[:, 2]) + 6 * scale * (
                robot.q[0] - shift
            )
            # Original input work is independently integrated from returned q.
            exact_work = run.parameters["lift_effort_N"] * float(
                expected["qz"]
            ) + 4 * scale * float(expected["qc"])
            np.testing.assert_allclose(cumulative_work, exact_work, rtol=0, atol=tol)
            assert kinetic + potential - cumulative_work <= tol
            if case == "slip":
                np.testing.assert_allclose(kinetic, (index * run.h) ** 2 * scale, rtol=0, atol=tol)
        if run.parameters["mode"] != "numpy":
            assert result.report.backend == "tyx:" + run.parameters["mode"]
            assert sub.diagnostics["execution"]["completed_launches"] > 0
            assert data["terminal_ledger"]["ledger"]["compute_precision"] == "float32"
            for value in (nodes.position, nodes.velocity, robot.q, robot.v):
                assert np.array_equal(value, value.astype(np.float32).astype(np.float64))
        previous = result.state


def assert_rebuild_equal(first, second):
    assert fingerprint((first.initial, first.inputs)) == fingerprint(
        (second.initial, second.inputs)
    )
    assert len(first.records) == len(second.records)
    for left, right in zip(first.records, second.records, strict=True):
        assert fingerprint((left.state, left.substep_states)) == fingerprint(
            (right.state, right.substep_states)
        )


@pytest.mark.parametrize("case", ["hold", "slip", "capture"])
def test_host_full_windows_halving_and_complete_rebuild(case):
    runs = []
    for h in (1 / 8, 1 / 16):
        first, second = (observe(case, "numpy", h, 1.0) for _ in range(2))
        assert_run_physics(first)
        assert_run_physics(second)
        assert_rebuild_equal(first, second)
        runs.append(first)
    if case == "slip":
        errors = [
            abs(
                r.final_state.domains["nodes"].position[0, 2]
                - r.initial.domains["nodes"].position[0, 2]
                + 1 / 3
            )
            for r in runs
        ]
        assert abs(errors[0] - 2 * errors[1]) <= runs[0].settings.tolerance
    if case == "capture":
        for run in runs:
            dz = (
                run.final_state.domains["nodes"].position[0, 2]
                - run.initial.domains["nodes"].position[0, 2]
            )
            assert dz > 0
            continuous = F(1, 6) - F(4, 3) * F(1, 16)
            bound = F(run.h) * (F(1, 6) + F(4, 3) * F(3, 8))
            assert abs(dz - float(continuous)) <= float(bound)


@pytest.mark.parametrize("h", [1 / 4, 1 / 8])
def test_host_material_active_one_step_and_rebuild(h):
    first, second = (observe("material", "numpy", h, h) for _ in range(2))
    assert_run_physics(first)
    assert_run_physics(second)
    assert_rebuild_equal(first, second)


@pytest.mark.parametrize("case", ["hold", "slip"])
@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_mass_effort_and_finite_geometry_variations(case, scale):
    for reverse in (False, True):
        run = observe(
            case,
            "numpy",
            1 / 8,
            1 / 4,
            mass_scale=scale,
            patch_half_extent_m=(3.0, 5.0),
            vertical_shift_m=3 / 8,
            reverse_declarations=reverse,
        )
        assert_run_physics(run)


@pytest.mark.parametrize(
    "config",
    [
        dict(h=0.0),
        dict(h=True),
        dict(h=float("nan")),
        dict(mass_scale=0.0),
        dict(reverse_declarations=1),
        dict(vertical_shift_m=float("inf")),
        dict(mode="fallback"),
        dict(case="unknown"),
    ],
)
def test_invalid_configuration(config):
    options = dict(case="hold", mode="numpy", h=1 / 8)
    options.update(config)
    with pytest.raises((ValueError, TypeError)):
        build_gripper_case(**options)


def test_duration_and_material_control_boundaries():
    with pytest.raises(ValueError, match="integral"):
        run_gripper_case("hold", mode="numpy", h=1 / 8, duration_s=0.2)
    with pytest.raises(ValueError, match="one step"):
        run_gripper_case("material", mode="numpy", h=1 / 8, duration_s=1 / 4)


def test_configuration_owns_original_inputs():
    case = build_gripper_case("hold", mode="numpy", h=1 / 8)
    try:
        for value in (
            case.initial.domains["nodes"].position,
            case.inputs.domains["robot"].effort,
            case.parameters["closing_effort_N"],
        ):
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.flat[0] = 42
    finally:
        case.program.close()


def test_lossless_artifact_keeps_raw_timing_and_refuses_overwrite(tmp_path):
    # Serialization-only synthetic fixture: not a physical acceptance sample.
    case = build_gripper_case("hold", mode="numpy", h=1 / 8)
    try:
        signature = case.program.problem_signature()
        record = SceneStepResult(
            case.initial,
            SceneStepReport(
                True, "numpy", (SceneSubstepReport(True, 0, 1 / 8, timings={"synthetic": 0.125}),)
            ),
            (case.initial,),
        )
        run = GripperClothRun(
            case.scene,
            case.initial,
            case.inputs,
            case.settings,
            case.parameters,
            signature,
            1 / 8,
            1,
            (record,),
        )
        target = tmp_path / "record"
        save_gripper_run(run, target, source={"purpose": "synthetic-format-control"})
        payload = json.loads((target / "run.json").read_text())
        assert payload["schema"] == "phymeo-finite-gripper-run-v1"
        typed = payload["run"]["fields"]["records"][0]["fields"]
        timing = typed["report"]["fields"]["substeps"][0]["fields"]["timings"]
        assert timing["synthetic"]["hex"] == (0.125).hex()
        with pytest.raises(FileExistsError):
            save_gripper_run(run, target, source={})
    finally:
        case.program.close()
