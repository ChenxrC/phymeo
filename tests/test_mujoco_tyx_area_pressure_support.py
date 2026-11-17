"""Registered loaded area contact cases, with a scalar oracle independent of TYX."""

import json

import numpy as np
import pytest

from experiments.area_pressure.loaded_support import (
    DUAL,
    SINGLE,
    build_support,
    run_support,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
MASS, DAMPING, GRAVITY = 0.5, 40.0, 9.81
STIFFNESS = {"center": 9000.0, "left": 4500.0, "right": 9000.0}


def _height(q):
    return float(np.float32(np.float32(0.002) + np.float32(q)))


def _scalar_force(q, v, force, h, stiffness):
    effective_mass = MASS + h * DAMPING
    free_velocity = (MASS * v + h * (force - MASS * GRAVITY)) / effective_mass
    depth = -_height(q)
    total = (
        max(0.0, stiffness * (depth - h * free_velocity) / (1 + h * h * stiffness / effective_mass))
        if depth >= 0
        else 0.0
    )
    return total, free_velocity + h * total / effective_mass


def _check_trajectory(payload):
    h, rows = payload["h"], payload["records"]
    names = payload["patch_names"]
    stiffness = sum(STIFFNESS[name] for name in names)
    loads = []
    positive_residual = 0.0
    previous_tick = 0
    for row in rows:
        q, v = row["input_q"][0], row["input_v"][0]
        qo, vo = row["output_q"][0], row["output_v"][0]
        force = (0.0, -3.0, 9.0)[row["stage"]]
        assert row["input_tick"] == previous_tick
        previous_tick += 1
        assert row["output_tick"] == previous_tick
        assert row["applied_generalized_force"] == [force]
        expected, expected_v = _scalar_force(q, v, force, h, stiffness)
        actual = sum(sum(patch["normal_force"]) for patch in row["patches"])
        assert abs(actual - expected) <= 0.001 + 0.001 * abs(expected)
        assert abs(vo - expected_v) <= 3e-5
        assert abs(qo - (q + h * vo)) <= 2e-7
        assert abs(row["generalized_contact_force"][0] - actual) <= 2e-5
        assert abs(MASS * (vo - v) / h - (force - MASS * GRAVITY - DAMPING * vo + actual)) <= 0.002
        dynamics = row["dynamics"]
        assert dynamics["applied"] == [force]
        np.testing.assert_allclose(dynamics["mass_matrix"], [[MASS]], atol=2e-6)
        np.testing.assert_allclose(dynamics["bias"], [MASS * GRAVITY], atol=2e-6)
        np.testing.assert_allclose(dynamics["passive"], [-DAMPING * v], atol=2e-6)
        lhs = (MASS + h * DAMPING) * dynamics["smooth_acceleration"][0]
        assert abs(lhs - (force - DAMPING * v - MASS * GRAVITY)) <= 0.0001
        for name, patch in zip(names, row["patches"], strict=True):
            assert patch["pair"] == ["pressure", name]
            assert patch["stamp"]["tick"] == row["input_tick"]
            assert patch["stamp"]["time"] == row["input_time"]
            assert patch["stamp"]["configuration"]
            local_force = sum(patch["normal_force"])
            assert local_force == pytest.approx(
                actual * STIFFNESS[name] / stiffness, rel=2e-4, abs=2e-5
            )
            expected_area = STIFFNESS[name] / 2e7 if _height(q) <= 0 else 0.0
            assert sum(patch["dA"]) == pytest.approx(expected_area, rel=1e-9, abs=1e-12)
            if patch["normal_force"]:
                np.testing.assert_allclose(patch["normal_jacobian"], np.ones((3, 1)), atol=2e-6)
                np.testing.assert_allclose(
                    patch["normal_force"], local_force / 3, rtol=2e-4, atol=2e-5
                )
                assert {source[0] for source in patch["source_ids"]} == {"bottom"}
                np.testing.assert_array_equal(
                    np.array(patch["force_on_a"]) + patch["force_on_b"], 0
                )
                np.testing.assert_array_equal(
                    np.array(patch["torque_on_a_world_origin"]) + patch["torque_on_b_world_origin"],
                    0,
                )
                points = np.array(patch["points"])
                np.testing.assert_allclose(points[:, 2], 0, atol=1e-10)
                np.testing.assert_allclose(
                    patch["torque_on_a_world_origin"],
                    np.cross(points, patch["force_on_a"]),
                    rtol=2e-5,
                    atol=1e-6,
                )
        height, next_height = _height(q), _height(qo)
        assert max(-height, -next_height) < 0.004
        old_e = (
            0.5 * MASS * v * v + 0.5 * stiffness * max(-height, 0) ** 2 + MASS * GRAVITY * height
        )
        new_e = (
            0.5 * MASS * vo * vo
            + 0.5 * stiffness * max(-next_height, 0) ** 2
            + MASS * GRAVITY * next_height
        )
        residual = new_e - old_e - force * (qo - q) + h * DAMPING * vo * vo
        positive_residual += max(residual, 0)
        loads.append(actual)
    initial_energy = MASS * GRAVITY * 0.002
    assert positive_residual <= 0.02 * initial_energy
    assert any(load > 0.1 for load in loads)
    assert all(load == 0 for row, load in zip(rows, loads, strict=True) if row["input_time"] >= 0.5)
    assert all(_height(row["output_q"][0]) > 0.001 for row in rows if row["input_time"] >= 0.5)
    plateaus = []
    for start, end, target in ((0.12, 0.15, 4.905), (0.32, 0.35, 7.905)):
        selected = [
            (r, f) for r, f in zip(rows, loads, strict=True) if start <= r["input_tick"] * h < end
        ]
        mean_force = float(np.mean([f for _, f in selected]))
        mean_depth = float(np.mean([-_height(r["output_q"][0]) for r, _ in selected]))
        assert mean_force == pytest.approx(target, rel=0.03)
        assert mean_depth == pytest.approx(target / stiffness, rel=0.03)
        assert max(abs(f / target - 1) for _, f in selected) <= 0.05
        assert max(abs(r["output_v"][0]) for r, _ in selected) <= 0.002
        plateaus.append(dict(mean_force=mean_force, mean_depth=mean_depth, target=target))
    return dict(
        plateaus=plateaus,
        peak_force=max(loads),
        max_depth=max(-_height(r["output_q"][0]) for r in rows),
        positive_energy_residual_fraction=positive_residual / initial_energy,
        final_height=_height(rows[-1]["output_q"][0]),
        final_velocity=rows[-1]["output_v"][0],
    )


@pytest.mark.parametrize("mode", MODES)
def test_current_area_support_tracks_sustained_load_and_releases(mode, tmp_path):
    payload = run_support(mode, 0.001, SINGLE)
    path = tmp_path / f"single-{mode}-0.001.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    summary = _check_trajectory(payload)
    (tmp_path / "checked-summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.mark.parametrize("mode", MODES)
def test_two_areas_share_one_motion_and_distribute_the_sustained_load(mode, tmp_path):
    summaries = []
    for h in (0.001, 0.0005):
        payload = run_support(mode, h, DUAL)
        path = tmp_path / f"dual-{mode}-{h}.json"
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        summaries.append(_check_trajectory(payload))
    for coarse, fine in zip(summaries[0]["plateaus"], summaries[1]["plateaus"], strict=True):
        assert coarse["mean_force"] == pytest.approx(fine["mean_force"], rel=0.03)
        assert coarse["mean_depth"] == pytest.approx(fine["mean_depth"], rel=0.03)
    (tmp_path / "checked-summary.json").write_text(json.dumps(summaries), encoding="utf-8")


@pytest.mark.parametrize("mode", MODES)
def test_shared_response_rejects_independent_patch_sum_and_is_order_invariant(mode):
    accepted = []
    for patches in (DUAL, tuple(reversed(DUAL))):
        runtime, scene = build_support(mode, 0.001, patches, initial_q=-0.003, initial_v=-0.02)
        with runtime:
            before = runtime.integration_snapshot()[13]
            scene.step(0.001, expected_revision=runtime.checkpoint_revision)
            after = runtime.integration_snapshot()[13]
            actual = {
                p["identity"].rigid_body: float(np.sum(p["normal_force"]))
                for p in after.last_step.raw_contact_fields["area_pressure"]
            }
            expected, velocity = _scalar_force(before.q[0], before.v[0], 0, 0.001, 13500)
            assert sum(actual.values()) == pytest.approx(expected, rel=0.001, abs=0.001)
            assert after.v[0] == pytest.approx(velocity, abs=3e-5)
            assert actual["right"] == pytest.approx(2 * actual["left"], rel=2e-4)
            independent = sum(
                _scalar_force(before.q[0], before.v[0], 0, 0.001, k)[0] for k in (4500, 9000)
            )
            assert independent / expected == pytest.approx(7462 / 7381, rel=1e-10)
            assert independent - sum(actual.values()) > 0.005 * expected
            accepted.append((actual, after.q.copy(), after.v.copy()))
    for name in ("left", "right"):
        assert accepted[0][0][name] == pytest.approx(accepted[1][0][name], rel=2e-4)
    np.testing.assert_allclose(accepted[0][1], accepted[1][1], rtol=0, atol=1e-7)
    np.testing.assert_allclose(accepted[0][2], accepted[1][2], rtol=0, atol=3e-5)


@pytest.mark.parametrize("mode", MODES)
def test_second_patch_exceeding_total_capacity_keeps_state_and_held_load(mode):
    runtime, scene = build_support(mode, 0.001, DUAL, initial_q=-0.003, capacity=3)
    with runtime:
        runtime.set_controls({13: MuJoCoControl([], qfrc_applied=[-3])})
        before = runtime.integration_snapshot()[13]
        revision = runtime.checkpoint_revision
        with pytest.raises(ValueError, match="capacity"):
            scene.step(0.001, expected_revision=revision)
        after = runtime.integration_snapshot()[13]
        np.testing.assert_array_equal(after.q, before.q)
        np.testing.assert_array_equal(after.v, before.v)
        assert after.time == before.time and after.tick == before.tick
        assert after.last_step is before.last_step
        assert runtime.checkpoint_revision == revision
        assert not scene.last_geometry
        np.testing.assert_array_equal(after.held_control.qfrc_applied, [-3])
