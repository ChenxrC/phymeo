"""Freeze an actual uniform ball, curved witnesses and a full controlled grasp/slip pair."""

import json

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.experiments.panda_cube import run_panda_cube

BALL = {"kind": "sphere", "radius_m": 0.025}
MASS = 0.080


def test_oblique_ball_sample_and_lateral_spinning_table_contact():
    from phymeo.experiments.panda_payload import SampledPayload

    shape = SampledPayload(payload_geometry=BALL)
    law = RobotContactMaterial(200, 0, 0.5, 0.04)
    # 3-4-5 radial normal; the second point lies in the enclosing cube but outside the ball.
    points = np.array([[0.012, 0.016, 0], [0.022, 0.022, 0]])
    w = np.array([0.0, 0, 2])
    velocity = np.cross(w, points) + [0, 0, 0.02]
    load = shape.pad_loads(points, velocity, np.zeros(3), np.eye(3), np.zeros(3), w, law)
    np.testing.assert_allclose(load.depths, [0.005, 0], atol=1e-14)
    np.testing.assert_allclose(load.forces, [[0.6, 0.8, -0.25], [0, 0, 0]], atol=1e-14)
    np.testing.assert_allclose(load.body_force, [-0.6, -0.8, 0.25], atol=1e-14)
    np.testing.assert_allclose(load.body_torque, [0.004, -0.003, 0], atol=1e-14)
    with pytest.raises(ValueError, match="center"):
        shape.pad_loads(
            np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(3), np.eye(3), np.zeros(3), w, law
        )
    # The instantaneous sphere witness remains at the world bottom with a tilted material frame.
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    table_center, half = np.array([0, 0, -0.05]), np.array([0.1, 0.1, 0.05])
    floor_law = RobotContactMaterial(4000, 0, 0.5, 0.04)
    forces, normal, arms = shape.table_loads(
        np.array([0.0, 0, 0.024]),
        rotation,
        np.array([0.08, 0, 0]),
        np.array([0.0, 2, 0]),
        table_center,
        half,
        floor_law,
    )
    np.testing.assert_allclose(arms, [[0, 0, -0.025]], atol=1e-15)
    np.testing.assert_allclose(normal, [4], atol=1e-13)
    np.testing.assert_allclose(forces, [[-1.5, 0, 4]], atol=1e-13)
    np.testing.assert_allclose(np.cross(arms, forces), [[0, 0.0375, 0]], atol=1e-14)
    # The finite tabletop edge has a 3-4-5 oblique normal, then no force beyond the sphere.
    for x, expected in ((0.112, [12, 0, 16]), (0.13, [0, 0, 0])):
        forces, _, _ = shape.table_loads(
            np.array([x, 0, 0.016]),
            np.eye(3),
            np.zeros(3),
            np.zeros(3),
            table_center,
            half,
            floor_law,
        )
        np.testing.assert_allclose(forces[0], expected, atol=1e-12)
    np.testing.assert_allclose(shape.inertia(MASS), [2e-5] * 3, atol=1e-16)


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "sphere", "radius_m": 0},
        {"kind": "sphere", "radius_m": float("nan")},
        {"kind": "sphere", "radius_m": 0.025, "half_height": 0.1},
        {"kind": "cylinder", "radius_m": 0.025},
    ],
)
def test_invalid_shape_fails_before_advance(bad):
    with pytest.raises(ValueError):
        run_panda_cube(payload_geometry=bad)


def _sphere_window(
    label, mu, radius, mass, height, expected_inertia, tmp_path, *, structural_response=None
):
    ball = {"kind": "sphere", "radius_m": radius}
    result = run_panda_cube(
        payload_geometry=ball,
        structural_response=structural_response,
        mass_kg=mass,
        grip_above_com_m=height,
        closed_width_m=0.024,
        friction=mu,
        record=True,
    )
    spec, trace = result["inputs"], result["trace"]
    (tmp_path / f"sphere-{label}.json").write_text(json.dumps(result), encoding="utf8")
    (tmp_path / f"sphere-{label}.inputs.json").write_text(json.dumps(spec), encoding="utf8")
    assert result["payload_geometry"] == spec["payload_geometry"] == ball
    assert result["steps"] == len(trace) == 3600 and trace[-1]["t"] == 2.25
    np.testing.assert_allclose(
        spec["inertia_diagonal_kg_m2"], [expected_inertia] * 3, atol=1e-18, rtol=0
    )
    assert result["cube_mass_kg"] == mass and spec["full_size_m"] == [2 * radius] * 3
    initial = np.array(spec["body_position_m"])
    top = spec["table_center_m"][2] + spec["table_full_size_m"][2] / 2
    assert initial[2] - radius == pytest.approx(top)
    moving = trace[2159]
    assert moving["t"] == 1.35
    assert result["carry_clearance_m"] == pytest.approx(moving["cube"][2] - radius - top)
    target = initial + [0.075, 0, 0]
    assert result["final_target_error_m"] == pytest.approx(
        np.linalg.norm(trace[-1]["cube"] - target)
    )
    assert (
        result["max_cube_momentum_residual_Ns"] < 1e-12
        and result["max_joint_limit_violation"] < 1e-4
    )
    assert result["peak_bilateral_normal_N"] > 1
    return result


def _registered_goal(result, carry):
    weight = 9.81 * result["cube_mass_kg"]
    checks = {
        "object_clearance": result["other_robot_cube_axis_clearance_m"] > 0,
        "table_clearance": result["robot_table_axis_clearance_m"] > 0,
        "joint_limits": result["max_joint_limit_violation"] < 1e-4,
        "bilateral_contact": result["peak_bilateral_normal_N"] > 1,
        "released_pads": result["release_pad_force_N"] < 0.001,
        "final_weight_on_table": abs(result["final_table_force_N"] - weight) < 0.025,
    }
    if carry:
        checks.update(
            height=result["carry_clearance_m"] > 0.035,
            travel=result["carry_displacement_m"] > 0.045,
            bilateral_during_transfer=result["min_carry_bilateral_normal_N"] > 0,
            unloaded_table=result["peak_carry_table_N"] < 0.001,
            final_target=result["final_target_error_m"] < 0.012,
            reported_result=result["success"]
            and result["carried_during_transfer"]
            and result["released_on_target"],
        )
    else:
        checks.update(
            height=result["carry_clearance_m"] < 0.005,
            travel=abs(result["carry_displacement_m"]) < 0.015,
            no_sustained_grip=result["min_carry_bilateral_normal_N"] == 0,
            supported_during_transfer=result["peak_carry_table_N"]
            > 0.65 * result["cube_mass_kg"] / 0.08,
            reported_result=not (
                result["success"]
                or result["carried_during_transfer"]
                or result["released_on_target"]
            ),
        )
    return [key for key, ok in checks.items() if not ok]


def _observe_window(result, historical_issue=None):
    """Separate the frozen application goal from the independently observable outcome."""
    radius = result["payload_geometry"]["radius_m"]
    mu, trace = result["finger_friction"], result["trace"]
    failures = _registered_goal(result, carry=mu > 0)
    issue = result.get("first_unresolved_structure", historical_issue)
    if radius == 0.025:
        assert not result["carried_during_transfer"] and failures
        if mu > 0:
            top = result["table_center_m"][2] + result["table_full_size_m"][2] / 2
            assert any(
                r["cube"][2] - top - radius > 0.02 and min(r["pad_normal_N"]) > 1
                for r in trace
                if 0.65 <= r["t"] <= 0.80
            )
            assert (
                result["min_carry_bilateral_normal_N"] == 0
                and abs(result["final_table_force_N"] - 0.7848) < 0.025
            )
            assert issue is None and result["other_robot_cube_axis_clearance_m"] > 0
        else:
            # This loss precedes both the table edge and the first unforced arm structure.
            escaping = trace[959]
            assert escaping["t"] == pytest.approx(0.60) and min(escaping["pad_normal_N"]) > 1
            assert escaping["velocity"][0] < -0.4 and escaping["table_N"] > 0
            assert issue["pair"] == "robot_payload" and issue["t_before_s"] > escaping["t"]
            assert issue["clearance_m"] < 0 and result["final_table_force_N"] == 0
    else:
        assert radius == 0.020 and not _registered_goal(result, carry=True)
        if mu == 0:
            assert failures and result["carried_during_transfer"]
            # Use the actual pre-step COM and the recorded eight forces on the fingers.
            row = trace[2159]
            arms = np.array(row["pad_points_before_m"]) - trace[2158]["cube"]
            force = -np.array(row["pad_forces_N"])
            normals = arms / np.linalg.norm(arms, axis=1)[:, None]
            tangent = force - np.einsum("ij,ij->i", force, normals)[:, None] * normals
            assert np.max(np.abs(tangent)) < 1e-7
            assert np.any(force[:, 2] > 0.1) and np.any(force[:, 2] < -0.1)
            assert force[:, 2].sum() > 0.35 and row["table_N"] == 0 and issue is None
    return {
        "actual_carried": result["carried_during_transfer"],
        "actual_placed": result["released_on_target"],
        "registered_goal": "carry_and_place" if mu > 0 else "stay_on_table",
        "registered_goal_met": not failures,
        "failed_registered_criteria": failures,
        "first_unforced_structure": issue,
    }


@pytest.mark.parametrize("label,mu", [("grip", 0.65), ("slip", 0.0)])
def test_50mm_ball_grasp_loses_load_before_transfer(label, mu, tmp_path):
    _observe_window(
        _sphere_window(label, mu, 0.025, 0.080, 0.008, 2e-5, tmp_path, structural_response=False)
    )


@pytest.mark.parametrize("label,mu", [("grip", 0.65), ("slip", 0.0)])
def test_equatorial_40mm_ball_transport_and_radial_support(label, mu, tmp_path):
    # 3.65 mm nominal palm clearance, same density, and same 76 -> 24 mm jaw command.
    _observe_window(_sphere_window(label, mu, 0.020, 0.04096, 0.0, 6.5536e-6, tmp_path))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Audit original sphere goals and actual saved Newton/Phymeo states."
    )
    parser.add_argument("--first-structure", type=Path, required=True)
    parser.add_argument("--motion", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    historic = json.loads(args.first_structure.read_text(encoding="utf8"))
    records = []
    for path in args.motion:
        data = json.loads(path.read_text(encoding="utf8"))
        issue = historic.get(path.name, {}).get("first_unresolved_structure")
        records.append(dict(motion=str(path), **_observe_window(data, issue)))
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(records, stream, allow_nan=False, indent=2)
    print(json.dumps(records, allow_nan=False))
