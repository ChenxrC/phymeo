"""Branch and nested transmission reactions checked against an unreduced full-coordinate solve."""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from reference_newton.robot.mimic_graph_oracle import H, fan, kkt, serial

from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.rigid import RigidBodyState
from phymeo.core.robot import RobotInputs, RobotState
from phymeo.dynamics.robot_dynamics import robot_dynamics_terms
from phymeo.dynamics.robot_kinematics import point_jacobians
from phymeo.runtime.robot_payload_scene import RobotPayloadScene

ROOT = Path(__file__).parent / "reference_newton" / "robot"


@pytest.mark.parametrize(
    "case,ref", [("fan", fan()), ("chain", serial()), ("chain", serial(-0.16, -0.24))]
)
def test_each_original_transmission_row_matches_full_kkt_and_total_body_energy(case, ref):
    asset = load_mimic_urdf(ROOT / f"mimic_{case}.urdf", gravity=(0, 0, 0))
    assert asset.tree.joint_names == ref["names"]
    ids = list(asset.independent_indices)
    state = RobotState(ref["q"][ids], ref["v"][ids])
    full = asset.full_state(state)
    np.testing.assert_allclose(full.q, ref["q"], atol=1e-15)
    np.testing.assert_allclose(full.v, ref["v"], atol=1e-15)
    np.testing.assert_array_equal(asset.matrix, ref["P"])
    np.testing.assert_array_equal(asset.constraint_rows, ref["C"])
    np.testing.assert_allclose(asset.constraint_rows @ asset.matrix, 0.0, atol=1e-15)
    M, b = asset.mass_bias(state)
    np.testing.assert_allclose(M, ref["P"].T @ ref["mass"] @ ref["P"], atol=4e-13)
    np.testing.assert_allclose(b, ref["P"].T @ ref["bias"], atol=2e-13)
    assert 0.5 * state.v @ M @ state.v == pytest.approx(ref["energy"], abs=5e-14)
    geo, linear, angular, full_M, _ = robot_dynamics_terms(asset.tree.model, full)
    axis_energy = 0.5 * full.v * (full_M @ full.v)
    np.testing.assert_allclose(axis_energy, 0.5 * ref["v"] * (ref["mass"] @ ref["v"]), atol=8e-14)
    body_energies = []
    for i, m in enumerate(asset.tree.model.mass):
        speed = linear[i] @ full.v
        omega = geo.rotation[i].T @ (angular[i] @ full.v)
        body_energies.append(
            0.5 * (m * (speed @ speed) + omega @ (asset.tree.model.inertia_diagonal[i] * omega))
        )
    np.testing.assert_allclose(body_energies, ref["body_energy"], atol=8e-14)
    idx = asset.tree.link_names.index(ref["point_link"])
    p = asset.tree.authored_to_model_points(ref["point_link"], [ref["point_offset"]])
    J, _ = point_jacobians(asset.tree.model, full.q, [idx], p)
    np.testing.assert_allclose(asset.project_jacobian(J)[0], ref["J"] @ ref["P"], atol=5e-14)
    qdd, multipliers = kkt(ref["mass"], ref["bias"], ref["external"], ref["motor"], ref["C"])
    shot = asset.step(
        state,
        effort=ref["motor"][ids],
        external=RobotInputs(np.zeros(len(ref["q"])), ref["forces"], ref["torques"]),
        dt_s=H,
    )
    np.testing.assert_allclose(shot.full_state.v, ref["v"] + H * qdd, atol=1e-13)
    np.testing.assert_allclose(shot.full_state.q, ref["q"] + H * shot.full_state.v, atol=7e-14)
    np.testing.assert_allclose(shot.loads.contact_effort_full, ref["external"], atol=8e-14)
    np.testing.assert_allclose(
        shot.loads.constraint_effort_full, ref["C"].T @ multipliers, atol=5e-12
    )
    assert shot.loads.multiplier_force is None
    rows = shot.loads.row_forces
    assert [(r.driver, r.follower) for r in rows] == list(ref["row_names"])
    for i, r in enumerate(rows):
        np.testing.assert_allclose(r.multiplier_force, multipliers[i], atol=4e-12)
        parent = ref["names"].index(r.driver)
        assert r.driver_effort == pytest.approx(ref["C"][i, parent] * multipliers[i], abs=4e-12)
        assert r.follower_effort == pytest.approx(multipliers[i], abs=4e-12)
        assert abs(r.constraint_power_W) < 1e-13
    assert abs(shot.loads.constraint_power_W) < 2e-13
    if case == "chain":
        # The middle joint transmits both rows: its total is not its incoming multiplier.
        assert abs(shot.loads.constraint_effort_full[1] - rows[0].multiplier_force) > 0.005
    print(
        case,
        ref["q"].tolist(),
        "lambda",
        multipliers.tolist(),
        "full acceleration",
        qdd.tolist(),
        "energy",
        ref["energy"],
        "axis terms",
        axis_energy.tolist(),
        "body energies",
        body_energies,
    )


def test_each_composed_descendant_bounds_its_own_independent_driver():
    fan_asset = load_mimic_urdf(ROOT / "mimic_fan.urdf")
    np.testing.assert_allclose(fan_asset.lower, [-0.01, -0.004])
    np.testing.assert_allclose(fan_asset.upper, [0.09, 0.016])
    np.testing.assert_allclose(fan_asset.velocity_limit, [0.45, 0.08])
    chain = load_mimic_urdf(ROOT / "mimic_chain.urdf")
    np.testing.assert_allclose(chain.lower, [-0.21])
    np.testing.assert_allclose(chain.upper, [17 / 60])
    np.testing.assert_allclose(chain.velocity_limit, [0.8])
    with pytest.raises(ValueError, match="intersection|position"):
        chain.configuration([0.30])
    with pytest.raises(ValueError, match="velocity"):
        chain.full_state(RobotState(np.zeros(1), np.array([0.81])))


def test_urdf_row_order_does_not_require_the_mechanical_driver_to_appear_first(tmp_path):
    doc = ET.parse(ROOT / "mimic_fan.urdf")
    root = doc.getroot()
    driver = root.find("joint[@name='screw']")
    root.remove(driver)
    root.append(driver)
    path = tmp_path / "forward-reference.urdf"
    doc.write(path)
    asset = load_mimic_urdf(path)
    assert asset.tree.joint_names.index("screw") > asset.tree.joint_names.index("meter")
    full = asset.configuration([0.01, 0.012])
    by_name = dict(zip(asset.tree.joint_names, full, strict=True))
    assert by_name == pytest.approx(dict(lift=0.01, screw=0.012, idler=-0.008, meter=0.008))


@pytest.mark.parametrize("problem", ["cycle", "two_leaders", "empty_chain"])
def test_graph_failures_identify_the_invalid_relation(tmp_path, problem):
    doc = ET.parse(ROOT / ("mimic_chain.urdf" if problem == "empty_chain" else "mimic_fan.urdf"))
    if problem == "cycle":
        ET.SubElement(doc.find(".//joint[@name='screw']"), "mimic", joint="meter")
        match = "cycle.*(screw|meter)|(screw|meter).*cycle"
    elif problem == "two_leaders":
        ET.SubElement(doc.find(".//joint[@name='meter']"), "mimic", joint="idler")
        match = "meter.*mimic|mimic.*meter"
    else:
        lim = doc.find(".//joint[@name='nozzle_drive']/limit")
        lim.set("lower", "0.65")
        lim.set("upper", "0.7")
        match = "intersection.*(nozzle_drive|shaft)|(nozzle_drive|shaft).*intersection"
    path = tmp_path / "invalid.urdf"
    doc.write(path)
    with pytest.raises(ValueError, match=match):
        load_mimic_urdf(path)


def test_public_scene_transmits_actual_gauge_contact_to_two_follower_rows():
    a = load_mimic_urdf(ROOT / "mimic_fan.urdf", gravity=(0, 0, 0))
    ref = fan()
    q = ref["q"][:2].copy()
    center = ref["point"] + np.array([0.015625 + 0.02 - 0.001, 0, 0])
    ball = RigidBodyState(
        center[None], np.array([[0.0, 0.0, 0.0, 1.0]]), np.zeros((1, 3)), np.zeros((1, 3))
    )
    scene = RobotPayloadScene(
        a,
        robot_state=RobotState(q, np.zeros(2)),
        payload_state=ball,
        payload_geometry={"kind": "sphere", "radius_m": 0.02},
        mass_kg=0.125,
        dt_s=0.0005,
        table_center_m=(0, 0, -1),
        table_half_extents_m=(1, 1, 0.01),
        kp=[40, 100],
        kd=[2, 3],
        structure_material=RobotContactMaterial(2000, 2.0, 0.0),
    )
    samples = 0
    max_error = 0.0
    for k in range(24):
        before = scene.observe()
        read = scene.step([q[0], 0.008])
        force = np.zeros(3)
        if read.structural_contacts:
            assert [r["shape"] for r in read.structural_contacts] == ["probe"]
            force = np.array(read.structural_contacts[0]["force_on_link_N"])
            samples += 1
        if k == 0:
            np.testing.assert_allclose(force, [-2, 0, 0], atol=4e-13)
        ext = np.array([force[2], 0, 0, force[0]])
        motor = np.r_[read.drive_effort, 0.0, 0.0]
        acc, lam = kkt(ref["mass"], ref["bias"], ext, motor, ref["C"])
        row = np.array([x.multiplier_force for x in read.mimic_loads.row_forces])
        max_error = max(max_error, np.max(abs(row - lam)))
        np.testing.assert_allclose((read.full_v - before.full_v) / scene.dt_s, acc, atol=2e-12)
        np.testing.assert_allclose(
            0.125 * (read.payload_velocity_m_s - before.payload_velocity_m_s),
            -scene.dt_s * force,
            atol=2e-15,
        )
    assert samples >= 4 and max_error < 2e-12 and read.payload_position_m[0] > center[0]
    print(
        "physical gauge",
        samples,
        "loaded states of",
        read.step_number,
        "payload travel m",
        read.payload_position_m[0] - center[0],
        "row error",
        max_error,
    )
