"""Independent native Featherstone step under the A1 explicit two-contact law.

This evaluates the shared k=2000 N/m, damping=friction=0 law from Newton's own
entry geometry. It does not invoke Newton contact generation or copy Phymeo
states. The original URDF slider and two free balls advance together once.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

H = 0.001
RADIUS = 1 / 16
STIFFNESS = 2000.0
ASSET = Path(__file__).with_name("multiobject_pusher.urdf")


def measure(device):
    import newton as nt
    import warp as wp

    b = nt.ModelBuilder(gravity=(0, 0, 0), up_axis=nt.Axis.Z)
    b.default_shape_cfg.density = 0.0
    b.add_urdf(
        str(ASSET),
        floating=False,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        ignore_inertial_definitions=False,
        joint_damping_mode="passive",
        up_axis=nt.Axis.Z,
    )
    names = [str(name).rsplit("/", 1)[-1] for name in b.joint_label]
    joint = names.index("slide")
    iq, iv = b.joint_q_start[joint], b.joint_qd_start[joint]
    slider = b.joint_child[joint]
    for field in (
        "joint_armature",
        "joint_damping",
        "joint_target_ke",
        "joint_target_kd",
        "joint_limit_ke",
        "joint_limit_kd",
    ):
        getattr(b, field)[iv] = 0.0
    bodies = [slider]
    cfg = b.ShapeConfig(density=0, gap=0, margin=0)
    for label, x in (("first_ball", 0.124), ("second_ball", 0.248)):
        body = b.add_body(
            xform=wp.transform((x, 0, 0.25), wp.quat_identity()),
            mass=1.0,
            inertia=np.eye(3) * (2 / 5) * RADIUS**2,
            lock_inertia=True,
            label=label,
        )
        b.add_shape_sphere(body, radius=RADIUS, cfg=cfg, label=label + "_sphere")
        bodies.append(body)
    b.add_shape_box(
        -1,
        xform=wp.transform((0, 0, -0.03125), wp.quat_identity()),
        hx=0.5,
        hy=0.5,
        hz=0.03125,
        cfg=cfg,
        label="finite_table",
    )
    model = b.finalize(device=device)
    before, after, control = model.state(), model.state(), model.control()
    nt.eval_fk(model, before.joint_q, before.joint_qd, before)
    positions = before.body_q.numpy()[bodies, :3].astype(float)
    initial_twists = before.body_qd.numpy()[bodies].astype(float)
    mass = model.body_mass.numpy()[bodies].astype(float)
    np.testing.assert_allclose(mass, [2, 1, 1], atol=0, rtol=0)
    np.testing.assert_array_equal(initial_twists, 0)
    np.testing.assert_array_equal(model.body_com.numpy()[bodies], 0)
    wrench = np.zeros_like(before.body_f.numpy())
    pairs = []
    for a, c in ((0, 1), (1, 2)):
        delta = positions[c] - positions[a]
        distance = float(np.linalg.norm(delta))
        normal = delta / distance
        depth = max(0.0, 2 * RADIUS - distance)
        force = STIFFNESS * depth * normal
        wrench[bodies[a], :3] -= force
        wrench[bodies[c], :3] += force
        pairs.append(
            dict(
                body_rows=[bodies[a], bodies[c]],
                distance_m=distance,
                depth_m=depth,
                normal_world=normal.tolist(),
                normal_load_N=float(np.linalg.norm(force)),
                first_surface_m=(positions[a] + RADIUS * normal).tolist(),
                second_surface_m=(positions[c] - RADIUS * normal).tolist(),
                force_on_first_N=(-force).tolist(),
                force_on_second_N=force.tolist(),
            )
        )
    before.clear_forces()
    before.body_f.assign(wrench)
    solver = nt.solvers.SolverFeatherstone(model, angular_damping=0.0)
    solver.step(before, after, control, None, H)
    final_positions = after.body_q.numpy()[bodies, :3].astype(float)
    final_twists = after.body_qd.numpy()[bodies].astype(float)
    impulse = mass[:, None] * (final_twists[:, :3] - initial_twists[:, :3])
    momentum = (mass[:, None] * final_twists[:, :3]).sum(axis=0)
    error = impulse - H * wrench[bodies, :3]
    # Gates were fixed before execution; position is the native FP32 public readout.
    expected_v = np.array([[-0.001, 0, 0], [0, 0, 0], [0.002, 0, 0]])
    checks = dict(
        two_2N_loads=all(abs(row["normal_load_N"] - 2) < 2e-5 for row in pairs),
        actual_linear_velocity=bool(np.max(abs(final_twists[:, :3] - expected_v)) < 3e-8),
        no_spin=bool(np.max(abs(final_twists[:, 3:])) < 1e-8),
        one_drift=bool(np.max(abs(final_positions - positions - H * final_twists[:, :3])) < 3e-8),
        slider_coordinate=abs(float(after.joint_q.numpy()[iq]) + 1e-6) < 3e-11,
        slider_velocity=abs(float(after.joint_qd.numpy()[iv]) + 0.001) < 3e-8,
        impulse_balance=bool(np.max(abs(error)) < 2e-9),
        momentum_conserved=bool(np.max(abs(momentum)) < 2e-9),
    )
    return dict(
        schema="robot-multiobject-explicit-native-v1",
        version=nt.__version__,
        device=str(wp.get_device(device)),
        dt_s=H,
        units="SI",
        interpretation="common explicit contact law; native Featherstone integration; no native contact solve",
        urdf_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        loaded_body_rows=bodies,
        loaded_masses_kg=mass.tolist(),
        shape_types=model.shape_type.numpy().tolist(),
        shape_body=model.shape_body.numpy().tolist(),
        material=dict(stiffness_N_m=STIFFNESS, damping_Ns_m=0, friction=0),
        entry_position_m=positions.tolist(),
        entry_twists=initial_twists.tolist(),
        pairs=pairs,
        applied_body_wrench_N_Nm=wrench.tolist(),
        final_position_m=final_positions.tolist(),
        final_twists=final_twists.tolist(),
        joint_q_after=after.joint_q.numpy().tolist(),
        joint_v_after=after.joint_qd.numpy().tolist(),
        momentum_kg_m_s=momentum.tolist(),
        impulse_error_Ns=error.tolist(),
        checks=checks,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("output already exists; preserve the first observation")
    result = measure(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
