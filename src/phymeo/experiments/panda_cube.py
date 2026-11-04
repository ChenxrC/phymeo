"""Fixed-base Panda and a free cube, with sampled finite rubber pads on HOST."""

from itertools import product

import numpy as np

from ..constraints.robot_contact import JointLimits, RobotContactMaterial
from ..control.site_ik import SiteIKTarget, solve_multi_site_ik
from ..core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from ..core.robot import RobotState
from ..dynamics.robot_dynamics import robot_dynamics_terms
from ..dynamics.robot_kinematics import _point_jacobians_at, forward_kinematics
from ..dynamics.rotation import matrix_quaternion, quaternion_matrix
from ..solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from ..solvers.sampled_box_contact import box_surface_samples, sample_loads


def _finger_samples(asset):
    links, local = [], []
    for name in ("panda_leftpad", "panda_rightpad"):
        site = asset.sites[name]
        rotation = quaternion_matrix(np.array([site.local_orientation_xyzw]))[0]
        # Four equal-area samples inside the manufacturer's 17.5 x 18.5 mm rubber face.
        offsets = np.array([[x, 0.0, z] for x, z in product((-0.005, 0.005), (-0.005, 0.005))])
        local.extend(np.asarray(site.local_position_m) + offsets @ rotation.T)
        links.extend([site.link_index] * len(offsets))
    return np.asarray(links), np.asarray(local)


def _world_points(geometry, links, local):
    return geometry.position[links] + np.einsum("nij,nj->ni", geometry.rotation[links], local)


class _PassiveGeometry:
    """Bound other Panda collision shapes along each box axis at the sampled poses."""

    def __init__(self, asset):
        local, links, self.starts, self.radii, self.pad = [], [], [], [], []
        for shape in asset.collisions:
            if shape.shape.kind not in ("capsule", "sphere", "hull"):
                raise ValueError(
                    f"unbounded proxy type in this clearance check: {shape.shape.kind}"
                )
            self.starts.append(len(local))
            self.radii.append(shape.shape.radius)
            self.pad.append(shape.name in ("panda_leftfinger_pad", "panda_rightfinger_pad"))
            rows = asset.authored_to_model_points(shape.link_name, shape.local_vertices)
            local.extend(rows)
            links.extend([shape.link_index] * len(rows))
        self.links, self.local = np.array(links), np.array(local)
        self.moving = self.links >= 0
        self.radii = np.array(self.radii)[:, None]
        self.pad = np.array(self.pad)

    def margin(self, points, extents):
        low = np.minimum.reduceat(points, self.starts, axis=0) - self.radii
        high = np.maximum.reduceat(points, self.starts, axis=0) + self.radii
        return np.max(np.maximum(low - extents, -extents - high), axis=1)

    def clearances(self, kin, center, rotation, extents, table_center, table_half):
        points = self.local.copy()
        points[self.moving] = _world_points(kin, self.links[self.moving], self.local[self.moving])
        cube = self.margin((points - center) @ rotation, extents)
        table = self.margin(points - table_center, table_half)
        return float(min(cube[~self.pad])), float(min(table))


def _waypoints(asset, q0, lower_by, lift, travel, opening, closed):
    tcp = asset.sites["panda_hand_tcp"]
    kin = forward_kinematics(asset.model, q0)
    r = kin.rotation[tcp.link_index]
    start = kin.position[tcp.link_index] + r @ tcp.local_position_m
    orient = matrix_quaternion(r @ quaternion_matrix(np.array([tcp.local_orientation_xyzw]))[0])
    bounds = JointLimits(np.array(asset.lower), np.array(asset.upper), 1.0)
    times = [0.0, 0.32, 0.55, 0.92, 1.35, 1.72, 1.95, 2.25]
    poses = [
        (0.0, 0.0, opening),
        (0.0, -lower_by, opening),
        (0.0, -lower_by, closed),
        (0.0, lift - lower_by, closed),
        (travel, lift - lower_by, closed),
        (travel, -lower_by, closed),
        (travel, -lower_by, opening),
        (travel, lift - lower_by, opening),
    ]
    knots = [q0.copy()]
    for x, z, width in poses[1:]:
        target = SiteIKTarget(
            tcp, position_world_m=start + [x, 0.0, z], orientation_world_xyzw=orient
        )
        result = solve_multi_site_ik(
            asset.model,
            knots[-1],
            [target],
            active_joints=range(7),
            limits=bounds,
            tolerance_m=2e-5,
            tolerance_rad=2e-5,
        )
        if not result.converged:
            raise RuntimeError(f"Panda waypoint IK failed: {result.status}")
        knots.append(asset.configuration(result.configuration[:7], width))
    return np.asarray(times), np.asarray(knots)


def run_panda_cube(
    *, friction=0.65, mass_kg=0.10, steps_per_second=1600, record=False, asset_path=None
):
    """Pick, translate and release under gravity. No actuator reads or sets the cube pose.

    The four points per rubber face use a compliant normal and regularized friction.
    Table loads use the eight free box vertices on a horizontal tabletop. Arm collision
    meshes outside the fingertip interface are excluded from this tabletop experiment.
    """
    from ..builders.panda import load_panda

    if not isinstance(steps_per_second, int) or steps_per_second < 800:
        raise ValueError("this sampled control requires at least 800 substeps per second")
    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("the cube must have finite positive mass")
    pad_law = RobotContactMaterial(1200.0, 1.2, friction, 0.04)
    table_law = RobotContactMaterial(2200.0, 3.0, 0.45, 0.04)
    asset = load_panda(asset_path) if asset_path is not None else load_panda()
    gravity = np.array([0.0, 0.0, -9.81])
    opened, closed, descend, lift, travel = 0.076, 0.029, 0.042, 0.065, 0.075
    initial_q = asset.configuration(
        [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4], opened
    )
    arm = RobotState(initial_q, np.zeros(9))
    links, local = _finger_samples(asset)
    start_points = _world_points(forward_kinematics(asset.model, initial_q), links, local)
    grip_above_com = 0.012  # Keep the finger stocks above the payload during side entry.
    start_center = start_points.mean(axis=0) - [0.0, 0.0, descend + grip_above_com]
    outward = start_points[:4].mean(axis=0) - start_points[4:].mean(axis=0)
    if abs(outward[2]) > 1e-6 or np.linalg.norm(outward[:2]) < 0.06:
        raise ValueError("home configuration must present opposed horizontal fingers")
    axis = outward / np.linalg.norm(outward)
    vertical = np.array([0.0, 0.0, 1.0])
    rotation = np.column_stack([axis, np.cross(vertical, axis), vertical])
    half = 0.025
    extents = np.full(3, half)
    vertices = np.array(list(product((-half, half), repeat=3)))
    table_z = start_center[2] - half
    passive = _PassiveGeometry(asset)
    table_half = np.array([travel / 2 + 0.052, 0.055, 0.008])
    table_center = np.array(
        [start_center[0] + travel / 2, start_center[1], table_z - table_half[2]]
    )
    cube_margin = table_margin = float("inf")
    rigid = RigidBodyModel(
        np.array([1.0 / mass_kg]), np.full((1, 3), mass_kg * (2 * half) ** 2 / 6), gravity
    )
    cube = RigidBodyState(
        start_center[None], matrix_quaternion(rotation)[None], np.zeros((1, 3)), np.zeros((1, 3))
    )
    initial_center = cube.position[0].copy()
    times, knots = _waypoints(asset, initial_q, descend, lift, travel, opened, closed)
    lower, upper = np.asarray(asset.lower), np.asarray(asset.upper)
    effort_limit = np.asarray(asset.effort_limit)
    dt = 1.0 / steps_per_second
    step_count = round(times[-1] * steps_per_second)
    integrator = RigidSemiImplicitSolver()
    trace = []
    bilateral_peak = depth_max = limit_error = 0.0
    carry_clearance = carry_displacement = None
    peak_carry_table = 0.0
    min_carry_bilateral = float("inf")
    final_table, final_pad = [], []
    normal_table = np.tile(vertical, (8, 1))
    momentum_error = 0.0
    for step in range(step_count):
        t = step * dt
        segment = min(np.searchsorted(times[1:], t, side="right"), len(times) - 2)
        duration = times[segment + 1] - times[segment]
        u = (t - times[segment]) / duration
        s = u**3 * (10 + u * (-15 + 6 * u))
        ds = 30 * u**2 * (1 - u) ** 2 / duration
        dq = knots[segment + 1] - knots[segment]
        qcmd, vcmd = knots[segment] + s * dq, ds * dq
        q_before = arm.q
        kin, _, _, inertia, bias = robot_dynamics_terms(asset.model, arm)
        samples = _world_points(kin, links, local)
        jac, _ = _point_jacobians_at(asset.model, kin, links, local)
        rotation = quaternion_matrix(cube.orientation)[0]
        pads = box_surface_samples(
            samples,
            jac @ arm.v,
            cube.position[0],
            rotation,
            cube.velocity[0],
            cube.angular_velocity[0],
            extents,
            pad_law,
        )
        corners = vertices @ rotation.T
        world = cube.position[0] + corners
        cube_gap, table_gap = passive.clearances(
            kin, cube.position[0], rotation, extents, table_center, table_half
        )
        cube_margin, table_margin = min(cube_margin, cube_gap), min(table_margin, table_gap)
        if np.any(np.abs(world[:, :2] - table_center[:2]) > table_half[:2]):
            raise RuntimeError("cube left the finite tabletop footprint")
        corner_v = cube.velocity[0] + np.cross(cube.angular_velocity[0], corners)
        table_n, table_f = sample_loads(table_z - world[:, 2], normal_table, corner_v, table_law)
        force = pads.body_force + table_f.sum(axis=0)
        torque = pads.body_torque + np.cross(corners, table_f).sum(axis=0)
        updated = integrator.step(rigid, cube, RigidBodyInputs(force[None], torque[None]), dt)
        if not updated.report.accepted:
            raise RuntimeError(updated.report.reason)
        error = updated.state.velocity[0] - cube.velocity[0] - dt * (gravity + force / mass_kg)
        momentum_error = max(momentum_error, mass_kg * np.linalg.norm(error))
        cube = updated.state
        commanded = bias + inertia @ (900.0 * (qcmd - arm.q) + 60.0 * (vcmd - arm.v))
        commanded[7:] = bias[7:] + 620.0 * (qcmd[7:] - arm.q[7:]) + 5.0 * (vcmd[7:] - arm.v[7:])
        drive = np.clip(commanded, -effort_limit, effort_limit)
        joint_load = np.einsum("ijk,ij->k", jac, pads.forces)
        new_v = arm.v + dt * np.linalg.solve(inertia, drive + joint_load - bias)
        arm = RobotState(arm.q + dt * new_v, new_v)
        limit_error = max(limit_error, float(np.max(lower - arm.q)), float(np.max(arm.q - upper)))
        finger_n = pads.normal_loads.reshape(2, 4).sum(axis=1)
        bilateral_peak = max(bilateral_peak, float(min(finger_n)))
        depth_max = max(depth_max, float(max(pads.depths)))
        # Metrics refer to the advanced body state; the recorded loads acted during this step.
        clearance = (
            cube.position[0, 2]
            + np.min((vertices @ quaternion_matrix(cube.orientation)[0].T)[:, 2])
            - table_z
        )
        elapsed = (step + 1) * dt
        if abs(elapsed - times[4]) <= 0.5 * dt:
            carry_clearance = float(clearance)
            carry_displacement = float(cube.position[0, 0] - initial_center[0])
        if 0.95 <= elapsed <= 1.35:
            min_carry_bilateral = min(min_carry_bilateral, float(min(finger_n)))
            peak_carry_table = max(peak_carry_table, float(table_n.sum()))
        if elapsed > times[-1] - 0.05:
            final_table.append(table_n.sum())
            final_pad.append(np.linalg.norm(pads.forces, axis=1).sum())
        if record:
            trace.append(
                dict(
                    t=elapsed,
                    q=arm.q.tolist(),
                    v=arm.v.tolist(),
                    q_before=q_before.tolist(),
                    pad_points_before_m=samples.tolist(),
                    cube=cube.position[0].tolist(),
                    xyzw=cube.orientation[0].tolist(),
                    velocity=cube.velocity[0].tolist(),
                    omega=cube.angular_velocity[0].tolist(),
                    pad_normal_N=finger_n.tolist(),
                    pad_forces_N=pads.forces.tolist(),
                    table_N=table_n.sum(),
                    drive=drive.tolist(),
                )
            )
    target_error = float(np.linalg.norm(cube.position[0] - initial_center - [travel, 0.0, 0.0]))
    weight = mass_kg * abs(gravity[2])
    success = (
        carry_clearance > 0.035
        and carry_displacement > 0.045
        and target_error < 0.012
        and cube_margin > 0
        and table_margin > 0
        and min_carry_bilateral > 0
        and peak_carry_table < 0.001
        and max(final_pad) < 0.001
        and abs(np.mean(final_table) - weight) < 0.025
        and limit_error < 1e-4
    )
    return dict(
        success=bool(success),
        steps=step_count,
        simulated_seconds=step_count * dt,
        cube_weight_N=weight,
        cube_mass_kg=mass_kg,
        finger_friction=friction,
        table_center_m=table_center.tolist(),
        table_full_size_m=(2 * table_half).tolist(),
        initial_cube_center_m=initial_center.tolist(),
        carry_clearance_m=carry_clearance,
        carry_displacement_m=carry_displacement,
        peak_carry_table_N=peak_carry_table,
        min_carry_bilateral_normal_N=min_carry_bilateral,
        other_robot_cube_axis_clearance_m=cube_margin,
        robot_table_axis_clearance_m=table_margin,
        final_target_error_m=target_error,
        final_table_force_N=float(np.mean(final_table)),
        release_pad_force_N=float(max(final_pad)),
        peak_bilateral_normal_N=bilateral_peak,
        max_joint_limit_violation=limit_error,
        max_pad_penetration_m=depth_max,
        max_cube_momentum_residual_Ns=momentum_error,
        trace=trace,
    )
