"""Fixed-base Panda and a free, uniform payload with finite sampled rubber pads on HOST."""

from hashlib import sha256
from itertools import product
from numbers import Real
from pathlib import Path

import numpy as np

from ..constraints.robot_contact import JointLimits, RobotContactMaterial
from ..control.site_ik import SiteIKTarget, solve_multi_site_ik
from ..core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from ..core.robot import RobotState
from ..dynamics.robot_dynamics import robot_dynamics_terms
from ..dynamics.robot_kinematics import _axis_rotation, _point_jacobians_at, forward_kinematics
from ..dynamics.rotation import matrix_quaternion, quaternion_matrix
from ..solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from ..solvers.sphere_link_contact import SphereLinkContacts
from .panda_payload import SampledPayload


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
        self.names = tuple(s.name for s in asset.collisions)
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
        self.starts = np.array(self.starts)
        self.counts = np.diff(np.r_[self.starts, len(local)])
        self.box_parts = np.array(
            [i for i, shape in enumerate(asset.collisions) if shape.shape.kind == "hull"], dtype=int
        )
        self.round_parts = np.array(
            [i for i, shape in enumerate(asset.collisions) if shape.shape.kind != "hull"], dtype=int
        )
        self.links, self.local = np.array(links), np.array(local)
        self.moving = self.links >= 0
        self.radii = np.array(self.radii)[:, None]
        self.pad = np.array(self.pad)

    def margin(self, points, extents):
        low = np.minimum.reduceat(points, self.starts, axis=0) - self.radii
        high = np.maximum.reduceat(points, self.starts, axis=0) + self.radii
        return np.max(np.maximum(low - extents, -extents - high), axis=1)

    @staticmethod
    def unresolved(step, time, object_gap, table_gap, object_shape, table_shape):
        """First unchecked solid pair. Sphere/robot uses exact distances; the table audit uses bounds."""
        if min(object_gap, table_gap) > 0:
            return None
        is_object = object_gap <= table_gap
        return dict(
            step_index=step,
            t_before_s=time,
            pair="robot_payload" if is_object else "robot_table",
            shape=object_shape if is_object else table_shape,
            clearance_m=min(object_gap, table_gap),
        )

    def sphere_margins(self, points, center, radius):
        """Signed ball clearance to actual round primitives and URDF cuboids on the live arm."""
        margins = np.empty(len(self.starts))
        indices = self.round_parts
        starts = self.starts[indices]
        a = points[starts]
        ab = points[starts + self.counts[indices] - 1] - a
        ab2 = np.einsum("ij,ij->i", ab, ab)
        along = np.divide(
            np.einsum("ij,ij->i", center - a, ab), ab2, out=np.zeros_like(ab2), where=ab2 > 0
        )
        nearest = a + np.clip(along, 0, 1)[:, None] * ab
        margins[indices] = (
            np.linalg.norm(center - nearest, axis=1) - self.radii[indices, 0] - radius
        )

        # make_collider enumerates URDF box vertices in local (-,+) product order.
        indices = self.box_parts
        first = self.starts[indices]
        corners = points[first]
        edges = points[first[:, None] + [4, 2, 1]] - corners[:, None, :]
        lengths = np.linalg.norm(edges, axis=2)
        local = np.einsum(
            "nij,nj->ni", edges / lengths[:, :, None], center - corners - 0.5 * edges.sum(axis=1)
        )
        gaps = np.abs(local) - 0.5 * lengths
        margins[indices] = (
            np.linalg.norm(np.maximum(gaps, 0), axis=1) + np.minimum(gaps.max(axis=1), 0) - radius
        )
        return margins

    def clearances(
        self, kin, center, rotation, extents, table_center, table_half, *, sphere_radius=None
    ):
        points = self.local.copy()
        points[self.moving] = _world_points(kin, self.links[self.moving], self.local[self.moving])
        cube = (
            self.margin((points - center) @ rotation, extents)
            if sphere_radius is None
            else self.sphere_margins(points, center, sphere_radius)
        )
        table = self.margin(points - table_center, table_half)
        cube[self.pad] = np.inf
        i, j = int(np.argmin(cube)), int(np.argmin(table))
        return float(cube[i]), float(table[j]), self.names[i], self.names[j]


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


def _command_at(times, knots, t):
    segment = min(np.searchsorted(times[1:], t, side="right"), len(times) - 2)
    duration = times[segment + 1] - times[segment]
    u = (t - times[segment]) / duration
    s = u**3 * (10 + u * (-15 + 6 * u))
    ds = 30 * u**2 * (1 - u) ** 2 / duration
    dq = knots[segment + 1] - knots[segment]
    return knots[segment] + s * dq, ds * dq


def _servo_effort(arm, qcmd, vcmd, inertia, bias, limit):
    effort = bias + inertia @ (900.0 * (qcmd - arm.q) + 60.0 * (vcmd - arm.v))
    effort[7:] = bias[7:] + 620.0 * (qcmd[7:] - arm.q[7:]) + 5.0 * (vcmd[7:] - arm.v[7:])
    return np.clip(effort, -limit, limit)


def run_panda_cube(
    *,
    friction=0.65,
    mass_kg=0.10,
    steps_per_second=1600,
    record=False,
    asset_path=None,
    box_size_m=None,
    payload_geometry=None,
    structural_response=None,
    closed_width_m=0.029,
    initial_xy_offset_m=(0.0, 0.0),
    initial_yaw_rad=0.0,
    grip_above_com_m=0.012,
):
    """Pick, translate and release under gravity. No actuator reads or sets the cube pose.

    The four points per rubber face use a compliant normal and regularized friction.
    A payload_geometry mapping accepts kind="sphere" with radius_m, or kind="box"
    with full_size_m. Omit it to use box_size_m, which defaults to a 50 mm cube.
    Sphere payloads exchange loads with non-pad solids; structural_response=False keeps the historical clearance audit.
    box_size_m gives full lengths along the initially grasp-aligned, transverse and
    vertical body axes at zero yaw. initial_yaw_rad rotates only the payload about
    world +Z from this orientation. The box rests flat on the finite table.
    initial_xy_offset_m moves the uniform payload COM relative to the gripper in
    world X/Y. grip_above_com_m sets the face-center grasp height. closed_width_m
    commands the jaws. Their actual force follows from contact and actuator effort.
    """
    from ..builders.panda import PANDA_URDF, load_panda

    if not isinstance(steps_per_second, int) or steps_per_second < 800:
        raise ValueError("this sampled control requires at least 800 substeps per second")
    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("the cube must have finite positive mass")
    payload = SampledPayload(box_size_m, payload_geometry=payload_geometry)
    if structural_response is not None and not isinstance(structural_response, (bool, np.bool_)):
        raise ValueError("structural_response must be a boolean or None")
    respond = (
        payload.radius is not None if structural_response is None else bool(structural_response)
    )
    if respond and payload.radius is None:
        raise ValueError("link structural response requires a spherical payload")
    size = payload.full_size
    offset = np.asarray(initial_xy_offset_m, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("initial_xy_offset_m must contain two finite world coordinates")
    if (
        isinstance(initial_yaw_rad, (bool, np.bool_))
        or not isinstance(initial_yaw_rad, Real)
        or not np.isfinite(initial_yaw_rad)
    ):
        raise ValueError("initial_yaw_rad must be a finite scalar angle in radians")
    if (
        not np.isscalar(closed_width_m)
        or not np.isfinite(closed_width_m)
        or not 0 <= closed_width_m <= 0.076
    ):
        raise ValueError("closed_width_m must be within the initial 76 mm jaw opening")
    if not np.isscalar(grip_above_com_m) or not np.isfinite(grip_above_com_m):
        raise ValueError("grip_above_com_m must be finite")
    pad_law = RobotContactMaterial(1200.0, 1.2, friction, 0.04)
    table_law = RobotContactMaterial(2200.0, 3.0, 0.45, 0.04)
    asset = load_panda(asset_path) if asset_path is not None else load_panda()
    gravity = np.array([0.0, 0.0, -9.81])
    opened, closed, descend, lift, travel = 0.076, float(closed_width_m), 0.042, 0.065, 0.075
    initial_q = asset.configuration(
        [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4], opened
    )
    arm = RobotState(initial_q, np.zeros(9))
    links, local = _finger_samples(asset)
    start_points = _world_points(forward_kinematics(asset.model, initial_q), links, local)
    grip_above_com = 0.012  # Keep the finger stocks above the payload during side entry.
    nominal_center = start_points.mean(axis=0) - [0.0, 0.0, descend + grip_above_com]
    outward = start_points[:4].mean(axis=0) - start_points[4:].mean(axis=0)
    if abs(outward[2]) > 1e-6 or np.linalg.norm(outward[:2]) < 0.06:
        raise ValueError("home configuration must present opposed horizontal fingers")
    axis = outward / np.linalg.norm(outward)
    vertical = np.array([0.0, 0.0, 1.0])
    rotation = np.column_stack([axis, np.cross(vertical, axis), vertical])
    if initial_yaw_rad != 0:
        rotation = _axis_rotation(vertical, initial_yaw_rad) @ rotation
    extents = payload.half
    # The calibrated table stays in place for every payload. Its top supports the initial base.
    table_z = nominal_center[2] - 0.025
    start_center = np.array([*(nominal_center[:2] + offset), table_z + extents[2]])
    passive = _PassiveGeometry(asset)
    excluded = [c.shape.key for c, pad in zip(asset.collisions, passive.pad, strict=True) if pad]
    structure = SphereLinkContacts.from_asset(asset, exclude_keys=excluded) if respond else None
    structure_law = RobotContactMaterial(6000.0, 16.0, 0.35, 0.04)
    table_half = np.array([travel / 2 + 0.052, 0.055, 0.008])
    table_center = np.array(
        [nominal_center[0] + travel / 2, nominal_center[1], table_z - table_half[2]]
    )
    if not payload.fits_on_table(start_center, rotation, table_center, table_half):
        raise ValueError("initial payload extends beyond the finite tabletop footprint")
    cube_margin = table_margin = float("inf")
    cube_minimum = table_minimum = first_unresolved = None
    diagonal = payload.inertia(mass_kg)
    rigid = RigidBodyModel(np.array([1.0 / mass_kg]), diagonal[None, :], gravity)
    cube = RigidBodyState(
        start_center[None], matrix_quaternion(rotation)[None], np.zeros((1, 3)), np.zeros((1, 3))
    )
    initial_center = cube.position[0].copy()
    initial_xyzw = cube.orientation[0].copy()
    # The table remains fixed. Position the pad centers relative to this object's actual COM.
    descent = float(start_points[:, 2].mean() - (initial_center[2] + grip_above_com_m))
    if descent <= 0:
        raise ValueError("the requested grip height must be below the open starting pads")
    times, knots = _waypoints(asset, initial_q, descent, lift, travel, opened, closed)
    lower, upper = np.asarray(asset.lower), np.asarray(asset.upper)
    effort_limit = np.asarray(asset.effort_limit)
    dt = 1.0 / steps_per_second
    step_count = round(times[-1] * steps_per_second)
    # This manifest alone can drive another engine; it contains no computed trajectory.
    run_inputs = dict(
        schema="panda-box-control-v1",
        urdf_sha256=sha256(
            Path(PANDA_URDF if asset_path is None else asset_path).read_bytes()
        ).hexdigest(),
        joint_names=list(asset.joint_names),
        initial_q=initial_q.tolist(),
        initial_v=np.zeros(9).tolist(),
        knot_times_s=times.tolist(),
        knot_q=knots.tolist(),
        dt_s=dt,
        steps=step_count,
        mass_kg=mass_kg,
        full_size_m=size.tolist(),
        inertia_diagonal_kg_m2=diagonal.tolist(),
        body_position_m=initial_center.tolist(),
        body_xyzw=initial_xyzw.tolist(),
        initial_linear_velocity_m_s=[0.0, 0.0, 0.0],
        initial_omega_rad_s=[0.0, 0.0, 0.0],
        gravity_m_s2=gravity.tolist(),
        table_center_m=table_center.tolist(),
        table_full_size_m=(2 * table_half).tolist(),
        pad_material=[pad_law.stiffness, pad_law.damping, pad_law.friction, pad_law.slip_speed_m_s],
        table_material=[
            table_law.stiffness,
            table_law.damping,
            table_law.friction,
            table_law.slip_speed_m_s,
        ],
        travel_m=travel,
    )
    if payload.radius is not None:
        run_inputs["payload_geometry"] = payload.descriptor()
    if respond:
        run_inputs["structural_material"] = [
            structure_law.stiffness,
            structure_law.damping,
            structure_law.friction,
            structure_law.slip_speed_m_s,
        ]
    first_contact = None
    integrator = RigidSemiImplicitSolver()
    trace = []
    bilateral_peak = depth_max = limit_error = 0.0
    carry_clearance = carry_displacement = None
    peak_carry_table = 0.0
    min_carry_bilateral = float("inf")
    final_table, final_pad = [], []
    momentum_error = 0.0
    for step in range(step_count):
        t = step * dt
        qcmd, vcmd = _command_at(times, knots, t)
        q_before = arm.q
        kin, linear, angular, inertia, bias = robot_dynamics_terms(asset.model, arm)
        samples = _world_points(kin, links, local)
        jac, _ = _point_jacobians_at(asset.model, kin, links, local)
        rotation = quaternion_matrix(cube.orientation)[0]
        pads = payload.pad_loads(
            samples,
            jac @ arm.v,
            cube.position[0],
            rotation,
            cube.velocity[0],
            cube.angular_velocity[0],
            pad_law,
        )
        cube_gap, table_gap, cube_shape, table_shape = passive.clearances(
            kin,
            cube.position[0],
            rotation,
            extents,
            table_center,
            table_half,
            sphere_radius=payload.radius,
        )
        structural_rows = []
        structural_joint = np.zeros_like(arm.v)
        structural_wrench = np.zeros(6)
        if structure is not None:
            vel = np.column_stack([linear @ arm.v, angular @ arm.v])
            structural = structure.evaluate(
                kin.position,
                kin.rotation,
                kin.com_position,
                vel,
                cube.position[0],
                payload.radius,
                cube.velocity[0],
                cube.angular_velocity[0],
                structure_law,
                fixed_reference_m=asset.root_xyz,
            )
            structural_rows = structural.contacts
            structural_joint = structural.joint_load(linear, angular)
            structural_wrench = structural.payload_wrench
            if first_contact is None and structural_rows:
                first_contact = dict(
                    step_index=step, t_before_s=t, shape=structural_rows[0]["shape"]
                )
        if first_unresolved is None:
            first_unresolved = passive.unresolved(
                step,
                t,
                cube_gap if structure is None else np.inf,
                table_gap,
                cube_shape,
                table_shape,
            )
        if cube_gap < cube_margin:
            cube_margin, cube_minimum = cube_gap, dict(t=t, shape=cube_shape)
        if table_gap < table_margin:
            table_margin, table_minimum = table_gap, dict(t=t, shape=table_shape)
        try:
            table_f, table_n, corners = payload.table_loads(
                cube.position[0],
                rotation,
                cube.velocity[0],
                cube.angular_velocity[0],
                table_center,
                table_half,
                table_law,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if record:
            table_points = cube.position[0] + corners
        force = pads.body_force + table_f.sum(axis=0) + structural_wrench[:3]
        torque = pads.body_torque + np.cross(corners, table_f).sum(axis=0) + structural_wrench[3:]
        updated = integrator.step(rigid, cube, RigidBodyInputs(force[None], torque[None]), dt)
        if not updated.report.accepted:
            raise RuntimeError(updated.report.reason)
        error = updated.state.velocity[0] - cube.velocity[0] - dt * (gravity + force / mass_kg)
        momentum_error = max(momentum_error, mass_kg * np.linalg.norm(error))
        cube = updated.state
        drive = _servo_effort(arm, qcmd, vcmd, inertia, bias, effort_limit)
        joint_load = np.einsum("ijk,ij->k", jac, pads.forces) + structural_joint
        new_v = arm.v + dt * np.linalg.solve(inertia, drive + joint_load - bias)
        arm = RobotState(arm.q + dt * new_v, new_v)
        limit_error = max(limit_error, float(np.max(lower - arm.q)), float(np.max(arm.q - upper)))
        finger_n = pads.normal_loads.reshape(2, 4).sum(axis=1)
        bilateral_peak = max(bilateral_peak, float(min(finger_n)))
        depth_max = max(depth_max, float(max(pads.depths)))
        # Metrics refer to the advanced body state; the recorded loads acted during this step.
        clearance = (
            payload.low_z(cube.position[0], quaternion_matrix(cube.orientation)[0]) - table_z
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
                    table_points_before_m=table_points.tolist(),
                    table_forces_N=table_f.tolist(),
                    applied_payload_wrench=np.r_[force, torque].tolist(),
                    structural_contacts=structural_rows,
                    structural_joint_load=structural_joint.tolist(),
                    drive=drive.tolist(),
                )
            )
    target_error = float(np.linalg.norm(cube.position[0] - initial_center - [travel, 0.0, 0.0]))
    weight = mass_kg * abs(gravity[2])
    carried = (
        carry_clearance > 0.035
        and carry_displacement > 0.045
        and min_carry_bilateral > 0
        and peak_carry_table < 0.001
    )
    placed = (
        target_error < 0.012
        and max(final_pad) < 0.001
        and abs(np.mean(final_table) - weight) < 0.025
    )
    success = carried and placed and cube_margin > 0 and table_margin > 0 and limit_error < 1e-4
    return dict(
        success=bool(success),
        carried_during_transfer=bool(carried),
        released_on_target=bool(placed),
        steps=step_count,
        simulated_seconds=step_count * dt,
        cube_weight_N=weight,
        cube_mass_kg=mass_kg,
        finger_friction=friction,
        cube_full_size_m=size.tolist(),
        payload_geometry=payload.descriptor(),
        cube_inertia_diagonal_kg_m2=diagonal.tolist(),
        closed_width_m=closed,
        grip_above_com_m=float(grip_above_com_m),
        initial_xy_offset_m=offset.tolist(),
        initial_yaw_rad=float(initial_yaw_rad),
        inputs=run_inputs,
        table_center_m=table_center.tolist(),
        table_full_size_m=(2 * table_half).tolist(),
        initial_cube_center_m=initial_center.tolist(),
        initial_arm_q=initial_q.tolist(),
        initial_payload_position_m=initial_center.tolist(),
        initial_payload_xyzw=initial_xyzw.tolist(),
        first_structural_contact=first_contact,
        carry_clearance_m=carry_clearance,
        carry_displacement_m=carry_displacement,
        peak_carry_table_N=peak_carry_table,
        min_carry_bilateral_normal_N=min_carry_bilateral,
        other_robot_cube_axis_clearance_m=cube_margin,
        robot_table_axis_clearance_m=table_margin,
        first_unresolved_structure=first_unresolved,
        minimum_object_axis_clearance_at=cube_minimum,
        minimum_table_axis_clearance_at=table_minimum,
        final_target_error_m=target_error,
        final_table_force_N=float(np.mean(final_table)),
        release_pad_force_N=float(max(final_pad)),
        peak_bilateral_normal_N=bilateral_peak,
        max_joint_limit_violation=limit_error,
        max_pad_penetration_m=depth_max,
        max_cube_momentum_residual_Ns=momentum_error,
        trace=trace,
    )
