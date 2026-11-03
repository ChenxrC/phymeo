"""Measure robot fixtures through public Newton or Phymeo operators in separate processes."""

import json
import warnings
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parent


def _assert_observations(case, observed):
    expected = case["expected"]
    tolerances = case["tolerances"]
    if set(tolerances) - set(expected):
        raise ValueError("Every tolerance must name an expected measurement")
    for name, target in expected.items():
        value = observed[name]
        if name in tolerances:
            if np.shape(value) != np.shape(target):
                raise AssertionError(
                    f"{name}: observed shape {np.shape(value)}, expected {np.shape(target)}"
                )
            np.testing.assert_allclose(
                value, target, rtol=0, atol=tolerances[name], equal_nan=False, err_msg=name
            )
        elif value != target:
            raise AssertionError(f"{name}: observed {value!r}, expected {target!r}")


def _newton_link(builder, fixture, index, label):
    return builder.add_link(
        label=label,
        mass=fixture["body_mass_kg"][index],
        com=fixture["body_com_m"][index],
        inertia=np.diag(fixture["body_inertia_diagonal_kg_m2"][index]),
        lock_inertia=True,
    )


def _build_newton_arm(fixture, source, device):
    import newton
    import warp as wp

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    if source == "builder":
        boom = _newton_link(builder, fixture, 0, "boom")
        tool = _newton_link(builder, fixture, 1, "tool")
        anchors = [
            wp.transform(p, q)
            for p, q in zip(fixture["joint_positions_m"], fixture["joint_frames_xyzw"], strict=True)
        ]
        yaw = builder.add_joint_revolute(
            -1,
            boom,
            parent_xform=anchors[0],
            axis=fixture["joint_axes"][0],
            label="yaw",
            limit_lower=fixture["yaw_limits_rad"][0],
            limit_upper=fixture["yaw_limits_rad"][1],
        )
        slide = builder.add_joint_prismatic(
            boom,
            tool,
            parent_xform=anchors[1],
            axis=fixture["joint_axes"][1],
            label="slide",
            limit_lower=fixture["slide_limits_m"][0],
            limit_upper=fixture["slide_limits_m"][1],
        )
        builder.add_articulation([yaw, slide])
        builder.add_shape_sphere(
            tool,
            radius=fixture["marker_radius_m"],
            label="marker",
            xform=wp.transform(fixture["marker_local_m"], wp.quat_identity()),
        )
    elif source == "arm.urdf":
        builder.add_urdf(str(DIRECTORY / source), floating=False, joint_damping_mode="passive")
    elif source == "arm.mjcf.xml":
        builder.add_mjcf(str(DIRECTORY / source), floating=False)
    elif source == "arm.usda":
        builder.add_usd(str(DIRECTORY / source), load_visual_shapes=False)
    else:
        raise ValueError(f"Unknown arm source {source}")
    reference_site = builder.add_site(-1, label="reference_world")
    model = builder.finalize(device=device)
    if str(model.device) != device:
        raise AssertionError(f"Model is on {model.device}, requested {device}")
    return model, reference_site


def _index(labels, name):
    matches = [index for index, label in enumerate(labels) if label.rsplit("/", 1)[-1] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name!r} among {labels!r}, found {matches}")
    return matches[0]


def _newton_arm(case, fixture, device):
    import newton
    import warp as wp
    from newton.sensors import SensorFrameTransform

    inputs = case["inputs"]
    model, reference = _build_newton_arm(fixture, inputs["source"], device)
    tool = _index(model.body_label, "tool")
    joints = [_index(model.joint_label, name) for name in ("yaw", "slide")]
    qs = model.joint_q_start.numpy()[joints]
    ds = model.joint_qd_start.numpy()[joints]
    types = model.joint_type.numpy()[joints]
    owners, flags = model.shape_body.numpy(), model.shape_flags.numpy()
    colliders = np.flatnonzero(
        (owners == tool) & ((flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0)
    )
    if len(colliders) != 1:
        raise AssertionError(f"Expected one physical tool shape, found {colliders.tolist()}")
    shape = int(colliders[0])
    sensor = SensorFrameTransform(model, shapes=[shape], reference_sites=[reference])
    state = model.state()
    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    ik = case["family"] == "point-ik"
    q[qs] = inputs["seed"] if ik else inputs["q"]
    qd[ds] = [0, 0] if ik else inputs["qd"]
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    if ik:
        from newton.ik import IKJacobianType, IKObjectiveJointLimit, IKObjectivePosition, IKSolver

        point = IKObjectivePosition(
            link_index=tool,
            link_offset=wp.vec3(*fixture["marker_local_m"]),
            target_positions=wp.array([inputs["target_world_m"]], dtype=wp.vec3, device=device),
        )
        limit = IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper)
        solver = IKSolver(
            model=model,
            n_problems=1,
            objectives=[point, limit],
            jacobian_mode=IKJacobianType.ANALYTIC,
        )
        batch = state.joint_q.reshape((1, model.joint_coord_count))
        solver.step(batch, batch, iterations=inputs["iterations"])

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    sensor.update(state)
    marker = sensor.transforms.numpy()[0, :3]
    observed = {
        "source": inputs["source"],
        "fk_device": str(state.body_q.device),
        "frame_sensor_device": str(sensor.transforms.device),
        "marker_world_m": marker.tolist(),
    }
    if ik:
        solved = state.joint_q.numpy()[qs]
        observed.update(
            yaw_rad=float(solved[0]),
            slide_m=float(solved[1]),
            residual_m=float(np.linalg.norm(marker - inputs["target_world_m"])),
        )
        return observed

    origin = state.body_q.numpy()[tool, :3].copy()
    com = model.body_com.numpy()[tool]
    com_velocity = state.body_qd.numpy()[tool, :3].tolist()
    # The fixture puts the sphere centre at the tool COM. This check makes
    # Newton's public body_qd linear component a velocity of the measured point.
    local = model.shape_transform.numpy()[shape, :3]
    np.testing.assert_allclose(
        local, com, rtol=0, atol=1e-6, err_msg="marker centre differs from COM"
    )
    # Measure the slide direction by displacing q; this also covers importers
    # that use different but equivalent child joint frames.
    q[qs[1]] += inputs["slide_probe_m"]
    state.joint_q.assign(q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    direction = (state.body_q.numpy()[tool, :3] - origin) / inputs["slide_probe_m"]
    lo, hi = model.joint_limit_lower.numpy()[ds], model.joint_limit_upper.numpy()[ds]
    observed.update(
        dof_count=model.joint_dof_count,
        tool_origin_m=origin.tolist(),
        slide_axis_world=direction.tolist(),
        marker_velocity_m_s=com_velocity,
        joint_kinds=[newton.JointType(int(t)).name.lower() for t in types],
        tool_colliders=len(colliders),
        tool_collider_kind=newton.GeoType(int(model.shape_type.numpy()[shape])).name.lower(),
        marker_radius_m=float(model.shape_scale.numpy()[shape, 0]),
        local_sphere_m=local.tolist(),
        yaw_limits_rad=[float(lo[0]), float(hi[0])],
        slide_limits_m=[float(lo[1]), float(hi[1])],
    )
    return observed


def _newton_usd_units(inputs):
    import newton
    from pxr import Sdf, Usd, UsdGeom

    # Isolate stage metadata in an anonymous layer so subsequent imports see
    # the original metre-authored fixture, including inside the same process.
    layer = Sdf.Layer.CreateAnonymous("arm_units.usda")
    if not layer.ImportFromString((DIRECTORY / inputs["source"]).read_text(encoding="utf-8")):
        raise ValueError("Could not open USD fixture in memory")
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, inputs["stage_meters_per_unit"])
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        result = newton.ModelBuilder().add_usd(stage, load_visual_shapes=False)
    unit_messages = [
        str(item.message) for item in messages if "non-unit linear units" in str(item.message)
    ]
    return {
        "phase": "HOST asset import",
        "reported_meters_per_unit": float(result["linear_unit"]),
        "nonunit_warning_count": len(unit_messages),
        "warnings": unit_messages,
    }


def _newton_drive(inputs, fixture, device):
    import newton
    from newton.actuators import ClampingMaxEffort, DrivePD
    from newton.solvers import SolverFeatherstone

    builder = newton.ModelBuilder(gravity=inputs["gravity_m_s2"])
    link = _newton_link(builder, fixture, 0, "carriage")
    joint = builder.add_joint_prismatic(
        -1,
        link,
        axis=fixture["joint_axes"][0],
        armature=0.0,
        damping=0.0,
        target_ke=0.0,
        target_kd=0.0,
        label="rail",
    )
    builder.add_articulation([joint])
    builder.add_actuator(
        DrivePD,
        index=builder.joint_qd_start[joint],
        kp=inputs["kp_n_m"],
        kd=inputs["kd_n_s_m"],
        clamping=[(ClampingMaxEffort, {"max_effort": inputs["force_limit_n"]})],
    )
    model = builder.finalize(device=device)
    state, next_state = model.state(), model.state()
    state.joint_q.assign([inputs["initial_q_m"]])
    state.joint_qd.assign([inputs["initial_v_m_s"]])
    control = model.control()
    solver = SolverFeatherstone(model, angular_damping=0.0)
    effort, velocity, position = [], [], []
    for target in inputs["targets_m"]:
        state.clear_forces()
        control.joint_target_q.assign([target])
        control.joint_f.zero_()
        model.actuators[0].step(state, control, dt=inputs["dt_s"])
        effort.append(float(control.joint_f.numpy()[0]))
        solver.step(state, next_state, control, None, inputs["dt_s"])
        state, next_state = next_state, state
        velocity.append(float(state.joint_qd.numpy()[0]))
        position.append(float(state.joint_q.numpy()[0]))
    return {
        "effort_n": effort,
        "velocity_m_s": velocity,
        "position_m": position,
        "actuator_device": str(control.joint_f.device),
        "state_device": str(state.joint_q.device),
    }


def _phymeo_model(fixture, gravity):
    from phymeo import RobotModel

    kinds = tuple(fixture["joint_types"])
    return RobotModel(
        parents=np.arange(-1, len(kinds) - 1),
        joint_types=kinds,
        joint_position=np.asarray(fixture["joint_positions_m"], dtype=float),
        joint_orientation=np.asarray(fixture["joint_frames_xyzw"], dtype=float),
        joint_axis=np.asarray(fixture["joint_axes"], dtype=float),
        mass=np.asarray(fixture["body_mass_kg"], dtype=float),
        com=np.asarray(fixture["body_com_m"], dtype=float),
        inertia_diagonal=np.asarray(fixture["body_inertia_diagonal_kg_m2"], dtype=float),
        gravity=np.asarray(gravity, dtype=float),
    )


def _phymeo_arm(case, fixture, device):
    from phymeo import JointLimits, RobotPointPort, RobotState, solve_point_ik
    from phymeo.dynamics.robot_kinematics import forward_kinematics

    inputs = case["inputs"]
    observed = {}
    if inputs["source"] == "builder":
        model = _phymeo_model(fixture, [0, 0, 0])
        tool = slide = 1
        marker_local = np.asarray(fixture["marker_local_m"], dtype=float)
    elif inputs["source"] in ("arm.urdf", "arm.mjcf.xml"):
        from phymeo.builders.mjcf import load_mjcf
        from phymeo.builders.urdf import load_urdf

        loader, format_name = {
            "arm.urdf": (load_urdf, "URDF"),
            "arm.mjcf.xml": (load_mjcf, "MJCF"),
        }[inputs["source"]]
        asset = loader(DIRECTORY / inputs["source"])
        model = asset.model
        tool = asset.link_names.index("tool")
        yaw, slide = (asset.joint_names.index(name) for name in ("yaw", "slide"))
        shapes = [item for item in asset.collisions if item.link_index == tool]
        (collider,) = shapes
        marker_local = np.asarray(collider.local_vertices[0])
        observed.update(
            asset_import=f"HOST {format_name}",
            joint_kinds=[model.joint_types[i] for i in (yaw, slide)],
            tool_colliders=len(shapes),
            tool_collider_kind=collider.shape.kind,
            marker_radius_m=collider.shape.radius,
            local_sphere_m=marker_local.tolist(),
            yaw_limits_rad=[asset.lower[yaw], asset.upper[yaw]],
            slide_limits_m=[asset.lower[slide], asset.upper[slide]],
        )
    else:
        raise ValueError(f"Unsupported Phymeo arm source {inputs['source']}")
    if case["family"] == "point-ik":
        limits = np.array([fixture["yaw_limits_rad"], fixture["slide_limits_m"]])
        result = solve_point_ik(
            model,
            np.array(inputs["seed"]),
            1,
            marker_local,
            np.array(inputs["target_world_m"]),
            limits=JointLimits(limits[:, 0], limits[:, 1], stiffness=1.0),
            max_iterations=inputs["iterations"],
        )
        q = result.configuration
        port = RobotPointPort(model, RobotState(q, np.zeros(2)), [tool], marker_local[None, :])
        marker = port.positions()[0]
        return {
            "fk_device": "HOST",
            "marker_world_m": marker.tolist(),
            "solver_status": result.status,
            "iterations": result.iterations,
            "yaw_rad": float(q[0]),
            "slide_m": float(q[1]),
            "residual_m": float(np.linalg.norm(marker - inputs["target_world_m"])),
        }
    q = np.array(inputs["q"], dtype=float)
    qd = np.array(inputs["qd"], dtype=float)
    if device == "cuda:0":
        from phymeo.backends.tyx_robot_math import TyxRobotMath

        with TyxRobotMath(model, "cuda") as kernels:
            geometry = kernels.geometry(q)
            linear, _ = kernels.jacobians(q, [tool], marker_local[None, :])
            stats = kernels.stats()
        if stats["completed_launches"] <= 0 or stats["downloads"] <= 0:
            raise AssertionError("Missing completed CUDA geometry kernels or device readback")
        marker = geometry.position[tool] + geometry.rotation[tool] @ marker_local
        velocity = linear[0] @ qd
        observed.update(fk_device="tyx cuda", kernel_stats=stats)
    else:
        geometry = forward_kinematics(model, q)
        port = RobotPointPort(model, RobotState(q, qd), [tool], marker_local[None, :])
        marker, velocity = port.positions()[0], port.velocities()[0]
        observed.update(fk_device="HOST")
    observed.update(
        dof_count=model.dof_count,
        tool_origin_m=geometry.position[tool].tolist(),
        marker_world_m=marker.tolist(),
        slide_axis_world=geometry.axis[slide].tolist(),
        marker_velocity_m_s=velocity.tolist(),
    )
    return observed


def _phymeo_drive(inputs, fixture):
    from phymeo import (
        RobotInputs,
        RobotState,
        SceneInputs,
        SceneState,
        make_robot_runtime,
        solve_bounded_pd,
    )
    from phymeo.dynamics.robot_dynamics import bias_forces, mass_matrix

    model = _phymeo_model(fixture, inputs["gravity_m_s2"])
    current = RobotState(
        np.array([inputs["initial_q_m"]], dtype=float),
        np.array([inputs["initial_v_m_s"]], dtype=float),
    )
    state = SceneState({"robot": current})
    effort, velocity, position = [], [], []
    with make_robot_runtime(model) as runtime:
        for target in inputs["targets_m"]:
            current = state.domains["robot"]
            command = solve_bounded_pd(
                mass_matrix(model, current.q),
                bias_forces(model, current),
                current.q,
                current.v,
                np.array([target], dtype=float),
                np.zeros(1),
                np.array([inputs["kp_n_m"]], dtype=float),
                np.array([inputs["kd_n_s_m"]], dtype=float),
                np.array([inputs["force_limit_n"]], dtype=float),
                inputs["dt_s"],
            )
            loads = RobotInputs.zeros(1)
            loads.effort[:] = command.effort
            step = runtime.step(state, SceneInputs({"robot": loads}), inputs["dt_s"])
            if not step.report.accepted:
                raise AssertionError(f"Single-axis rail step rejected: {step.report}")
            effort.append(float(command.effort[0]))
            state = step.state
            velocity.append(float(state.domains["robot"].v[0]))
            position.append(float(state.domains["robot"].q[0]))
    return {
        "effort_n": effort,
        "velocity_m_s": velocity,
        "position_m": position,
        "state_device": "HOST",
    }


def run_case(case, *, engine, device):
    policy = case[engine]
    if policy["status"] != "run" or device not in policy["devices"]:
        raise ValueError(f"{case['id']} is not registered for {engine}/{device}")
    inputs = case["inputs"]
    if case["family"] == "usd-unit-diagnostic":
        observed = _newton_usd_units(inputs)
    else:
        fixture = json.loads((DIRECTORY / "cases.json").read_text(encoding="utf-8"))["fixtures"][
            inputs["fixture"]
        ]
        if case["family"] == "saturated-rail":
            observed = (
                _newton_drive(inputs, fixture, device)
                if engine == "newton"
                else _phymeo_drive(inputs, fixture)
            )
        else:
            observed = (
                _newton_arm(case, fixture, device)
                if engine == "newton"
                else _phymeo_arm(case, fixture, device)
            )
    _assert_observations(case, observed)
    return observed
