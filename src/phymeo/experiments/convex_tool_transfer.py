"""A force-driven two-axis arm lifts, drags and withdraws a finite tray from an elastic sheet."""

from dataclasses import dataclass

import numpy as np

from ..constraints.bending import bending_angle_gradient
from ..constraints.triangle_prism_surface import TrianglePrismSurface
from ..core.model import ParticleModel
from ..core.robot import RobotInputs, RobotModel, RobotState
from ..core.scene import DomainBinding, SceneInputs, SceneModel, SceneState, StepSettings
from ..core.state import ParticleState, StepInputs
from ..core.validation import finite_real
from ..dynamics.robot_dynamics import robot_dynamics_terms
from ..dynamics.robot_kinematics import forward_kinematics
from ..solvers.cloth_tool_contact import LinkHullTool
from ..solvers.collision_loads import NormalLayer
from ..solvers.compliant_cloth_contact import CompliantClothContactProgram
from .sheet_fabric import HALF_THICKNESS_M, MASS_KG, fabric_preset, material_for_sheet, sheet_grid

H = 1 / 512
G = np.array([0.0, 0.0, -8.0])
BASE = np.array([0.0, 0.0, 0.27])
HALF_DEPTH = 0.009
WIDTH = 1 / 64
TOTAL_STIFFNESS = 96.0
LIMITS = np.array([22.0, 20.0])
DURATION = {"lift": 0.21875, "drag": 0.25, "withdraw": 0.21875}
POSITION = {"lift": (0.0, 0.032), "drag": (0.055, 0.032), "withdraw": (0.192, -0.205)}


@dataclass(frozen=True)
class ToolTransferCase:
    program: CompliantClothContactProgram
    initial: SceneState
    robot: RobotModel
    local_vertices: np.ndarray
    tool_faces: tuple
    faces: tuple
    mass: np.ndarray
    fabric: object
    footprint: str
    friction: float
    grid: tuple
    drive_mass: np.ndarray
    gravity_bias: np.ndarray


def _tray(footprint):
    if isinstance(footprint, str):
        choices = {
            "rectangle": [(-0.105, -0.055), (0.105, -0.055), (0.105, 0.055), (-0.105, 0.055)],
            "taper": [
                (-0.085, -0.043),
                (0.062, -0.055),
                (0.101, -0.028),
                (0.094, 0.031),
                (0.055, 0.051),
                (-0.085, 0.043),
            ],
        }
        if footprint not in choices:
            raise ValueError("unknown finite tray footprint")
        polygon = np.asarray(choices[footprint], dtype=float)
        name = footprint
    else:
        polygon = np.asarray(footprint, dtype=float)
        name = "custom"
    if (
        polygon.ndim != 2
        or polygon.shape[1] != 2
        or not 3 <= len(polygon) <= 32
        or not np.isfinite(polygon).all()
    ):
        raise ValueError("tray requires between 3 and 32 finite xy vertices")
    for i, a in enumerate(polygon):
        edge = polygon[(i + 1) % len(polygon)] - a
        rel = np.delete(polygon, [i, (i + 1) % len(polygon)], axis=0) - a
        if np.any(edge[0] * rel[:, 1] - edge[1] * rel[:, 0] <= 0):
            raise ValueError("tray footprint must be a strictly convex counter-clockwise ring")
    n = len(polygon)
    vertices = np.vstack([np.c_[polygon, np.full(n, z)] for z in (-HALF_DEPTH, HALF_DEPTH)])
    sides = tuple((i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n))
    return name, vertices, (tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n)), *sides)


def build_tool_transfer(*, fabric="soft", footprint="rectangle", friction=0.75, grid=(2, 2)):
    """Build an ordinary cloth/robot scene. The input geometry, elastic material and friction are replaceable."""
    spec = fabric_preset(fabric)
    coefficient = finite_real(friction, "tool friction")
    if coefficient < 0:
        raise ValueError("tool friction must be nonnegative")
    name, local, tool_faces = _tray(footprint)
    uv, faces = sheet_grid(grid)
    uv = uv.copy()
    uv[:, 1] -= (uv[:, 1].min() + uv[:, 1].max()) / 2
    # Place the material near static normal support, with a fixed small initial cross-slope.
    preload = MASS_KG * (-G[2]) / TOTAL_STIFFNESS
    height = BASE[2] + HALF_DEPTH + HALF_THICKNESS_M + WIDTH - preload
    x = np.c_[uv, height + 0.004 * uv[:, 1]]
    weights = np.zeros(len(x))
    for ids in faces:
        a, b, c = x[list(ids)]
        area = np.linalg.norm(np.cross(b - a, c - a)) / 2
        weights[list(ids)] += area / 3
    mass = MASS_KG * weights / weights.sum()
    cloth = ParticleModel(1 / mass, G)
    robot = RobotModel(
        np.array([-1, 0]),
        ("prismatic", "prismatic"),
        np.array([BASE, [0.0, 0, 0]]),
        np.tile([0.0, 0, 0, 1], (2, 1)),
        np.array([[1.0, 0, 0], [0, 0, 1]]),
        np.array([0.75, 0.5]),
        np.zeros((2, 3)),
        np.array([[0.01, 0.01, 0.01], [0.006, 0.006, 0.006]]),
        G,
    )
    scene = SceneModel(
        (DomainBinding("sheet", "particles", cloth), DomainBinding("arm", "robot", robot))
    )
    surfaces = tuple(
        TrianglePrismSurface(f"cloth-{i}", "sheet", ids, half_thickness_m=HALF_THICKNESS_M)
        for i, ids in enumerate(faces)
    )
    tool = LinkHullTool(
        "tray",
        "arm",
        1,
        local,
        NormalLayer(
            WIDTH,
            TOTAL_STIFFNESS / len(faces),
            damping_s_per_m=6.0,
            friction=coefficient,
            slip_speed_m_s=0.025,
        ),
    )
    material = material_for_sheet(x, faces, spec)
    program = CompliantClothContactProgram(
        scene,
        "sheet",
        material,
        surfaces,
        x,
        (),
        link_hull_tools=(tool,),
        material_damping_s=0.0015,
    )
    rs = RobotState(np.zeros(2), np.zeros(2))
    initial = SceneState({"arm": rs, "sheet": ParticleState(x, np.zeros_like(x))})
    _, _, _, M, bias = robot_dynamics_terms(robot, rs)
    for array in (local, mass, M, bias):
        array.flags.writeable = False
    return ToolTransferCase(
        program,
        initial,
        robot,
        local,
        tool_faces,
        faces,
        mass,
        spec,
        name,
        coefficient,
        tuple(map(int, grid)),
        M,
        bias,
    )


def _reference(phase, elapsed):
    n = DURATION[phase]
    u = np.clip(elapsed / n, 0.0, 1.0)
    start = {"lift": (0.0, 0.0), "drag": POSITION["lift"], "withdraw": POSITION["drag"]}[phase]
    start = np.array(start)
    delta = np.asarray(POSITION[phase]) - start
    progress = u**3 * (10 + u * (-15 + 6 * u))
    velocity = 30 * u**2 * (1 - u) ** 2 / n
    acceleration = 60 * u * (1 - u) * (1 - 2 * u) / n**2
    return start + progress * delta, velocity * delta, acceleration * delta


def _command(case, phase, elapsed, robot):
    ref, rate, accel = _reference(phase, elapsed)
    frequency = np.array([24.0, 28.0])
    requested = case.gravity_bias + case.drive_mass @ (
        accel + 2 * frequency * (rate - robot.v) + frequency**2 * (ref - robot.q)
    )
    return np.clip(requested, -LIMITS, LIMITS), ref


def _rows(observation):
    return [
        dict(
            face=row["face"],
            gap_m=row["gap_m"],
            middle_gap_m=row["middle_gap_m"],
            normal=list(row["normal"]),
            force_n=row["force_n"],
            tool_force_n=row["force_on_tool_n"].tolist(),
            tool_torque_nm=row["torque_about_com_nm"].tolist(),
            power_w=row["power_w"],
            sites=[
                dict(
                    cloth=s["cloth_surface_point_m"],
                    tool=s["tool_point_m"],
                    material=s["cloth_weights"],
                    tool_material=s["tool_weights"],
                    force_on_tool_n=s["force_on_tool_n"],
                )
                for s in row["sites"]
            ],
        )
        for row in observation["contacts"]
    ]


def _frame(case, state, observation, step, phase, motor, reference):
    cloth, arm = state.domains["sheet"], state.domains["arm"]
    geometry = forward_kinematics(case.robot, arm.q)
    tray = case.local_vertices @ geometry.rotation[1].T + geometry.position[1]
    weighted = np.average(cloth.position, weights=case.mass, axis=0)
    v = np.average(cloth.velocity, weights=case.mass, axis=0)
    mat = case.program.material
    lengths = np.linalg.norm(
        cloth.position[mat.distance.indices[:, 1]] - cloth.position[mat.distance.indices[:, 0]],
        axis=1,
    )
    bends = [
        bending_angle_gradient(cloth.position[ids])[0] - rest
        for ids, rest in zip(mat.bending.indices, mat.bending.rest_angle, strict=True)
    ]
    rows = _rows(observation)
    reaction = sum((np.array(c["tool_force_n"]) for c in rows), np.zeros(3))
    return dict(
        step=step,
        time_s=step * H,
        phase=phase,
        nodes_m=cloth.position.tolist(),
        node_velocity_m_s=cloth.velocity.tolist(),
        com_m=weighted.tolist(),
        com_velocity_m_s=v.tolist(),
        q=arm.q.tolist(),
        qd=arm.v.tolist(),
        target_q_m=np.asarray(reference).tolist(),
        motor_n=np.asarray(motor).tolist(),
        robot_links_m=geometry.position.tolist(),
        tool_vertices_m=tray.tolist(),
        contacts=rows,
        material_sites=sum(len(c["sites"]) for c in rows),
        tool_force_n=reaction.tolist(),
        cloth_contact_n=observation["nodal_force_n"].sum(axis=0).tolist(),
        node_contact_n=observation["nodal_force_n"].tolist(),
        generalized_contact_n=observation["efforts"]["arm"].tolist(),
        peak_strain=float(max(abs(lengths / mat.distance.rest_length - 1))),
        peak_bend_rad=float(max(map(abs, bends), default=0)),
    )


def run_tool_transfer(case):
    """Run bounded motor commands until measured lift, travel and an uninterrupted free flight complete."""
    state = case.initial
    patch = case.program.observe_hull_tools(state)
    first = _frame(case, state, patch, 0, "initial", np.zeros(2), np.zeros(2))
    frames = [first]
    events = []
    phase = "lift"
    phase_start = 0
    held = 0
    free = 0
    drag_dx = None
    status = "incomplete"
    reason = None
    weight = -G[2] * case.mass.sum()
    ext = SceneInputs({"sheet": StepInputs.zeros(len(case.mass)), "arm": RobotInputs.zeros(2)})
    with case.program.stepper(StepSettings(substeps=1, record_substeps=True)) as runtime:
        for step in range(1, 513):
            elapsed = (step - phase_start) * H
            effort, ref = _command(case, phase, elapsed - H / 2, state.domains["arm"])
            ext.domains["arm"].effort[:] = effort
            result = runtime.step(state, ext, H)
            if not result.report.accepted:
                reason = f"{phase} step {step}: {result.report.reason}"
                break
            state = result.state
            obs = result.report.substeps[0].diagnostics["hull_tools"]
            row = _frame(case, state, obs, step, phase, effort, ref)
            frames.append(row)
            if phase == "lift":
                support = row["cloth_contact_n"][2] >= 0.5 * weight
                held = held + 1 if support else 0
                if (
                    elapsed >= DURATION[phase]
                    and held >= 3
                    and row["q"][1] >= 0.020
                    and row["com_m"][2] - first["com_m"][2] >= 0.012
                ):
                    events.append(
                        dict(
                            kind="lifted",
                            step=step,
                            com_m=row["com_m"],
                            support_n=row["cloth_contact_n"][2],
                        )
                    )
                    phase, phase_start = "drag", step
                elif elapsed > 0.37:
                    reason = "tool and material failed the measured lift-and-load gate"
                    break
            elif phase == "drag":
                if elapsed >= DURATION[phase] and row["q"][0] >= 0.9 * POSITION["drag"][0]:
                    drag_dx = row["com_m"][0] - events[0]["com_m"][0]
                    events.append(
                        dict(kind="withdraw", step=step, com_m=row["com_m"], drag_dx_m=drag_dx)
                    )
                    phase, phase_start = "withdraw", step
                elif elapsed > 0.40:
                    reason = "finite actuator did not reach the withdrawal position"
                    break
            else:
                free = free + 1 if row["material_sites"] == 0 else 0
                if free >= 32 and frames[step - free]["com_m"][2] - row["com_m"][2] >= 0.005:
                    events.append(
                        dict(kind="independent_flight", step=step, start_step=step - free)
                    )
                    status = "completed"
                    break
    if status != "completed" and reason is None:
        reason = "physical horizon expired before uninterrupted free flight"
    return dict(
        schema="phymeo-convex-tool-transfer-v1",
        dt_s=H,
        status=status,
        reason=reason,
        qualified=bool(status == "completed" and drag_dx is not None and drag_dx >= 0.015),
        model=dict(
            fabric=case.fabric.name,
            stretch_compliance_m_n=case.fabric.stretch_compliance,
            bend_compliance_per_nm=case.fabric.bend_compliance,
            grid=list(case.grid),
            footprint=case.footprint,
            friction=case.friction,
            half_thickness_m=HALF_THICKNESS_M,
            width_m=WIDTH,
            total_stiffness_n_m=TOTAL_STIFFNESS,
            normal_damping_s_m=6.0,
            slip_speed_m_s=0.025,
            material_damping_s=0.0015,
            mass_kg=case.mass.tolist(),
            gravity_m_s2=G.tolist(),
            base_m=BASE.tolist(),
            faces=case.faces,
            tool_local_m=case.local_vertices.tolist(),
            tool_faces=case.tool_faces,
            effort_limits_n=LIMITS.tolist(),
            arm_link_masses_kg=case.robot.mass.tolist(),
            stage_targets_m=POSITION,
            stage_durations_s=DURATION,
        ),
        events=events,
        frames=frames,
    )
