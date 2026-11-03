"""Force-gated handoff shared by interchangeable, physically actuated robot layouts."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ..constraints.robot_contact import CollisionPlane
from ..constraints.robot_contact_patch import RobotContactPatch
from ..constraints.triangle_prism_surface import TrianglePrismSurface
from ..control.material_grasp import ParallelJawWindow
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.robot import RobotInputs
from ..core.scene import DomainBinding, SceneInputs, SceneModel, SceneState, StepSettings
from ..core.state import ParticleState, StepInputs
from ..core.validation import finite_real
from ..dynamics.robot_contact import RobotContactSystem
from ..solvers.cloth_link_contact import ClothLinkMaterial
from ..solvers.collision_loads import NormalLayer
from ..solvers.compliant_cloth_contact import CompliantClothContactProgram, CompliantPatch
from .compliant_cloth import Case, record_step, trace_start
from .handoff_grippers import (
    BASE_Z,
    CLOSED,
    PAD_HALF_HEIGHT,
    PAD_HALF_LENGTH,
    Gripper,
    grippers_for,
)
from .handoff_reception import CooperativeReception, _shared_support
from .sheet_fabric import (
    HALF_THICKNESS_M,
    HEIGHT_M,
    MASS_KG,
    SheetFabric,
    checked_grid,
    fabric_preset,
    material_for_sheet,
    sheet_grid,
)

PHASES = ("grasp_A", "lift_A", "share", "release_A", "carry_B", "lower_B", "open_B", "settle")


@dataclass
class HandoffCase(Case):
    grippers: dict[str, Gripper]
    receiver_nodes: np.ndarray
    fabric: SheetFabric
    grid: tuple[int, int]
    initial_tilt_deg: float


def build_handoff(
    *,
    grip_friction=0.9,
    robot="cartesian",
    fabric="reference",
    grid=None,
    initial_tilt_deg=0.0,
    finger_pad_radius_m=None,
    finger_backing_clearance_m=None,
    self_contact=False,
):
    """Build the handoff with a fabric preset or a custom SheetFabric."""
    return build_gripper_sheet(
        ("A", "B"),
        grip_friction=grip_friction,
        robot=robot,
        fabric=fabric,
        grid=grid,
        initial_tilt_deg=initial_tilt_deg,
        finger_pad_radius_m=finger_pad_radius_m,
        finger_backing_clearance_m=finger_backing_clearance_m,
        self_contact=self_contact,
    )


def build_gripper_sheet(
    roles,
    *,
    grip_friction,
    robot,
    fabric,
    grid,
    initial_tilt_deg,
    finger_pad_radius_m,
    finger_backing_clearance_m=None,
    self_contact=False,
):
    """Build only the requested physical robot roles around the shared material and table."""
    fabric = fabric_preset(fabric)
    grid = checked_grid(fabric.suggested_grid if grid is None else grid)
    uv, faces = sheet_grid(grid)
    initial_tilt_deg = finite_real(initial_tilt_deg, "initial_tilt_deg")
    if abs(initial_tilt_deg) >= 90:
        raise ValueError("initial tilt must keep the sheet above its lower edge")
    tilt = np.deg2rad(initial_tilt_deg)
    x = np.column_stack(
        (uv[:, 0], uv[:, 1] * np.sin(tilt), HALF_THICKNESS_M + uv[:, 1] * np.cos(tilt))
    )
    # The receiver follows the upper half of the same material edge on any grid.
    receiver_nodes = np.flatnonzero((uv[:, 0] == uv[:, 0].max()) & (uv[:, 1] >= HEIGHT_M / 2))
    cloth = ParticleModel(np.full(len(x), len(x) / MASS_KG), np.array([0.0, 0.0, -9.81]))
    mounted = isinstance(robot, str) and robot == "articulated-palm"
    if mounted:
        if finger_pad_radius_m is None:
            finger_pad_radius_m = 0.002
        if finger_backing_clearance_m is None:
            finger_backing_clearance_m = finger_pad_radius_m / 16
    grippers = (
        grippers_for(robot, roles=roles, finger_pad_radius_m=finger_pad_radius_m)
        if isinstance(robot, str)
        else dict(robot)
    )
    if set(grippers) != set(roles):
        raise ValueError("robot roles do not match the gripper scene")
    models = {n: g.model for n, g in grippers.items()}
    scene = SceneModel(
        (
            DomainBinding("cloth", "particles", cloth),
            *(DomainBinding(n, "robot", m) for n, m in models.items()),
        )
    )
    surfaces = tuple(
        TrianglePrismSurface(f"face_{i}", "cloth", f, half_thickness_m=HALF_THICKNESS_M)
        for i, f in enumerate(faces)
    )
    table = RobotContactPatch(
        "table",
        0,
        np.array([[-0.27, -0.14, 0.0], [0.27, -0.14, 0.0], [0.27, 0.14, 0.0], [-0.27, 0.14, 0.0]]),
    )
    patches = [
        CompliantPatch(
            table,
            width_m=0.0015,
            stiffness_pa_per_m=1.1e6,
            damping_s_per_m=45.0,
            friction=0.45,
            slip_speed_m_s=0.04,
        )
    ]
    pad = np.array(
        [
            [-PAD_HALF_LENGTH, 0.0, -PAD_HALF_HEIGHT],
            [-PAD_HALF_LENGTH, 0.0, PAD_HALF_HEIGHT],
            [PAD_HALF_LENGTH, 0.0, PAD_HALF_HEIGHT],
            [PAD_HALF_LENGTH, 0.0, -PAD_HALF_HEIGHT],
        ]
    )
    for n, gripper in grippers.items():
        left, right = gripper.fingers
        for side, link, sign in (("left", left, 1), ("right", right, -1)):
            patches.append(
                CompliantPatch(
                    RobotContactPatch(f"{n}_{side}", link, pad),
                    n,
                    sign,
                    width_m=0.0015,
                    stiffness_pa_per_m=1.6e6,
                    damping_s_per_m=9.0,
                    friction=grip_friction,
                    radius_m=finger_pad_radius_m,
                    backing_clearance_m=finger_backing_clearance_m,
                    slip_speed_m_s=0.06,
                )
            )
    shapes = tuple(c for g in grippers.values() for c in g.capsules)
    limits = {n: g.limits for n, g in grippers.items() if g.limits is not None}
    contact = None
    if shapes or limits:
        contact = RobotContactSystem(
            models,
            limits=limits,
            capsules=shapes,
            planes=(CollisionPlane("floor", np.array([0.0, 0.0, 1.0]), 0.0),),
            excluded_pairs=tuple(p for g in grippers.values() for p in g.excluded_pairs),
        )
    if not isinstance(self_contact, bool):
        raise TypeError("self_contact must be a boolean")
    self_layers = None
    if self_contact:
        tri = x[np.array(faces)]
        areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
        # This fixture uses the same area stiffness and boundary interaction width
        # as its mounted pads. Each face pair has a separately registered load.
        self_layers = {
            (surfaces[i].key, surfaces[j].key): NormalLayer(
                0.0015, 1.6e6 * min(areas[i], areas[j]), 9.0
            )
            for i, j in combinations(range(len(faces)), 2)
        }
    program = CompliantClothContactProgram(
        scene,
        "cloth",
        material_for_sheet(x, faces, fabric),
        surfaces,
        x,
        patches,
        material_damping_s=0.002,
        max_penetration_m=0.010,
        robot_contact=contact,
        cloth_link_material=ClothLinkMaterial(
            0.0015, 1.6e6, 9.0, contact.material.friction, contact.material.slip_speed_m_s
        )
        if mounted
        else None,
        max_joint_violation=0.02,
        cloth_self_layer=self_layers,
    )
    state = SceneState(
        {
            "cloth": ParticleState(x, np.zeros_like(x)),
            **{n: g.initial for n, g in grippers.items()},
        }
    )
    inputs = SceneInputs(
        {
            "cloth": StepInputs.zeros(len(x)),
            **{n: RobotInputs.zeros(m.dof_count) for n, m in models.items()},
        }
    )
    return HandoffCase(
        program, state, inputs, 0.01, 128, grippers, receiver_nodes, fabric, grid, initial_tilt_deg
    )


def _observations(state, row, weight, body=None):
    x = state.domains["cloth"].position
    o = dict(
        bottom_m=float(x[:, 2].min()),
        cloth_center_x_m=float(x[:, 0].mean()),
        speed_m_s=float(np.max(np.linalg.norm(state.domains["cloth"].velocity, axis=1))),
        table_up_n=float(row["table"]["force_n"][:, 2].sum()),
        weight_n=weight,
    )
    for n in ("A", "B"):
        fingers = [row[n + "_" + k] for k in ("left", "right")]
        o[n + "_both_normal_n"] = float(min(a["normal_force_n"].sum() for a in fingers))
        o[n + "_normal_n"] = float(sum(a["normal_force_n"].sum() for a in fingers))
        o[n + "_up_n"] = float(sum(a["force_n"][:, 2].sum() for a in fingers))
        if body is not None and "cloth_links" in body:
            links = [c for c in body["cloth_links"] if c["robot"] == n]
            o[n + "_link_normal_n"] = float(sum(r["normal_force_n"] for r in links))
            o[n + "_link_up_n"] = float(
                sum(site["force_on_cloth_n"][2] for r in links for site in r["sites"])
            )
    return o


def _travel(start, end, elapsed, duration):
    s = min(max(elapsed / duration, 0.0), 1.0)
    return start + (end - start) * (s * s * (3.0 - 2.0 * s))


def run_handoff(
    *,
    grip_friction=0.9,
    robot="cartesian",
    progress=None,
    fabric="reference",
    grid=None,
    initial_tilt_deg=0.0,
    finger_pad_radius_m=None,
    finger_backing_clearance_m=None,
    receiver_control="snapshot",
    self_contact=False,
):
    if receiver_control not in ("snapshot", "material", "cooperative"):
        raise ValueError("unknown receiver control")
    track_material = receiver_control != "snapshot"
    case = build_handoff(
        grip_friction=grip_friction,
        robot=robot,
        fabric=fabric,
        grid=grid,
        initial_tilt_deg=initial_tilt_deg,
        finger_pad_radius_m=finger_pad_radius_m,
        finger_backing_clearance_m=finger_backing_clearance_m,
        self_contact=self_contact,
    )
    layout = robot if isinstance(robot, str) else "custom"
    case_name = "handoff" if grip_friction else "no-grip"
    trace = trace_start(case, case_name if layout == "cartesian" else f"{layout}-{case_name}")
    trace["robot_layout"] = layout
    trace["fabric"] = case.fabric.record(case.grid)
    trace["initial_tilt_deg"] = case.initial_tilt_deg
    trace["receiver_nodes"] = case.receiver_nodes.tolist()
    state = case.initial
    weight = float(np.sum(1 / case.program.cloth.inverse_mass) * 9.81)
    phase = 0
    entered = 0.0
    stable = 0
    finger_targets = {"A": 0.0, "B": 0.0}

    def position(name):
        return case.grippers[name].position(state.domains[name])

    a0, b0 = position("A"), position("B")
    size = np.ptp(state.domains["cloth"].position, axis=0)
    span = size[0]
    transit_z = BASE_Z + np.linalg.norm(size) + 0.04
    transfer = np.array([a0[0] + 0.025, transit_z])
    bhold = None
    edge_target = None
    finger_layer = next(p for p in case.program.patches if p.geometry.key == "B_left")
    reserve = HALF_THICKNESS_M + (finger_layer.radius_m or 0.0)
    coordinator = None
    if track_material:
        trace["receiver_control"] = receiver_control
    if receiver_control == "cooperative":
        coordinator = CooperativeReception(
            case.receiver_nodes,
            case.dt,
            weight,
            reserve,
            a0,
            b0,
            transfer,
            layer_width=finger_layer.width_m,
            axial_clearance=HALF_THICKNESS_M + finger_layer.width_m,
        )
        trace["coordination_events"] = coordinator.events

    def observed_window(frame):
        parts = frame["patch_vertices_m"]
        return ParallelJawWindow(parts["B_left"], parts["B_right"])

    carry = None
    carry_start = None
    place = None
    lower_start = None
    release_x = None
    trace["controller"] = "finite force-limited gripper drives"

    def emit(name, t, measurements=None):
        event = {"phase": name, "t": t}
        if measurements is not None:
            event["measurements"] = measurements.copy()
        trace["events"].append(event)
        if progress is not None:
            progress(name, t)

    emit(PHASES[0], 0.0)
    with case.program.stepper(
        StepSettings(substeps=case.substeps, position_iterations=1, velocity_iterations=1)
    ) as runtime:
        for i in range(620):
            name = PHASES[phase]
            x = state.domains["cloth"].position
            right = x[case.receiver_nodes].mean(axis=0)[[0, 2]]
            elapsed = i * case.dt - entered
            at = (
                a0
                if phase == 0
                else _travel(a0, transfer, elapsed, 0.55)
                if phase == 1
                else transfer
                if phase == 2
                else transfer + [-0.055, 0.010]
            )
            velocity_command = donor_rate = None
            close_receiver = 2 <= phase <= 5
            receiver_observation = None
            if phase == 0:
                bt = b0
            elif phase == 1:
                bt = right
            elif phase <= 3:
                bt = bhold
            elif phase == 4:
                bt = _travel(carry_start, carry, elapsed, 0.48)
            elif phase == 5:
                bt = _travel(lower_start, place, elapsed, 0.42)
            else:
                bt = place if phase == 6 else place + [0.045, 0.010]
            if coordinator is not None and 1 <= phase <= 3:
                trace["frames"][-1]["actual_tool_velocity_xz_m_s"] = {
                    arm: case.grippers[arm].velocity(state.domains[arm]).tolist()
                    for arm in ("A", "B")
                }
                command = coordinator.command(name, trace["frames"][-1], at, elapsed)
                at, bt = command.donor, command.receiver
                donor_rate, velocity_command = command.donor_rate, command.receiver_rate
                close_receiver = command.closed
                receiver_observation = command.observation
                edge_target = coordinator.anchor
            elif track_material and 2 <= phase <= 5:
                previous = trace["frames"][-1]
                window = observed_window(previous)
                point = edge_target.evaluate(x)
                velocity = edge_target.evaluate(state.domains["cloth"].velocity)
                if (
                    phase <= 3
                    and bhold is None
                    and previous["measurements"]["B_both_normal_n"] > 0.20
                ):
                    bhold = position("B") + np.array([0.0, 0.003])
                    bt = bhold
                if phase <= 3 and bhold is None:
                    # Follow the physical material address until both fingers develop load.
                    bt = (point + case.dt * velocity)[[0, 2]]
                    velocity_command = velocity[[0, 2]]
                    action = "track"
                else:
                    goal = window.center.copy()
                    goal[[0, 2]] = bt
                    bt = window.limit_transport(point, goal, clearance=reserve)[[0, 2]]
                    action = "hold" if phase <= 3 else "transport"
                receiver_observation = dict(
                    action=action,
                    nodes=list(edge_target.nodes),
                    fraction=edge_target.fraction,
                    material_point_m=point.tolist(),
                    material_velocity_m_s=velocity.tolist(),
                    tool_velocity_command_m_s=None
                    if velocity_command is None
                    else velocity_command.tolist(),
                )
            # Limit jaw command speed through layer entry and release; measured contact still determines each transition.
            for arm, closed in (("A", phase <= 2), ("B", close_receiver)):
                target = CLOSED if closed else 0.0
                finger_targets[arm] += float(
                    np.clip(target - finger_targets[arm], -0.09 * case.dt, 0.09 * case.dt)
                )
            requested = {"A": at, "B": bt}
            try:
                controls = {}
                for arm, target in requested.items():
                    if (
                        arm == "B"
                        and coordinator is not None
                        and case.program.cloth_links is not None
                        and receiver_observation is not None
                        and receiver_observation["action"]
                        in ("stage", "await_aperture", "approach", "track")
                    ):
                        controls[arm] = coordinator.drive_approach(
                            arm,
                            case.grippers[arm],
                            state,
                            command,
                            finger_targets[arm],
                            case.program.cloth_links,
                        )
                        requested[arm] = command.receiver
                        continue
                    rate = donor_rate if arm == "A" else velocity_command
                    controls[arm] = case.grippers[arm].command(
                        state.domains[arm],
                        target,
                        finger_targets[arm],
                        case.dt,
                        **({"tool_velocity_m_s": rate} if rate is not None else {}),
                    )
            except SolverFailure as exc:
                trace.update(
                    completed=False,
                    failure_phase=name,
                    failure=str(exc),
                    time_s=i * case.dt,
                    failure_robot=arm,
                    failure_target_xz_m=target.tolist(),
                )
                break
            inputs = SceneInputs(
                {
                    "cloth": case.inputs.domains["cloth"],
                    **{arm: cmd for arm, (cmd, _) in controls.items()},
                }
            )
            out = runtime.step(state, inputs, case.dt)
            if not record_step(case, trace, out, i * case.dt):
                trace["failure_phase"] = name
                break
            state = out.state
            t = (i + 1) * case.dt
            diagnostics = out.report.substeps[-1].diagnostics
            o = _observations(
                state, diagnostics["patches"], weight, diagnostics.get("robot_contact")
            )
            trace["frames"][-1].update(
                phase=name,
                measurements=o,
                actuator_effort={n: inputs.domains[n].effort.tolist() for n in ("A", "B")},
                requested_tool_xz_m={n: x.tolist() for n, x in requested.items()},
                actual_tool_xz_m={n: position(n).tolist() for n in requested},
                joint_target={n: goal.tolist() for n, (_, goal) in controls.items()},
            )
            if receiver_observation is not None:
                trace["frames"][-1]["receiver_observation"] = receiver_observation
            if coordinator is not None and phase in (1, 2, 3, 4) and coordinator.lost(o):
                trace.update(
                    completed=False,
                    failure_phase=name,
                    failure="both grippers lost the sheet before placement",
                    time_s=t,
                )
                break
            visible_target = None
            if name == "grasp_A":
                ready = o["A_both_normal_n"] > 0.20
            elif name == "lift_A" and coordinator is not None:
                ready = coordinator.holder is not None and _shared_support(o, weight)
            elif name == "lift_A":
                ready = (
                    o["bottom_m"] > 0.019
                    and o["table_up_n"] < 0.02 * weight
                    and o["A_up_n"] > 0.25 * weight
                    and np.linalg.norm(position("A") - transfer) < 0.005
                    and np.linalg.norm(position("B") - right) < 0.005
                )
                if ready and track_material:
                    visible_target = observed_window(trace["frames"][-1]).acquire(
                        state.domains["cloth"].position, case.receiver_nodes
                    )
                    ready = visible_target is not None
            elif name == "share":
                ready = _shared_support(o, weight)
                if coordinator is not None:
                    ready = (
                        ready
                        and t - entered >= coordinator.motion.duration
                        and np.linalg.norm(position("A") - transfer) < 0.005
                    )
            elif name == "release_A":
                ready = (
                    o["A_normal_n"] + o.get("A_link_normal_n", 0.0) < 0.01
                    and o["B_both_normal_n"] > 0.20
                    and o["B_up_n"] > 0.3 * weight
                    and o["bottom_m"] > 0.012
                    and o["table_up_n"] < 0.02 * weight
                )
            elif name == "carry_B":
                ready = (
                    o["cloth_center_x_m"] - release_x > 0.035
                    and o["B_up_n"] > 0.20 * weight
                    and np.linalg.norm(position("B") - carry) < 0.005
                    and o["bottom_m"] > 0.012
                )
            elif name == "lower_B":
                ready = (
                    0.35 * weight < o["table_up_n"] < 1.4 * weight
                    and o["B_both_normal_n"] > 0.2
                    and o["speed_m_s"] < 0.075
                    and np.linalg.norm(position("B") - place) < 0.005
                )
            else:
                ready = (
                    o["A_normal_n"] + o.get("A_link_normal_n", 0.0) < 0.01
                    and o["B_normal_n"] + o.get("B_link_normal_n", 0.0) < 0.01
                    and o["table_up_n"] > 0.35 * weight
                    and o["cloth_center_x_m"] - release_x > 0.03
                )
                if name == "settle":
                    ready = (
                        ready
                        and o["speed_m_s"] < 0.05
                        and np.linalg.norm(position("B") - bt) < 0.005
                    )
            stable = stable + 1 if ready else 0
            needed = 10 if name == "settle" else 4 if name == "release_A" else 2
            if stable >= needed:
                if name == "settle":
                    trace.update(completed=True, time_s=t)
                    emit("placed", t, o)
                    break
                if name == "lift_A" and coordinator is not None:
                    coordinator.join(position("A"), position("B"))
                    edge_target = coordinator.anchor
                elif name == "lift_A" and track_material:
                    edge_target = visible_target
                elif name == "lift_A":
                    bhold = state.domains["cloth"].position[case.receiver_nodes].mean(axis=0)[
                        [0, 2]
                    ] + [
                        0.0,
                        0.003,
                    ]
                if name == "release_A":
                    carry_start = position("B")
                    carry = np.array(
                        [carry_start[0] + span / 2 + 0.05, max(transit_z, carry_start[1])]
                    )
                    release_x = o["cloth_center_x_m"]
                if name == "carry_B":
                    lower_start = position("B")
                    place = lower_start + [0.0, 0.0007 - o["bottom_m"]]
                phase += 1
                stable = 0
                entered = t
                emit(PHASES[phase], t, o)
            elif t - entered >= {
                "lift_A": 1.25,
                "carry_B": 1.1,
                "lower_B": 1.25,
                "settle": 2.4,
            }.get(name, 0.65):
                trace.update(
                    completed=False,
                    failure_phase=name,
                    failure="measured phase condition timed out",
                    time_s=t,
                )
                break
        else:
            trace.update(
                completed=False, failure_phase=PHASES[phase], failure="case time budget exhausted"
            )
    trace["metrics"] = dict(
        weight_n=weight,
        phase_count=len(trace["events"]),
        max_sampled_penetration_m=max((r["penetration_m"] for r in trace["samples"]), default=0.0),
        final_center_x_m=float(state.domains["cloth"].position[:, 0].mean()),
        max_robot_penetration_m=max(
            (r["penetration_m"] for r in trace.get("robot_samples", [])), default=0.0
        ),
        max_joint_violation=max(
            (r["joint_violation"] for r in trace.get("robot_samples", [])), default=0.0
        ),
    )
    if case.program.cloth_links is not None:
        trace["metrics"]["max_cloth_link_penetration_m"] = max(
            (s["cloth_link_penetration_m"] for s in trace.get("robot_samples", ())), default=0.0
        )
    return trace
