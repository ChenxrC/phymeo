"""Two actuated Cartesian grippers hand the same freely simulated cloth to a table."""

import numpy as np

from ..constraints.bending import BendingConstraints, bending_angle_gradient
from ..constraints.cloth_material import ClothMaterial
from ..constraints.distance import DistanceConstraints
from ..constraints.robot_contact_patch import RobotContactPatch
from ..constraints.triangle_prism_surface import TrianglePrismSurface
from ..core.model import ParticleModel
from ..core.robot import RobotInputs, RobotModel, RobotState
from ..core.scene import DomainBinding, SceneInputs, SceneModel, SceneState, StepSettings
from ..core.state import ParticleState, StepInputs
from ..solvers.compliant_cloth_contact import CompliantClothContactProgram, CompliantPatch
from .compliant_cloth import Case, record_step, trace_start

BASE_Z = 0.039
HALF_OPEN = 0.013
CLOSED = HALF_OPEN - 0.0006
BASE_X = {"A": -0.06, "B": 0.06}
PHASES = ("grasp_A", "lift_A", "share", "release_A", "carry_B", "lower_B", "open_B", "settle")


def _arm(base_x):
    # Two carriage axes and independently actuated opposed fingers.
    return RobotModel(
        np.array([-1, 0, 1, 1]),
        ("prismatic",) * 4,
        np.array(
            [[base_x, 0.0, BASE_Z], [0.0, 0.0, 0.0], [0.0, -HALF_OPEN, 0.0], [0.0, HALF_OPEN, 0.0]]
        ),
        np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)),
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
        np.array([0.12, 0.10, 0.06, 0.06]),
        np.zeros((4, 3)),
        np.full((4, 3), 0.001),
        np.array([0.0, 0.0, -9.81]),
    )


def _cloth_material(x, faces):
    sides = {}
    for a, b, c in faces:
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            sides.setdefault(tuple(sorted((i, j))), []).append((i, j, k))
    edges = np.array(sorted(sides))
    hinges = []
    for incidents in sides.values():
        if len(incidents) == 2:
            (a, b, c), (j, i, d) = incidents
            if (a, b) != (i, j):
                raise ValueError("inconsistent handoff mesh winding")
            hinges.append((a, b, c, d))
    hinge = np.array(hinges)
    rest = np.array([bending_angle_gradient(x[r])[0] for r in hinge])
    return ClothMaterial(
        DistanceConstraints(
            edges,
            np.linalg.norm(x[edges[:, 1]] - x[edges[:, 0]], axis=1),
            np.full(len(edges), 0.001),
        ),
        BendingConstraints(hinge, rest, np.full(len(hinge), 20.0)),
    )


def build_handoff(*, grip_friction=0.9):
    x = np.array([[a, 0.0, 0.0015 + b] for b in (0.0, 0.025, 0.05) for a in (-0.06, 0.06)])
    faces = [(0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)]
    cloth = ParticleModel(np.full(len(x), 1 / 0.008), np.array([0.0, 0.0, -9.81]))
    models = {n: _arm(a) for n, a in BASE_X.items()}
    scene = SceneModel(
        (
            DomainBinding("cloth", "particles", cloth),
            *(DomainBinding(n, "robot", m) for n, m in models.items()),
        )
    )
    surfaces = tuple(
        TrianglePrismSurface(f"face_{i}", "cloth", f, half_thickness_m=0.0015)
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
        [[-0.022, 0.0, -0.019], [-0.022, 0.0, 0.019], [0.022, 0.0, 0.019], [0.022, 0.0, -0.019]]
    )
    for n in models:
        for side, link, sign in (("left", 2, 1), ("right", 3, -1)):
            patches.append(
                CompliantPatch(
                    RobotContactPatch(f"{n}_{side}", link, pad),
                    n,
                    sign,
                    width_m=0.0015,
                    stiffness_pa_per_m=1.6e6,
                    damping_s_per_m=9.0,
                    friction=grip_friction,
                    slip_speed_m_s=0.06,
                )
            )
    program = CompliantClothContactProgram(
        scene,
        "cloth",
        _cloth_material(x, faces),
        surfaces,
        x,
        patches,
        material_damping_s=0.002,
        max_penetration_m=0.010,
    )
    state = SceneState(
        {
            "cloth": ParticleState(x, np.zeros_like(x)),
            "A": RobotState(np.zeros(4), np.zeros(4)),
            "B": RobotState(np.array([0.055, 0.0, 0.0, 0.0]), np.zeros(4)),
        }
    )
    inputs = SceneInputs(
        {"cloth": StepInputs.zeros(len(x)), **{n: RobotInputs.zeros(4) for n in models}}
    )
    return Case(program, state, inputs, 0.01, 128)


def _pose(state, name):
    return np.array([BASE_X[name], BASE_Z]) + state.domains[name].q[:2]


def _command(state, name, target, finger_target):
    robot = state.domains[name]
    wanted = np.r_[target - [BASE_X[name], BASE_Z], np.full(2, finger_target)]
    effort = (
        np.array([165.0, 250.0, 1150.0, 1150.0]) * (wanted - robot.q)
        - np.array([15.0, 15.0, 8.0, 8.0]) * robot.v
    )
    effort[1] += 0.22 * 9.81
    limits = np.array([16.0, 20.0, 5.0, 5.0])
    zero = np.zeros((4, 3))
    return RobotInputs(np.clip(effort, -limits, limits), zero, zero)


def _observations(state, row, weight):
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
    return o


def _travel(start, end, elapsed, duration):
    s = min(max(elapsed / duration, 0.0), 1.0)
    return start + (end - start) * (s * s * (3.0 - 2.0 * s))


def run_handoff(*, grip_friction=0.9, progress=None):
    case = build_handoff(grip_friction=grip_friction)
    trace = trace_start(case, "handoff" if grip_friction else "no-grip")
    state = case.initial
    weight = float(np.sum(1 / case.program.cloth.inverse_mass) * 9.81)
    phase = 0
    entered = 0.0
    stable = 0
    finger_targets = {"A": 0.0, "B": 0.0}
    a0 = _pose(state, "A")
    size = np.ptp(state.domains["cloth"].position, axis=0)
    span = size[0]
    transit_z = BASE_Z + np.linalg.norm(size) + 0.04
    transfer = np.array([a0[0] + 0.025, transit_z])
    bhold = None
    carry = None
    carry_start = None
    place = None
    lower_start = None
    release_x = None
    trace["controller"] = "finite force-limited Cartesian PD"

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
            right = x[[3, 5]].mean(axis=0)[[0, 2]]
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
            if phase == 0:
                bt = np.array([BASE_X["B"] + 0.055, BASE_Z])
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
            # Limit jaw command speed through layer entry and release; measured contact still determines each transition.
            for arm, closed in (("A", phase <= 2), ("B", 2 <= phase <= 5)):
                target = CLOSED if closed else 0.0
                finger_targets[arm] += float(
                    np.clip(target - finger_targets[arm], -0.09 * case.dt, 0.09 * case.dt)
                )
            inputs = SceneInputs(
                {
                    "cloth": case.inputs.domains["cloth"],
                    "A": _command(state, "A", at, finger_targets["A"]),
                    "B": _command(state, "B", bt, finger_targets["B"]),
                }
            )
            out = runtime.step(state, inputs, case.dt)
            if not record_step(case, trace, out, i * case.dt):
                trace["failure_phase"] = name
                break
            state = out.state
            t = (i + 1) * case.dt
            o = _observations(state, out.report.substeps[-1].diagnostics["patches"], weight)
            trace["frames"][-1].update(
                phase=name,
                measurements=o,
                actuator_effort={n: inputs.domains[n].effort.tolist() for n in ("A", "B")},
            )
            if name == "grasp_A":
                ready = o["A_both_normal_n"] > 0.20
            elif name == "lift_A":
                ready = (
                    o["bottom_m"] > 0.019
                    and o["table_up_n"] < 0.02 * weight
                    and o["A_up_n"] > 0.25 * weight
                    and np.linalg.norm(_pose(state, "A") - transfer) < 0.005
                    and np.linalg.norm(_pose(state, "B") - right) < 0.005
                )
            elif name == "share":
                ready = (
                    o["A_both_normal_n"] > 0.20
                    and o["B_both_normal_n"] > 0.20
                    and min(o["A_up_n"], o["B_up_n"]) > 0.08 * weight
                    and o["table_up_n"] < 0.02 * weight
                )
            elif name == "release_A":
                ready = (
                    o["A_normal_n"] < 0.01
                    and o["B_both_normal_n"] > 0.20
                    and o["B_up_n"] > 0.3 * weight
                    and o["bottom_m"] > 0.012
                    and o["table_up_n"] < 0.02 * weight
                )
            elif name == "carry_B":
                ready = (
                    o["cloth_center_x_m"] - release_x > 0.035
                    and o["B_up_n"] > 0.20 * weight
                    and np.linalg.norm(_pose(state, "B") - carry) < 0.005
                    and o["bottom_m"] > 0.012
                )
            elif name == "lower_B":
                ready = (
                    0.35 * weight < o["table_up_n"] < 1.4 * weight
                    and o["B_both_normal_n"] > 0.2
                    and o["speed_m_s"] < 0.075
                    and np.linalg.norm(_pose(state, "B") - place) < 0.005
                )
            else:
                ready = (
                    o["A_normal_n"] < 0.01
                    and o["B_normal_n"] < 0.01
                    and o["table_up_n"] > 0.35 * weight
                    and o["cloth_center_x_m"] - release_x > 0.03
                )
                if name == "settle":
                    ready = (
                        ready
                        and o["speed_m_s"] < 0.05
                        and np.linalg.norm(_pose(state, "B") - bt) < 0.005
                    )
            stable = stable + 1 if ready else 0
            needed = 10 if name == "settle" else 4 if name == "release_A" else 2
            if stable >= needed:
                if name == "settle":
                    trace.update(completed=True, time_s=t)
                    emit("placed", t, o)
                    break
                if name == "lift_A":
                    bhold = state.domains["cloth"].position[[3, 5]].mean(axis=0)[[0, 2]] + [
                        0.0,
                        0.003,
                    ]
                if name == "release_A":
                    carry_start = _pose(state, "B")
                    carry = np.array(
                        [carry_start[0] + span / 2 + 0.05, max(transit_z, carry_start[1])]
                    )
                    release_x = o["cloth_center_x_m"]
                if name == "carry_B":
                    lower_start = _pose(state, "B")
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
    )
    return trace
