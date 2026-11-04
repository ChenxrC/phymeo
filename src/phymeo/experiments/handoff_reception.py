"""Task-space feedback for a donor that stops until a receiver really takes load."""

from dataclasses import dataclass

import numpy as np

from ..control.grasp_transfer import PairedTransport, SoloLiftHold, bounded_approach
from ..control.material_grasp import MaterialEdgeStandoff, ParallelJawWindow
from ..control.planar_presentation import PlanarPresentation
from ..core.capabilities import SolverFailure
from ..core.robot import RobotState
from ..core.scene import SceneState
from ..core.state import ParticleState


def _solo_support(m, weight):
    return (
        m["bottom_m"] > 0.019
        and m["table_up_n"] < 0.02 * weight
        and m["A_both_normal_n"] > 0.20
        and m["A_up_n"] > 0.25 * weight
        and m["B_normal_n"] + m.get("B_link_normal_n", 0.0) < 0.01
    )


def _shared_support(m, weight):
    return (
        m["A_both_normal_n"] > 0.20
        and m["B_both_normal_n"] > 0.20
        and min(m["A_up_n"], m["B_up_n"]) > 0.08 * weight
        and m["table_up_n"] < 0.02 * weight
    )


@dataclass
class GripMotion:
    donor: np.ndarray
    receiver: np.ndarray
    closed: bool
    donor_rate: np.ndarray | None = None
    receiver_rate: np.ndarray | None = None
    observation: dict | None = None


class CooperativeReception:
    def __init__(
        self,
        nodes,
        dt,
        weight,
        clearance,
        donor_home,
        receiver_home,
        transfer,
        *,
        layer_width=0.0,
        axial_clearance=None,
        paired_presentation=False,
        boundary_access=None,
    ):
        self.nodes = nodes
        self.dt = dt
        self.weight = weight
        self.clearance = clearance
        self.home = receiver_home.copy()
        if not np.isfinite(layer_width) or layer_width < 0:
            raise ValueError("contact layer width must be finite and nonnegative")
        if axial_clearance is not None and (
            not np.isfinite(axial_clearance) or axial_clearance < 0
        ):
            raise ValueError("axial clearance must be finite and nonnegative")
        self.axial_clearance = axial_clearance
        self.standoff_clearance = clearance + layer_width
        self.standoff = None
        self.transfer = transfer.copy()
        travel = np.linalg.norm(transfer - donor_home)
        self.peak_speed = 1.5 * travel / 0.55
        self.peak_acceleration = 6 * travel / 0.55**2
        self.solo = SoloLiftHold()
        self.presentation = PlanarPresentation() if paired_presentation else None
        self.boundary = boundary_access
        if self.boundary is not None and axial_clearance is None:
            raise ValueError("boundary selection needs the measured axial clearance")
        self.anchor = None
        self.holder = None
        self.motion = None
        self.events = []
        self.unloaded_frames = 0

    def _event(self, name, frame):
        self.events.append(
            dict(
                event=name,
                t=frame["t"],
                measurements=frame["measurements"].copy(),
                actual_tool_xz_m=frame["actual_tool_xz_m"].copy(),
            )
        )

    def drive_approach(self, domain, gripper, state, motion, aperture, contacts):
        """Check measured-link clearance on the proposed IK and moving material segments."""
        robot = state.domains[domain]
        tip, speed = gripper.position(robot), gripper.velocity(robot)
        stop, stop_rate = bounded_approach(
            tip,
            speed,
            tip,
            np.zeros_like(speed),
            self.dt,
            peak_speed=self.peak_speed,
            peak_acceleration=self.peak_acceleration,
        )
        cloth = state.domains[contacts.cloth_domain]
        predicted = ParticleState(cloth.position + self.dt * cloth.velocity, cloth.velocity)
        blocked = []
        for fraction in (1.0, 0.5, 0.25, 0.0):
            goal = stop + fraction * (motion.receiver - stop)
            rate = stop_rate + fraction * (motion.receiver_rate - stop_rate)
            try:
                drive, joints = gripper.command(
                    robot, goal, aperture, self.dt, tool_velocity_m_s=rate
                )
            except SolverFailure:
                if not fraction:
                    raise
                blocked.append(dict(fraction=fraction, reason="ik_unavailable"))
                continue
            candidate = SceneState(
                {
                    **state.domains,
                    domain: RobotState(joints, robot.v),
                    contacts.cloth_domain: predicted,
                }
            )
            pair = contacts.approach_blocker(domain, state, candidate)
            if pair is not None:
                blocked.append(dict(fraction=fraction, pair=list(pair)))
                if fraction:
                    continue
            # A moving sheet can approach even a braking hand. In that case
            # the contact program remains active and the prediction is marked.
            motion.receiver, motion.receiver_rate = goal, rate
            motion.observation.update(
                tool_velocity_command_m_s=rate.tolist(),
                approach_fraction=fraction,
                predicted_structural_clear=pair is None,
                blocked_approach_candidates=blocked,
            )
            return drive, joints

    def join(self, donor, receiver):
        if self.holder is None:
            raise ValueError("both measured grips are required before shared transport")
        receiver_goal = receiver + self.transfer - donor
        receiver_goal[1] = max(self.transfer[1], receiver[1])
        self.motion = PairedTransport(
            [donor, receiver], [self.transfer, receiver_goal], peak_speed=self.peak_speed
        )

    def lost(self, measurements):
        empty = self.solo.position is not None and all(
            measurements[arm + "_normal_n"] + measurements.get(arm + "_link_normal_n", 0.0) < 0.01
            for arm in "AB"
        )
        self.unloaded_frames = self.unloaded_frames + 1 if empty else 0
        return self.unloaded_frames >= 4

    def command(self, phase, frame, donor_default, elapsed):
        motion = self._command(phase, frame, donor_default, elapsed)
        if (
            self.boundary is not None
            and self.boundary.report is not None
            and motion.observation is not None
        ):
            motion.observation["boundary_selection"] = self.boundary.report
        if self.presentation is None or self.solo.position is None or phase != "lift_A":
            return motion
        row = motion.observation or {}
        if "material_point_m" not in row:
            return motion
        measurement = frame["measurements"]
        active = self.presentation.goal is None
        target = row.get("exterior_point_m", row["material_point_m"])
        velocities = frame["actual_tool_velocity_xz_m_s"]
        result = self.presentation.update(
            donor=frame["actual_tool_xz_m"]["A"],
            donor_velocity=velocities["A"],
            receiver=frame["actual_tool_xz_m"]["B"],
            receiver_velocity=velocities["B"],
            material=np.asarray(row["material_point_m"])[[0, 2]],
            waypoint=np.asarray(target)[[0, 2]],
            waypoint_velocity=np.asarray(row["material_velocity_m_s"])[[0, 2]],
            held_height=self.solo.position[1],
            loaded=measurement["A_both_normal_n"] > 0.20
            and measurement["A_up_n"] > 0.08 * self.weight,
            closing=motion.closed,
            dt=self.dt,
            peak_speed=self.peak_speed,
            peak_acceleration=self.peak_acceleration,
        )
        if result is not None:
            motion.donor, motion.donor_rate, bt, bv, observation = result
            if not motion.closed:
                motion.receiver, motion.receiver_rate = bt, bv
                row["tool_velocity_command_m_s"] = bv.tolist()
            row["presentation"] = observation
            if active:
                self._event("paired_presentation_started", frame)
        return motion

    def _command(self, phase, frame, donor_default, elapsed):
        a, b = (np.asarray(frame["actual_tool_xz_m"][arm]) for arm in "AB")
        m = frame["measurements"]
        if phase == "lift_A":
            waiting = self.solo.position is None
            parked = self.solo.observe(a, _solo_support(m, self.weight))
            if parked is not None:
                if waiting:
                    self._event("solo_lift_held", frame)
                donor_default = parked
        x = np.asarray(frame["cloth_position_m"])
        v = np.asarray(frame["cloth_velocity_m_s"])
        patches = frame["patch_vertices_m"]
        window = ParallelJawWindow(patches["B_left"], patches["B_right"])

        def approach(p, vel, closed):
            actual_velocity = np.asarray(frame["actual_tool_velocity_xz_m_s"]["B"])
            target, rate = bounded_approach(
                b,
                actual_velocity,
                p[[0, 2]],
                vel[[0, 2]],
                self.dt,
                peak_speed=self.peak_speed,
                peak_acceleration=self.peak_acceleration,
            )
            return GripMotion(donor_default, target, closed, receiver_rate=rate)

        def exterior(action):
            if self.standoff is None:
                self.standoff = MaterialEdgeStandoff.from_window(
                    x, self.nodes, window.center, [0, 1, 0]
                )
            if self.standoff is None:
                cmd = approach(window.center, np.zeros(3), False)
                cmd.observation = dict(
                    action=action,
                    reason="no_exterior_direction",
                    tool_velocity_command_m_s=cmd.receiver_rate.tolist(),
                )
                return cmd
            reserve = window.outer_radius + self.standoff_clearance
            goal = self.standoff.point(x, reserve)
            predicted = self.standoff.point(x + self.dt * v, reserve)
            cmd = approach(goal, (predicted - goal) / self.dt, False)
            cmd.observation = dict(
                action=action,
                standoff_direction=list(self.standoff.direction),
                exterior_point_m=goal.tolist(),
                material_point_m=self.standoff.target.evaluate(x).tolist(),
                material_velocity_m_s=self.standoff.target.evaluate(v).tolist(),
                tool_velocity_command_m_s=cmd.receiver_rate.tolist(),
            )
            return cmd

        if phase == "lift_A" and self.solo.position is None:
            return exterior("stage")
        if self.anchor is None and self.boundary is not None:
            target = self.boundary.observe(frame, window, self.axial_clearance)
            in_band = (
                target is not None
                and window.approach_target(
                    x, target.nodes, preferred=target, axial_clearance=self.axial_clearance
                )
                is target
            )
            if not in_band:
                cmd = approach(window.center, np.zeros(3), False)
                cmd.observation = dict(
                    action="await_aperture",
                    reason="no_certified_boundary"
                    if target is None
                    else "selected_address_outside_axial_band",
                    tool_velocity_command_m_s=cmd.receiver_rate.tolist(),
                )
                return cmd
            point, velocity = target.evaluate(x), target.evaluate(v)
            if not window.contains(point, clearance=self.clearance):
                cmd = approach(point, velocity, False)
                cmd.observation = dict(
                    action="approach",
                    material_point_m=point.tolist(),
                    material_velocity_m_s=velocity.tolist(),
                    guidance_nodes=list(target.nodes),
                    guidance_fraction=target.fraction,
                    tool_velocity_command_m_s=cmd.receiver_rate.tolist(),
                )
                return cmd
            self.anchor = target
            self._event("receiver_address_acquired", frame)
        if self.anchor is None:
            visible = window.acquire(x, self.nodes, clearance=self.clearance)
            if visible is None:
                target = self.standoff.target if self.standoff else None
                if self.axial_clearance is not None:
                    target = window.approach_target(
                        x, self.nodes, preferred=target, axial_clearance=self.axial_clearance
                    )
                    if target is None:
                        return exterior("await_aperture")
                if target is None:
                    p, vel = x[self.nodes].mean(0), v[self.nodes].mean(0)
                else:
                    p, vel = target.evaluate(x), target.evaluate(v)
                cmd = approach(p, vel, False)
                cmd.observation = dict(
                    action="approach",
                    material_point_m=p.tolist(),
                    material_velocity_m_s=vel.tolist(),
                    tool_velocity_command_m_s=cmd.receiver_rate.tolist(),
                )
                if self.axial_clearance is not None:
                    cmd.observation.update(
                        guidance_nodes=list(target.nodes), guidance_fraction=target.fraction
                    )
                return cmd
            self.anchor = visible
            self._event("receiver_address_acquired", frame)
        p, vel = self.anchor.evaluate(x), self.anchor.evaluate(v)
        if self.holder is None and m["B_both_normal_n"] > 0.20:
            self.holder = b + np.array([0.0, 0.003])
            self._event("receiver_bilateral_contact", frame)
        if self.holder is None:
            cmd = approach(p, vel, True)
            action = "track"
        else:
            donor_rate = receiver_rate = None
            if phase == "share":
                goals, rates = self.motion.sample(elapsed)
                donor_default, nominal = goals
                donor_rate, receiver_rate = rates
                action = "shared_transport"
            elif phase == "release_A":
                nominal = self.motion.goals[1]
                action = "receive_load"
            else:
                nominal = self.holder
                action = "hold"
            world = window.center.copy()
            world[[0, 2]] = nominal
            target = window.limit_transport(p, world, clearance=self.clearance)[[0, 2]]
            if not np.array_equal(target, nominal):
                receiver_rate = None
            cmd = GripMotion(donor_default, target, True, donor_rate, receiver_rate)
        cmd.observation = dict(
            action=action,
            nodes=list(self.anchor.nodes),
            fraction=self.anchor.fraction,
            material_point_m=p.tolist(),
            material_velocity_m_s=vel.tolist(),
            tool_velocity_command_m_s=None
            if cmd.receiver_rate is None
            else cmd.receiver_rate.tolist(),
        )
        return cmd
