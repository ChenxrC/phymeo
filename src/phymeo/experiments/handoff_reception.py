"""Task-space feedback for a donor that stops until a receiver really takes load."""

from dataclasses import dataclass

import numpy as np

from ..control.grasp_transfer import PairedTransport, SoloLiftHold
from ..control.material_grasp import ParallelJawWindow


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
    def __init__(self, nodes, dt, weight, clearance, donor_home, receiver_home, transfer):
        self.nodes = nodes
        self.dt = dt
        self.weight = weight
        self.clearance = clearance
        self.home = receiver_home.copy()
        self.transfer = transfer.copy()
        self.peak_speed = 1.5 * np.linalg.norm(transfer - donor_home) / 0.55
        self.solo = SoloLiftHold()
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
        a, b = (np.asarray(frame["actual_tool_xz_m"][arm]) for arm in "AB")
        m = frame["measurements"]
        if phase == "lift_A":
            waiting = self.solo.position is None
            parked = self.solo.observe(a, _solo_support(m, self.weight))
            if parked is None:
                return GripMotion(donor_default, self.home, False)
            if waiting:
                self._event("solo_lift_held", frame)
            donor_default = parked
        x = np.asarray(frame["cloth_position_m"])
        v = np.asarray(frame["cloth_velocity_m_s"])
        patches = frame["patch_vertices_m"]
        window = ParallelJawWindow(patches["B_left"], patches["B_right"])
        if self.anchor is None:
            visible = window.acquire(x, self.nodes)
            p = x[self.nodes].mean(0) if visible is None else visible.evaluate(x)
            vel = v[self.nodes].mean(0) if visible is None else visible.evaluate(v)
            if visible is None or np.linalg.norm(b - p[[0, 2]]) >= 0.005:
                return GripMotion(
                    donor_default, (p + self.dt * vel)[[0, 2]], False, receiver_rate=vel[[0, 2]]
                )
            self.anchor = visible
            self._event("receiver_address_acquired", frame)
        p, vel = self.anchor.evaluate(x), self.anchor.evaluate(v)
        if self.holder is None and m["B_both_normal_n"] > 0.20:
            self.holder = b + np.array([0.0, 0.003])
            self._event("receiver_bilateral_contact", frame)
        if self.holder is None:
            cmd = GripMotion(
                donor_default, (p + self.dt * vel)[[0, 2]], True, receiver_rate=vel[[0, 2]]
            )
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
