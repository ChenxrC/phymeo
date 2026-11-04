"""Acquire one actual material boundary address using the mounted receiver's finite link model."""

from dataclasses import asdict

import numpy as np

from ..control.boundary_edge import BoundaryEdges
from ..control.point_ik import solve_point_ik
from ..core.robot import RobotState
from ..core.scene import SceneState
from ..core.state import ParticleState
from ..dynamics.robot_kinematics import forward_kinematics


class MountedBoundaryAccess:
    """Certify the joint-space approach against the current sheet, then latch its material address."""

    def __init__(self, case, domain="B"):
        self.domain = domain
        self.contacts = case.program.cloth_links
        self.gripper = case.grippers[domain]
        if (
            self.contacts is None
            or self.gripper.model.joint_types[:2] != ("revolute", "revolute")
            or not any(c.domain == domain for c in self.contacts.links)
        ):
            raise ValueError(
                "boundary presentation requires a planar mounted hand with finite structural capsules"
            )
        model = self.gripper.model
        pose = forward_kinematics(model, self.gripper.initial.q)
        tip = (
            pose.position[self.gripper.tool_link]
            + pose.rotation[self.gripper.tool_link] @ self.gripper.tool_point
        )
        controlled = [0, 1, *self.gripper.fingers]
        if (
            self.contacts.models[domain] is not model
            or not np.allclose(np.abs(pose.axis[controlled]), [0.0, 1.0, 0.0], atol=1e-12, rtol=0)
            or abs(tip[1]) > 1e-12
        ):
            raise ValueError(
                "planar boundary IK and capsules must bind the same y-invariant receiver"
            )
        self.topology = BoundaryEdges(case.program.faces)
        self.target = None
        self.report = None

    def observe(self, frame, window, clearance):
        if not np.allclose(np.abs(window.axis), [0.0, 1.0, 0.0], atol=1e-12, rtol=0):
            raise ValueError("the mounted planar receiver needs a jaw axis parallel to world y")
        if self.target is not None:
            return self.target
        x = np.asarray(frame["cloth_position_m"])
        v = np.asarray(frame["cloth_velocity_m_s"])
        row = frame["robots"][self.domain]
        robot = RobotState(np.asarray(row["q"]), np.asarray(row["v"]))
        first = SceneState({self.domain: robot, self.contacts.cloth_domain: ParticleState(x, v)})

        def obstruction(address):
            point = address.evaluate(x)
            ik = solve_point_ik(
                self.gripper.model,
                robot.q,
                self.gripper.tool_link,
                self.gripper.tool_point,
                np.array([point[0], 0.0, point[2]]),
                active_joints=(0, 1),
                limits=self.gripper.limits,
            )
            if not ik.converged:
                return ("receiver_ik", ik.status)
            posed = SceneState(
                {**first.domains, self.domain: RobotState(ik.configuration, robot.v)}
            )
            return self.contacts.approach_blocker(self.domain, first, posed)

        chosen = self.topology.choose(x, window, axial_clearance=clearance, obstruction=obstruction)
        self.target = chosen.target
        self.report = dict(
            decision_time_s=frame["t"],
            boundary_edges=len(self.topology.edges),
            axial_candidates=chosen.axial_candidates,
            trials=[asdict(t) for t in chosen.trials],
            target=None if self.target is None else asdict(self.target),
            test="bounded joint IK; finite receiver capsules swept against current sheet",
        )
        return self.target
