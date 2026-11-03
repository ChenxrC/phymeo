"""Force-limited receiver trials using measured material and actual articulated dynamics."""

from dataclasses import dataclass

import numpy as np

from ..control.kinodynamic_approach import SegmentCheck, plan_approach
from ..core.capabilities import SolverFailure
from ..core.robot import RobotState
from ..core.scene import SceneInputs, SceneState
from ..core.state import ParticleState
from ..dynamics.robot_kinematics import forward_kinematics


@dataclass(frozen=True)
class ReceiverControl:
    inputs: object
    joint_target: np.ndarray
    goal: np.ndarray
    rate: np.ndarray


@dataclass(frozen=True)
class _Forecast:
    scene: SceneState
    middle: SceneState | None = None
    sampled_issue: str | None = None


def structural_clearance(contacts, domain, scene):
    """Smallest signed clearance to the configured structural force layer, in metres."""
    pose = forward_kinematics(contacts.models[domain], scene.domains[domain].q)
    geo = {domain: (pose.position, pose.rotation)}
    triangles = scene.domains[contacts.cloth_domain].position[contacts.faces]
    best, pair = np.inf, None
    for index, link in enumerate(contacts.links):
        if link.domain != domain:
            continue
        ends = contacts.endpoints(link, geo)
        skin = link.radius_m + contacts.rho + contacts.law.width_m
        lower = contacts.bounds_distance(ends, triangles) - skin
        for face in np.argsort(lower, kind="stable"):
            if lower[face] > best:
                break
            witness = contacts._witness(contacts.capsules[index], face, ends, triangles[face])
            gap = float(witness.gap_m - contacts.rho[face] - contacts.law.width_m)
            if gap < best:
                best, pair = gap, (link.key, contacts.face_names[face])
    if pair is None:
        raise ValueError("receiver requires registered structural links and material faces")
    return best, pair


def plan_receiver_approach(
    program,
    gripper,
    domain,
    entry,
    inputs,
    target,
    target_velocity,
    aperture,
    budget,
    *,
    material_at=None,
):
    """Forecast from accepted q/v and overwritable actuator inputs; return uncommitted controls.

    Both robots respond to the loaded mass/bias equations of the scene. Material
    samples follow the entered velocity unless the caller supplies a measured or
    modeled path. The coarse midpoint forecast omits unilateral backing impulses.
    A forecast is only a proposal for the original, fully coupled physical stepper.
    """
    contacts = program.cloth_links
    if contacts is None or domain not in contacts.models:
        raise ValueError("receiver needs a registered cloth-link contact domain")
    cloth_name = contacts.cloth_domain
    cloth = entry.domains[cloth_name]
    if material_at is None:

        def material_at(time):
            return ParticleState(cloth.position + time * cloth.velocity, cloth.velocity)

    def observe(observation):
        robot = observation.scene.domains[domain]
        return gripper.position(robot), gripper.velocity(robot)

    def integrate(scene, acc, h, t):
        parts = {cloth_name: material_at(t)}
        for name in program.robots:
            robot = scene.domains[name]
            new_v = robot.v + h * acc[name]
            q_rate = (robot.v + new_v) / 2
            parts[name] = RobotState(robot.q + h * q_rate, new_v)
        return SceneState(parts)

    def body_issue(body):
        if (
            program.max_joint_violation is not None
            and body["joint_violation"] > program.max_joint_violation
        ):
            return "joint_violation"
        if (
            program.max_penetration_m is not None
            and body["penetration_m"] > program.max_penetration_m
        ):
            return "body_penetration"
        return None

    def advance(observation, goal, rate, index):
        scene = observation.scene
        jaw = aperture(index) if callable(aperture) else aperture
        control, joints = gripper.command(
            scene.domains[domain], goal, jaw, budget.dt, tool_velocity_m_s=rate
        )
        if np.any(np.abs(control.effort) > gripper.effort_limits):
            raise SolverFailure("actuator_effort")
        commands = SceneInputs({**inputs.domains, domain: control})
        t, h = index * budget.dt, budget.dt
        acc, _, _, initial_body = program.accelerations(scene, commands)
        middle = integrate(scene, acc, h / 2, t + h / 2)
        mid_acc, _, _, mid_body = program.accelerations(middle, commands)
        end = integrate(scene, mid_acc, h, t + h)
        issue = body_issue(initial_body) or body_issue(mid_body)
        if np.linalg.norm(gripper.velocity(middle.domains[domain])) > budget.peak_speed * (
            1 + 1e-12
        ):
            issue = issue or "measured_speed"
        result = ReceiverControl(control, joints, goal.copy(), rate.copy())
        return result, _Forecast(end, middle, issue)

    def sweep(first, last, index):
        for start, end in ((first.scene, last.middle), (last.middle, last.scene)):
            pair = contacts.approach_blocker(domain, start, end)
            if pair is not None:
                gap, _ = structural_clearance(contacts, domain, end)
                return SegmentCheck("/".join(pair), gap)
        issue = last.sampled_issue or body_issue(program.observe_robot_contact(last.scene))
        return SegmentCheck(issue)

    return plan_approach(
        _Forecast(entry),
        target,
        target_velocity,
        budget,
        observe=observe,
        advance=advance,
        sweep=sweep,
    )
