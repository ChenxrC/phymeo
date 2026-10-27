"""Composition of real participant stages under a single SceneStepper."""

from contextlib import ExitStack
from dataclasses import fields
from types import MappingProxyType

from ..constraints.node_robot import NodeRobotConstraints, NodeRobotContact
from ..core.scene import SceneState, StageMetrics, StepSettings, fingerprint
from ..geometry.robot_sphere import RobotSphereAttachments
from ..runtime.program import ProgramCapabilities
from ..runtime.registry import StateFamily, StateRegistry
from ..runtime.scene_stepper import SceneStepper


def _capability_signature(capabilities):
    return {
        field.name: tuple(sorted(value)) if isinstance(value, frozenset) else value
        for field in fields(capabilities)
        for value in (getattr(capabilities, field.name),)
    }


class ArticulatedCouplingProgram:
    history_fields = ()

    def __init__(self, scene, participants, attachments, contacts, robot_domain="robot"):
        scene.assert_unchanged()
        participants, contacts = dict(participants), tuple(contacts)
        bindings = {binding.name: binding for binding in scene.domains}
        if set(participants) != set(bindings) or robot_domain not in bindings:
            raise ValueError("participants must match all scene domains including the robot")
        if not isinstance(attachments, RobotSphereAttachments):
            raise TypeError("expected RobotSphereAttachments")
        registry, families = StateRegistry(defaults=False), {}
        for name, participant in participants.items():
            family, caps = participant.family, participant.capabilities
            if not isinstance(family, StateFamily) or not isinstance(caps, ProgramCapabilities):
                raise TypeError("participant requires StateFamily and ProgramCapabilities")
            binding = bindings[name]
            if family.kind != binding.family or fingerprint(participant.model) != fingerprint(
                binding.model
            ):
                raise ValueError("participant family and model must match its domain")
            if (
                caps.backend != "numpy"
                or "float64" not in caps.dtypes
                or family.kind not in caps.families
            ):
                raise ValueError("participant backend/family/precision unsupported")
            if caps.history != "none" or caps.provider_kind != "mathematical":
                raise ValueError("this program requires stateless mathematical participants")
            role = "articulated" if name == robot_domain else "node"
            if participant.point_role != role:
                raise ValueError("participant point role does not match the contact topology")
            if family.kind not in families:
                registry.register(family)
                families[family.kind] = family
            elif families[family.kind] != family:
                raise ValueError("participants disagree on a shared state family")
        robot = participants[robot_domain]
        if any(attachments.link_indices >= robot.model.dof_count):
            raise ValueError("sphere link outside robot model")
        for contact in contacts:
            if not isinstance(contact, NodeRobotContact):
                raise TypeError("expected NodeRobotContact")
            if contact.domain not in bindings or contact.domain == robot_domain:
                raise ValueError("contact requires a node domain")
            if any(contact.node_indices >= participants[contact.domain].model.particle_count):
                raise ValueError("contact node outside model")
            if any(contact.sphere_indices >= len(attachments.radii)):
                raise ValueError("contact sphere outside attachments")
        self.scene, self.participants = scene, MappingProxyType(participants)
        self.attachments, self.contacts, self.robot_domain = attachments, contacts, robot_domain
        self.registry = registry
        self.capabilities = ProgramCapabilities(
            "numpy", {"float64"}, set(families), {"node_sphere", "two_way", "staged_participants"}
        )

    def problem_signature(self):
        return {
            "law": "node-articulated-sphere-position-normal-velocity-v1",
            "scene": self.scene.identity,
            "robot_domain": self.robot_domain,
            "attachments": self.attachments,
            "contacts": self.contacts,
            "participants": tuple(
                (
                    name,
                    participant.point_role,
                    participant.problem_signature(),
                    _capability_signature(participant.capabilities),
                )
                for name, participant in self.participants.items()
            ),
        }

    def create_workspace(self, scene, state, inputs, h, settings, context):
        self.scene.assert_unchanged()
        if scene.identity != self.scene.identity:
            raise ValueError("coupling program must use its bound scene")
        return ArticulatedWorkspace(self, state, inputs, h, settings, context)

    def close(self):
        pass


class ArticulatedWorkspace:
    def __init__(self, program, state, inputs, h, settings, context):
        self.participants = {}
        self.history = {name: value.copy() for name, value in state.history.items()}
        # Register each successful construction before starting the next one.
        with ExitStack() as stack:
            for name, factory in program.participants.items():
                workspace = factory.create_workspace(
                    name, state.domains[name], inputs.domains[name], h, settings, context
                )
                stack.callback(workspace.close)
                self.participants[name] = workspace
            self.contact = NodeRobotConstraints(
                self.participants,
                program.robot_domain,
                program.attachments,
                program.contacts,
                h,
                settings,
            )
            self.resources = stack.pop_all()

    def predict(self):
        for participant in self.participants.values():
            participant.predict()

    def position_sweep(self):
        for participant in self.participants.values():
            participant.position_sweep()
        self.contact.sweep("position")

    def _metrics(self, phase):
        metrics = {
            name: getattr(participant, f"{phase}_metrics")()
            for name, participant in self.participants.items()
        }
        metrics["contact"] = self.contact.metrics(phase)
        return StageMetrics(
            {
                f"{name}/{key}": value
                for name, metric in metrics.items()
                for key, value in metric.residuals.items()
            },
            geometric_error=max(metric.geometric_error for metric in metrics.values()),
            penetration=max(metric.penetration for metric in metrics.values()),
        )

    def position_metrics(self):
        return self._metrics("position")

    def prepare_velocity(self):
        for participant in self.participants.values():
            participant.prepare_velocity()

    def velocity_sweep(self):
        for participant in self.participants.values():
            participant.velocity_sweep()
        self.contact.sweep("velocity")

    def velocity_metrics(self):
        return self._metrics("velocity")

    def result(self):
        return SceneState(
            {name: participant.result() for name, participant in self.participants.items()},
            self.history,
        )

    def diagnostics(self):
        return {
            "participants": {
                name: participant.diagnostics() for name, participant in self.participants.items()
            },
            "contact_events": tuple(self.contact.events),
            "position_load": self.contact.position_load,
            "velocity_impulse": self.contact.velocity_impulse,
        }

    def close(self):
        self.resources.close()


def make_articulated_runtime(program, *, settings=None, context=None):
    if not isinstance(program, ArticulatedCouplingProgram):
        raise TypeError("expected ArticulatedCouplingProgram")
    return SceneStepper(
        program.scene,
        program,
        StepSettings(substeps=1) if settings is None else settings,
        context=context,
        registry=program.registry,
    )
