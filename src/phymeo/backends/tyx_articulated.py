"""tyx numerical contacts composed under the existing global transaction owner."""

import numpy as np

from ..constraints.node_robot import NodeRobotConstraints
from ..core.capabilities import SolverFailure
from ..core.scene import BackendContext, StepSettings
from ..runtime.scene_stepper import SceneStepper
from ..solvers.articulated_coupling import ArticulatedCouplingProgram, ArticulatedWorkspace
from .tyx_math import TyxMath, as_compute
from .tyx_participants import PRECISION


class TyxNodeRobotConstraints(NodeRobotConstraints):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = self.robot.engine

    def _dot(self, x, y):
        from ._tyx_contact_kernels import contact_dot

        x, y = as_compute(x, "contact vector"), as_compute(y, "contact vector")
        if x.shape != (3,) or y.shape != (3,):
            raise ValueError("contact dot requires 3D vectors")
        (result,) = self.engine.call(contact_dot, x, y, np.zeros(1, np.float32), outputs=(2,))
        # Independent relative backward-error check only. A subnormal product
        # may be irrelevant beside a normal term, but cannot silently be the
        # entire answer. Never substitute the host reference into the solver.
        terms = x.astype(np.float64) * y.astype(np.float64)
        reference, scale = float(np.sum(terms)), float(np.sum(np.abs(terms)))
        value = float(result[0])
        if abs(value - reference) > 2e-5 * scale:
            raise SolverFailure("contact dot lost a significant FP32 contribution")
        return value

    def _geometry(self, row):
        from ._tyx_contact_kernels import node_sphere_geometry

        domain, node, sphere = row
        indices = np.array([self.attachments.link_indices[sphere]])
        centers = self.attachments.local_centers[sphere : sphere + 1]
        center = self.robot.point_port(indices, centers).positions()[0]
        node_port = self.participants[domain].point_port(np.array([node]))
        point = node_port.positions()[0]
        normal, gap, failed = self.engine.call(
            node_sphere_geometry,
            self.attachments.radii[sphere],
            as_compute(point, "node point"),
            as_compute(center, "sphere center"),
            np.zeros(3, np.float32),
            np.zeros(1, np.float32),
            np.zeros(1, np.int32),
            outputs=(3, 4, 5),
        )
        if failed[0]:
            raise SolverFailure("node/sphere has coincident or unrepresentable center")
        local = self.robot.world_to_local(indices, point[None])
        robot_port = self.robot.point_port(indices, local)
        return float(gap[0]), normal.astype(np.float64), point, local, node_port, robot_port

    def _response(self, normal, node_port, robot_port):
        response = self.engine.axpy(
            node_port.response(normal[None])[0], robot_port.response(normal[None])[0], 1.0
        )
        value = self._dot(normal, response)
        if not np.isfinite(value) or value < 0:
            raise SolverFailure("contact has invalid normal mass response")
        return value

    def _speed(self, normal, node_port, robot_port):
        relative = self.engine.axpy(node_port.velocities()[0], robot_port.velocities()[0], -1.0)
        return self._dot(normal, relative)

    def _apply(self, phase, row, normal, point, local, magnitude):
        from ._tyx_contact_kernels import paired_contact_load

        if magnitude == 0:
            return
        domain, node, sphere = row
        participant = self.participants[domain]
        link = int(self.attachments.link_indices[sphere])
        node_indices, link_indices = np.array([node]), np.array([link])
        (pair,) = self.engine.call(
            paired_contact_load,
            magnitude,
            as_compute(normal, "contact normal"),
            np.zeros((2, 3), np.float32),
            outputs=(2,),
        )
        load, reaction = pair.astype(np.float64)
        nq, nv = participant.configuration(), participant.velocity()
        rq, rv = self.robot.configuration(), self.robot.velocity()
        generalized = self.robot.generalized_load(link_indices, local, reaction[None])
        if phase == "position":
            participant.apply_position_load(node_indices, None, load[None])
            self.robot.apply_position_load(link_indices, local, reaction[None])
        else:
            participant.apply_velocity_impulses(node_indices, None, load[None])
            self.robot.apply_velocity_impulses(link_indices, local, reaction[None])
        # Deltas below are independent host bookkeeping, not solver corrections.
        self.events.append(
            dict(
                phase=phase,
                domain=domain,
                node=node,
                sphere=sphere,
                link=link,
                world_point=point,
                robot_local_point=local[0],
                normal=normal,
                node_load=load,
                robot_load=reaction,
                robot_generalized_load=generalized,
                node_configuration_before=nq,
                node_velocity_before=nv,
                robot_configuration_before=rq,
                robot_velocity_before=rv,
                node_configuration_delta=participant.configuration() - nq,
                node_velocity_delta=participant.velocity() - nv,
                robot_configuration_delta=self.robot.configuration() - rq,
                robot_velocity_delta=self.robot.velocity() - rv,
                h=self.h,
            )
        )

    def sweep(self, phase):
        from ._tyx_contact_kernels import contact_multiplier

        if phase not in ("position", "velocity"):
            raise ValueError("unknown contact phase")
        multipliers = self.position_load if phase == "position" else self.velocity_impulse
        for index, row in enumerate(self.rows):
            gap, normal, point, local, node_port, robot_port = self._geometry(row)
            if phase == "position":
                error = gap
            else:
                if gap > self.settings.tolerance * self.settings.reference_length:
                    continue
                error = self._speed(normal, node_port, robot_port)
            if error >= 0 and multipliers[index] == 0:
                continue
            response = self._response(normal, node_port, robot_port)
            if response == 0:
                if error < 0:
                    raise SolverFailure("contact cannot respond to penetration or closing motion")
                continue
            (values,) = self.engine.call(
                contact_multiplier,
                multipliers[index],
                error,
                response,
                np.zeros(2, np.float32),
                outputs=(3,),
            )
            self._apply(phase, row, normal, point, local, float(values[1]))
            multipliers[index] = float(values[0])

    def metrics(self, phase):
        from ..core.scene import StageMetrics

        residual, penetration = 0.0, 0.0
        for index, row in enumerate(self.rows):
            gap, normal, _, _, node_port, robot_port = self._geometry(row)
            penetration = max(penetration, -gap)
            if phase == "position":
                error = abs(gap) if self.position_load[index] > 0 else max(0.0, -gap)
                residual = max(residual, error / self.settings.reference_length)
            elif gap <= self.settings.tolerance * self.settings.reference_length:
                speed = self._speed(normal, node_port, robot_port)
                error = abs(speed) if self.velocity_impulse[index] > 0 else max(0.0, -speed)
                residual = max(residual, error / self.settings.reference_speed)
        return StageMetrics({"normal_contact": residual}, penetration=penetration)


class TyxArticulatedCouplingProgram(ArticulatedCouplingProgram):
    _contact_type = TyxNodeRobotConstraints

    def __init__(self, scene, participants, attachments, contacts, robot_domain="robot"):
        participants = dict(participants)
        if robot_domain not in participants:
            raise ValueError("participants must include the robot")
        self.mode = getattr(participants[robot_domain], "mode", None)
        if self.mode not in TyxMath.MODES:
            raise ValueError("tyx program requires explicit matching participant modes")
        self._backend = "tyx:" + self.mode
        super().__init__(scene, participants, attachments, contacts, robot_domain)

    def problem_signature(self):
        return dict(
            super().problem_signature(),
            **PRECISION,
            backend=self._backend,
            device_law="tyx-node-articulated-fp32-v1",
        )

    def create_workspace(self, scene, state, inputs, h, settings, context):
        self.scene.assert_unchanged()
        if scene.identity != self.scene.identity:
            raise ValueError("coupling program must use its bound scene")
        if context.backend != self._backend or context.dtype != np.float64:
            raise ValueError("tyx program backend/storage precision unsupported")
        return _TyxArticulatedWorkspace(self, state, inputs, h, settings, context)


class _TyxArticulatedWorkspace(ArticulatedWorkspace):
    def diagnostics(self):
        result = super().diagnostics()
        counts = [item["execution"] for item in result["participants"].values()]
        result.update(
            PRECISION,
            backend="tyx:" + self.contact.engine.mode,
            execution={
                key: sum(item[key] for item in counts)
                for key in ("launches", "completed_launches", "uploads", "downloads")
            },
        )
        return result


def make_tyx_articulated_runtime(program, *, settings=None):
    if not isinstance(program, TyxArticulatedCouplingProgram):
        raise TypeError("expected TyxArticulatedCouplingProgram")
    return SceneStepper(
        program.scene,
        program,
        StepSettings(substeps=1, tolerance=2e-6) if settings is None else settings,
        context=BackendContext(program.capabilities.backend, np.float64),
        registry=program.registry,
    )
