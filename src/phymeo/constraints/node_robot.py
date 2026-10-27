"""Serial frictionless node/link-sphere complementarity and paired load audit."""

from dataclasses import dataclass

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.scene import StageMetrics, _name
from ..geometry.robot_sphere import attachment_indices


@dataclass(frozen=True, eq=False)
class NodeRobotContact:
    domain: str
    node_indices: np.ndarray
    sphere_indices: np.ndarray

    def __post_init__(self):
        _name(self.domain, "contact domain")
        nodes = attachment_indices(self.node_indices, "node_indices")
        spheres = attachment_indices(self.sphere_indices, "sphere_indices")
        if nodes.shape != spheres.shape:
            raise ValueError("contact node and sphere rows must match")
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "sphere_indices", spheres)


class NodeRobotConstraints:
    def __init__(self, participants, robot_domain, attachments, contacts, h, settings):
        self.participants, self.robot = participants, participants[robot_domain]
        self.attachments, self.h, self.settings = attachments, h, settings
        self.rows = tuple(
            (contact.domain, int(node), int(sphere))
            for contact in contacts
            for node, sphere in zip(contact.node_indices, contact.sphere_indices, strict=True)
        )
        self.position_load = np.zeros(len(self.rows))
        self.velocity_impulse = np.zeros(len(self.rows))
        self.events = []

    def _geometry(self, row):
        domain, node, sphere = row
        indices = np.array([self.attachments.link_indices[sphere]])
        centers = self.attachments.local_centers[sphere : sphere + 1]
        center = self.robot.point_port(indices, centers).positions()[0]
        node_port = self.participants[domain].point_port(np.array([node]))
        point = node_port.positions()[0]
        offset = point - center
        distance = np.linalg.norm(offset)
        if not np.isfinite(distance) or distance == 0:
            raise SolverFailure("node/sphere has coincident or unrepresentable center")
        normal = offset / distance
        local = self.robot.world_to_local(indices, point[None])
        robot_port = self.robot.point_port(indices, local)
        return (
            distance - self.attachments.radii[sphere],
            normal,
            point,
            local,
            node_port,
            robot_port,
        )

    @staticmethod
    def _response(normal, node_port, robot_port):
        response = node_port.response(normal[None])[0] + robot_port.response(normal[None])[0]
        value = float(normal @ response)
        if not np.isfinite(value) or value < 0:
            raise SolverFailure("contact has invalid normal mass response")
        return value

    def _apply(self, phase, row, normal, point, local, magnitude):
        if magnitude == 0:
            return
        domain, node, sphere = row
        participant = self.participants[domain]
        link = int(self.attachments.link_indices[sphere])
        node_indices, link_indices = np.array([node]), np.array([link])
        load = normal * magnitude
        nq, nv = participant.configuration(), participant.velocity()
        rq, rv = self.robot.configuration(), self.robot.velocity()
        generalized = self.robot.generalized_load(link_indices, local, -load[None])
        if phase == "position":
            participant.apply_position_load(node_indices, None, load[None])
            self.robot.apply_position_load(link_indices, local, -load[None])
        else:
            participant.apply_velocity_impulses(node_indices, None, load[None])
            self.robot.apply_velocity_impulses(link_indices, local, -load[None])
        self.events.append(
            {
                "phase": phase,
                "domain": domain,
                "node": node,
                "sphere": sphere,
                "link": link,
                "world_point": point,
                "robot_local_point": local[0],
                "normal": normal,
                "node_load": load,
                "robot_load": -load,
                "robot_generalized_load": generalized,
                "node_configuration_before": nq,
                "node_velocity_before": nv,
                "robot_configuration_before": rq,
                "robot_velocity_before": rv,
                "node_configuration_delta": participant.configuration() - nq,
                "node_velocity_delta": participant.velocity() - nv,
                "robot_configuration_delta": self.robot.configuration() - rq,
                "robot_velocity_delta": self.robot.velocity() - rv,
                "h": self.h,
            }
        )

    def sweep(self, phase):
        multipliers = self.position_load if phase == "position" else self.velocity_impulse
        for index, row in enumerate(self.rows):
            gap, normal, point, local, node_port, robot_port = self._geometry(row)
            if phase == "position":
                error = gap
            else:
                if gap > self.settings.tolerance * self.settings.reference_length:
                    continue
                error = float(normal @ (node_port.velocities()[0] - robot_port.velocities()[0]))
            if error >= 0 and multipliers[index] == 0:
                continue
            response = self._response(normal, node_port, robot_port)
            if response == 0:
                if error < 0:
                    raise SolverFailure("contact cannot respond to penetration or closing motion")
                continue
            updated = max(0.0, multipliers[index] - error / response)
            if not np.isfinite(updated):
                raise SolverFailure("contact load is not representable")
            self._apply(phase, row, normal, point, local, updated - multipliers[index])
            multipliers[index] = updated

    def metrics(self, phase):
        residual, penetration = 0.0, 0.0
        for index, row in enumerate(self.rows):
            gap, normal, _, _, node_port, robot_port = self._geometry(row)
            penetration = max(penetration, -gap)
            if phase == "position":
                error = abs(gap) if self.position_load[index] > 0 else max(0.0, -gap)
                residual = max(residual, error / self.settings.reference_length)
            elif gap <= self.settings.tolerance * self.settings.reference_length:
                speed = float(normal @ (node_port.velocities()[0] - robot_port.velocities()[0]))
                error = abs(speed) if self.velocity_impulse[index] > 0 else max(0.0, -speed)
                residual = max(residual, error / self.settings.reference_speed)
        return StageMetrics({"normal_contact": residual}, penetration=penetration)
