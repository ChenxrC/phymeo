"""Input-state owner response for the HOST scene tangent adapter.

A mechanism is one mass block, including every shared ancestor and a physical
floating root. These functions consume prepared dynamics; they do not prepare
controls, integrate, reconstruct inertia from a URDF, or publish a live state.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..dynamics.rotation import quaternion_matrix
from .material_stations import OwnerPose


def _array(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape or not np.isfinite(raw).all():
        raise ValueError(f"{name} requires finite shape {shape}")
    array = np.asarray(raw, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(shape)


def _skew(value):
    x, y, z = value
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


@dataclass(frozen=True, eq=False)
class OwnerPort:
    """Owned full physical J and one reduced positive-mass response block.

    free_effort excludes contact, but includes the original bounded motor force,
    declared external COM loads, gravity and bias, each exactly once. Fixed rigid
    bodies use a six-by-zero projection; their contact wrench is still recorded.
    """

    address: tuple
    robot: bool
    projection: np.ndarray
    mass: np.ndarray
    velocity: np.ndarray
    full_velocity: np.ndarray
    linear: np.ndarray
    angular: np.ndarray
    centers: np.ndarray
    free_effort: np.ndarray
    response_mass: np.ndarray | None = None

    @property
    def response_metric(self):
        """Integrator response; physical mass remains distinct for diagnostics."""
        return self.mass if self.response_mass is None else self.response_mass

    @property
    def full_count(self):
        return len(self.full_velocity)

    @property
    def body_count(self):
        return len(self.centers)

    def index(self, owner):
        if (owner.world, owner.body) != self.address:
            raise ValueError("material owner differs from its response block")
        if not self.robot:
            if owner.kind != "rigid" or owner.local_frame != "rigid_com":
                raise ValueError("rigid response requires the registered COM owner")
            return 0
        if (
            owner.kind == "robot_root"
            and owner.local_frame == "robot_root"
            and owner.link_index == -1
        ):
            return -1
        if (
            owner.kind not in {"robot_link", "robot_root"}
            or owner.local_frame != "robot_model_link"
            or owner.link_index is None
            or not 0 <= owner.link_index < self.body_count
            or (owner.kind == "robot_root" and owner.link_index != 0)
        ):
            raise ValueError("material owner has no physical robot Jacobian")
        return owner.link_index

    def jacobian(self, owner, point):
        index = self.index(owner)
        point = _array(point, (3,), "application point")
        if index == -1:
            return np.zeros((3, self.full_count))
        return self.linear[index] - _skew(point - self.centers[index]) @ self.angular[index]

    def center(self, owner, frames):
        index = self.index(owner)
        return np.asarray(frames[owner].position_m) if index == -1 else self.centers[index]

    def free_velocity(self, contact, h):
        contact = _array(contact, (self.full_count,), "full contact effort")
        if not len(self.velocity):
            return np.empty(0)
        return self.velocity + h * np.linalg.solve(
            self.response_metric,
            self.free_effort + self.projection.T @ contact,
        )


def _mass(value, size):
    matrix = _array(value, (size, size), "reduced mass")
    if size:
        scale = np.max(np.abs(matrix))
        if not scale or np.max(np.abs(matrix / scale - matrix.T / scale)) > 1e-12:
            raise ValueError("reduced mass must be symmetric positive definite")
        np.linalg.cholesky(matrix)
    return matrix


def _robot_port(scene, address, prepared):
    if hasattr(scene, "_participants"):
        participant = scene._participants[address]
        floating = bool(getattr(participant, "is_floating", False))
        asset = participant.asset
        mechanism = None if floating else participant.mechanism
    else:
        if address != scene._addr:
            raise ValueError("prepared owner differs from the single robot")
        floating, asset, mechanism = False, scene._asset, scene._mechanism
    full_n, reduced_n = len(prepared.physical.v), len(prepared.entry.v)
    full_v = _array(prepared.physical.v, (full_n,), "full robot velocity")
    velocity = _array(prepared.entry.v, (reduced_n,), "independent robot velocity")
    if floating:
        projection = getattr(asset, "velocity_projection", np.eye(full_n))
    else:
        projection = np.eye(full_n) if mechanism is None else mechanism.matrix
    projection = _array(projection, (full_n, reduced_n), "physical velocity projection")
    if not np.allclose(projection @ velocity, full_v, atol=1e-12, rtol=1e-12):
        raise ValueError("prepared full and independent robot velocities disagree")
    count = len(prepared.geometry.com_position)
    centers = _array(prepared.geometry.com_position, (count, 3), "physical COM positions")
    linear = _array(prepared.linear, (count, 3, full_n), "full linear COM Jacobian")
    angular = _array(prepared.angular, (count, 3, full_n), "full angular COM Jacobian")
    full_mass = _array(prepared.full_mass, (full_n, full_n), "full mass")
    bias = _array(prepared.full_bias, (full_n,), "full dynamics bias")
    mass = _mass(projection.T @ full_mass @ projection, reduced_n)
    response_mass = None
    step_mass = getattr(prepared, "full_step_mass", None)
    if step_mass is not None:
        step_mass = _array(step_mass, (full_n, full_n), "full step response metric")
        response_mass = _mass(projection.T @ step_mass @ projection, reduced_n)
    if floating:
        if reduced_n < 6 or full_n != count + 5:
            raise ValueError("floating physical root/joint dimensions disagree")
        motors = _array(prepared.effort, (reduced_n - 6,), "independent floating motor effort")
        force = _array(prepared.external.force, (count, 3), "external floating COM force")
        torque = _array(prepared.external.torque, (count, 3), "external floating COM torque")
        effort = _array(prepared.external.effort, (full_n - 6,), "external full joint effort")
        if np.any(effort):
            raise ValueError("floating external joint effort must remain zero")
        external = np.einsum("nki,nk->i", linear, force) + np.einsum("nki,nk->i", angular, torque)
        free_effort = np.r_[np.zeros(6), motors] + projection.T @ (external - bias)
    else:
        if full_n != count:
            raise ValueError("fixed-tree physical body/velocity counts disagree")
        effort = _array(prepared.effort, (reduced_n,), "bounded independent motor effort")
        free_effort = effort - projection.T @ bias
        if not np.allclose(mass, prepared.mass, atol=1e-12, rtol=1e-12):
            raise ValueError("prepared reduced mass differs from the actual projection")
    return OwnerPort(
        tuple(address),
        True,
        projection,
        mass,
        velocity,
        full_v,
        linear,
        angular,
        centers,
        _array(free_effort, (reduced_n,), "free effort"),
        response_mass,
    )


def build_ports(scene, prepared, external):
    """Copy only the already prepared physical response; never call a participant."""
    if not isinstance(prepared, Mapping) or set(prepared) != set(scene.assets):
        raise ValueError("prepared responses must cover exactly the scene robot owners")
    ports = {
        address: _robot_port(scene, address, prepared[address]) for address in sorted(prepared)
    }
    body, model = scene._body, scene._rigid_model
    count = model.body_count
    forces = _array(external.rigid.force, (count, 3), "external rigid force")
    torques = _array(external.rigid.torque, (count, 3), "external rigid torque")
    rotations = quaternion_matrix(body.orientation)
    for address, row in sorted(scene.rigid_rows.items()):
        if address in ports:
            raise ValueError("a material owner cannot be both a robot and a free rigid body")
        linear, angular = np.zeros((1, 3, 6)), np.zeros((1, 3, 6))
        linear[0, :, :3], angular[0, :, 3:] = np.eye(3), np.eye(3)
        full_v = np.r_[body.velocity[row], body.angular_velocity[row]]
        inverse_mass = model.inverse_mass[row]
        if inverse_mass:
            rotation, inertia = rotations[row], model.inertia_diagonal[row]
            mass = np.zeros((6, 6))
            mass[:3, :3] = np.eye(3) / inverse_mass
            mass[3:, 3:] = (rotation * inertia) @ rotation.T
            omega_body = rotation.T @ body.angular_velocity[row]
            gyro = rotation @ np.cross(omega_body, inertia * omega_body)
            free = np.r_[forces[row] + model.gravity / inverse_mass, torques[row] - gyro]
            projection, velocity = np.eye(6), full_v
        else:
            if np.any(full_v):
                raise ValueError("fixed rigid support cannot have a prescribed twist")
            mass, projection = np.empty((0, 0)), np.empty((6, 0))
            velocity, free = np.empty(0), np.empty(0)
        size = len(velocity)
        ports[address] = OwnerPort(
            tuple(address),
            False,
            _array(projection, (6, size), "rigid projection"),
            _mass(mass, size),
            _array(velocity, (size,), "rigid velocity"),
            _array(full_v, (6,), "rigid full velocity"),
            _array(linear, (1, 3, 6), "rigid linear Jacobian"),
            _array(angular, (1, 3, 6), "rigid angular Jacobian"),
            _array(body.position[row : row + 1], (1, 3), "rigid COM"),
            _array(free, (size,), "rigid free effort"),
        )
    return MappingProxyType(dict(sorted(ports.items())))


def owner_poses(scene, body, geometries):
    """Real body-origin frames, including moving root zero and fixed root minus one."""
    if not isinstance(geometries, Mapping) or set(geometries) != set(scene.assets):
        raise ValueError("material frames require every real robot geometry")
    adapter = scene._event_adapter
    rotations, poses = quaternion_matrix(body.orientation), {}
    for metadata in adapter._frames.values():
        owner = metadata.owner
        if owner in poses:
            continue
        if owner.kind == "rigid":
            row = scene.rigid_rows[owner.world, owner.body]
            position, rotation = body.position[row], rotations[row]
        elif owner.local_frame == "robot_root":
            position, rotation = adapter._root_frames[owner.world, owner.body]
        else:
            geometry = geometries[owner.world, owner.body]
            position, rotation = (
                geometry.position[owner.link_index],
                geometry.rotation[owner.link_index],
            )
        poses[owner] = OwnerPose(owner, position, rotation)
    return MappingProxyType(poses)


def relative_response(ports, bindings, measured, loads, rigid_rows, h):
    """Full station cross response, summing each independent physical owner once."""
    count = len(bindings)
    matrices = {
        address: np.zeros((3 * count, len(port.velocity))) for address, port in ports.items()
    }
    for i, binding in enumerate(bindings):
        for owner, point, sign in (
            (binding.first_owner, binding.application_point_first_m, -1.0),
            (binding.second_owner, binding.application_point_second_m, 1.0),
        ):
            address = owner.world, owner.body
            port = ports[address]
            matrices[address][3 * i : 3 * i + 3] += (
                sign * port.jacobian(owner, point) @ port.projection
            )
    world, free = np.zeros((3 * count, 3 * count)), np.zeros(3 * count)
    for address, port in ports.items():
        if port.robot:
            contact = measured[address].joint_effort_full
        else:
            row = rigid_rows[address]
            contact = np.r_[loads.inputs.rigid.force[row], loads.inputs.rigid.torque[row]]
        velocity = port.free_velocity(contact, h)
        rows = matrices[address]
        if len(velocity):
            world += rows @ np.linalg.solve(port.response_metric, rows.T)
            free += rows @ velocity
    return world, free.reshape(count, 3)
