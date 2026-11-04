"""Compliant contact between moving material samples and one oriented rigid box."""

from dataclasses import dataclass

import numpy as np

from ..constraints.robot_contact import RobotContactMaterial


@dataclass(frozen=True)
class SurfaceLoads:
    """World forces on samples and the opposite resultant and moment about the box COM."""

    forces: np.ndarray
    normal_loads: np.ndarray
    depths: np.ndarray
    body_force: np.ndarray
    body_torque: np.ndarray


def sample_loads(depths, normals, relative_velocity, material):
    """Per-sample spring/dashpot normals and slip-speed-regularized Coulomb traction."""
    depths, normals, relative_velocity = (
        np.asarray(a, dtype=float) for a in (depths, normals, relative_velocity)
    )
    if (
        not isinstance(material, RobotContactMaterial)
        or depths.ndim != 1
        or normals.shape != (len(depths), 3)
        or relative_velocity.shape != normals.shape
        or not all(np.isfinite(a).all() for a in (depths, normals, relative_velocity))
        or not np.allclose(np.linalg.norm(normals[depths > 0], axis=1), 1.0, atol=1e-10, rtol=0.0)
    ):
        raise ValueError("invalid sample depths, active normals, velocities or material")
    vn = np.einsum("ij,ij->i", relative_velocity, normals)
    load = np.where(
        depths > 0.0, np.maximum(0.0, material.stiffness * depths - material.damping * vn), 0.0
    )
    tangent = relative_velocity - vn[:, None] * normals
    speed = np.maximum(np.linalg.norm(tangent, axis=1), material.slip_speed_m_s)
    force = load[:, None] * (normals - material.friction * tangent / speed[:, None])
    if not np.isfinite(force).all() or not np.isfinite(load).all():
        raise ValueError("nonfinite sample contact load")
    return load, force


def box_surface_samples(
    points, velocities, center, rotation, velocity, omega, half_extents, material
):
    """Evaluate one force per sample inside the box, at the same world point on both owners.

    Stiffness and damping belong to one sample. Outside points transmit zero force.
    A sample on the interior medial set has no unique exit normal and is rejected.
    The supplied rotation is body to world. This is an instantaneous sampled model.
    """
    if not isinstance(material, RobotContactMaterial):
        raise TypeError("expected RobotContactMaterial")
    points, velocities, center, rotation, velocity, omega, extents = (
        np.asarray(a, dtype=float)
        for a in (points, velocities, center, rotation, velocity, omega, half_extents)
    )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape != velocities.shape
        or any(a.shape != (3,) for a in (center, velocity, omega, extents))
        or rotation.shape != (3, 3)
        or np.any(extents <= 0)
        or not all(
            np.isfinite(a).all()
            for a in (points, velocities, center, rotation, velocity, omega, extents)
        )
    ):
        raise ValueError("invalid material samples or box state")
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-10)
        or np.linalg.det(rotation) <= 0
    ):
        raise ValueError("expected an orthonormal right-handed box frame")
    levers = points - center
    local = levers @ rotation
    distances = extents - np.abs(local)
    axes = distances.argmin(axis=1)
    raw = distances[np.arange(len(points)), axes]
    sides = np.sign(local[np.arange(len(points)), axes])
    if np.any((raw > 0) & ((np.sum(distances == raw[:, None], axis=1) > 1) | (sides == 0))):
        raise ValueError("interior box sample has ambiguous face normal")
    normals = sides[:, None] * rotation.T[axes]
    depths = np.maximum(raw, 0.0)
    rel = velocities - (velocity + np.cross(omega, levers))
    loads, forces = sample_loads(depths, normals, rel, material)
    return SurfaceLoads(
        forces, loads, depths, -forces.sum(axis=0), np.cross(levers, -forces).sum(axis=0)
    )
