"""Uniform free-body mass properties and sampled contact against finite pads and tables."""

from itertools import product
from numbers import Real

import numpy as np

from .sampled_box_contact import SurfaceLoads, box_surface_samples, sample_loads


def box_corners(half_extents):
    return np.array(list(product(*[(-h, h) for h in half_extents])))


def box_tabletop_load(vertices, center, rotation, twist, table_center, table_half, law):
    """Apply the original eight world-corner loads within the finite top-face domain."""
    arms = vertices @ rotation.T
    world = center + arms
    if np.any(np.abs(world[:, :2] - table_center[:2]) > table_half[:2]):
        raise ValueError("payload corner left the finite tabletop footprint")
    velocity = twist[:3] + np.cross(twist[3:], arms)
    normal = np.tile([0.0, 0.0, 1.0], (len(world), 1))
    loads, forces = sample_loads(
        table_center[2] + table_half[2] - world[:, 2], normal, velocity, law
    )
    return forces, loads, arms


class SampledPayload:
    """A uniform solid box or sphere. Lengths are metres; spatial quantities are in world axes."""

    def __init__(self, box_size_m=None, *, payload_geometry=None):
        self.radius = None
        if payload_geometry is None:
            size = (0.05, 0.05, 0.05) if box_size_m is None else box_size_m
        else:
            if box_size_m is not None:
                raise ValueError("choose either box_size_m or payload_geometry")
            if not isinstance(payload_geometry, dict):
                raise ValueError("payload_geometry must describe a sphere or box")
            kind = payload_geometry.get("kind")
            if kind == "box" and set(payload_geometry) == {"kind", "full_size_m"}:
                size = payload_geometry["full_size_m"]
            elif kind == "sphere" and set(payload_geometry) == {"kind", "radius_m"}:
                radius = payload_geometry["radius_m"]
                if (
                    isinstance(radius, (bool, np.bool_))
                    or not isinstance(radius, Real)
                    or not np.isfinite(radius)
                    or radius <= 0
                ):
                    raise ValueError("sphere radius_m must be finite and positive")
                self.radius = float(radius)
                size = (2 * self.radius,) * 3
            else:
                raise ValueError("unsupported payload_geometry kind or dimensions")
        self.full_size = np.asarray(size, dtype=float)
        if (
            self.full_size.shape != (3,)
            or not np.isfinite(self.full_size).all()
            or np.any(self.full_size <= 0)
        ):
            raise ValueError(
                "box_size_m or payload dimensions must contain three finite positive full sizes"
            )
        self.half = self.full_size / 2
        self.vertices = box_corners(self.half) if self.radius is None else None

    @classmethod
    def from_manifest(cls, spec):
        descriptor = spec.get("payload_geometry")
        payload = (
            cls(spec["full_size_m"]) if descriptor is None else cls(payload_geometry=descriptor)
        )
        if not np.array_equal(payload.full_size, spec["full_size_m"]):
            raise ValueError("payload_geometry is inconsistent with full_size_m")
        return payload

    def descriptor(self):
        if self.radius is None:
            return {"kind": "box", "full_size_m": self.full_size.tolist()}
        return {"kind": "sphere", "radius_m": self.radius}

    def inertia(self, mass):
        if self.radius is not None:
            return np.full(3, (2 / 5) * mass * self.radius**2)
        x, y, z = self.full_size
        return mass / 12 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])

    def fits_on_table(self, center, rotation, table_center, table_half):
        if self.radius is not None:
            return np.all(np.abs(center[:2] - table_center[:2]) + self.radius <= table_half[:2])
        world = center + self.vertices @ rotation.T
        return np.all(np.abs(world[:, :2] - table_center[:2]) <= table_half[:2])

    def low_z(self, center, rotation):
        if self.radius is not None:
            return center[2] - self.radius
        return center[2] + (self.vertices @ rotation.T)[:, 2].min()

    def pad_loads(self, points, speeds, center, rotation, velocity, omega, law):
        if self.radius is None:
            return box_surface_samples(
                points, speeds, center, rotation, velocity, omega, self.half, law
            )
        levers = np.asarray(points) - center
        distances = np.linalg.norm(levers, axis=1)
        if np.any(distances == 0):
            raise ValueError("finger sample at sphere center has no unique normal")
        normals = levers / distances[:, None]
        depths = np.maximum(self.radius - distances, 0.0)
        relative = speeds - (velocity + np.cross(omega, levers))
        loads, forces = sample_loads(depths, normals, relative, law)
        return SurfaceLoads(
            forces, loads, depths, -forces.sum(axis=0), np.cross(levers, -forces).sum(axis=0)
        )

    def table_loads(self, center, rotation, velocity, omega, table_center, table_half, law):
        if self.radius is None:
            return box_tabletop_load(
                self.vertices,
                center,
                rotation,
                np.r_[velocity, omega],
                table_center,
                table_half,
                law,
            )
        # The outward normal is radial from the closest point on the finite table solid.
        local = center - table_center
        delta = local - np.clip(local, -table_half, table_half)
        distance = np.linalg.norm(delta)
        if distance > 0:
            normal = delta / distance
        else:
            gaps = table_half - np.abs(local)
            axis = int(np.argmin(gaps))
            if np.count_nonzero(gaps == gaps[axis]) > 1 or local[axis] == 0:
                raise ValueError("sphere center has no unique finite-table separation normal")
            normal = np.zeros(3)
            normal[axis] = np.sign(local[axis])
            distance = -gaps[axis]
        arm = -self.radius * normal
        relative = velocity + np.cross(omega, arm)
        loads, forces = sample_loads(
            [self.radius - distance], normal[None, :], relative[None, :], law
        )
        return forces, loads, arm[None, :]
