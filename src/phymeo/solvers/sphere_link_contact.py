"""Reciprocal world-space compliant contact of a free sphere with rigid robot links."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from ..constraints.robot_contact import RobotContactMaterial
from ..dynamics.rotation import quaternion_matrix
from .sampled_box_contact import sample_loads


@dataclass(frozen=True, eq=False)
class LinkSolid:
    """A body-local sphere, finite capsule (two centres), or eight product-ordered box corners.

    owner_index=-1 denotes a fixed support; its vertices are world coordinates.
    """

    shape: str
    link: str
    owner_index: int
    kind: str
    vertices: tuple
    radius: float = 0.0

    def __post_init__(self):
        points = np.array(self.vertices, dtype=float, copy=True)
        count = {"sphere": 1, "capsule": 2, "box": 8}.get(self.kind)
        if (
            count is None
            or points.shape != (count, 3)
            or not np.isfinite(points).all()
            or not isinstance(self.owner_index, (int, np.integer))
            or self.owner_index < -1
            or not self.shape
            or not self.link
            or not np.isfinite(self.radius)
        ):
            raise ValueError(
                f"{self.shape}: unsupported or invalid {self.kind} structural geometry"
            )
        if self.kind == "box":
            edges = points[[4, 2, 1]] - points[0]
            lengths = np.linalg.norm(edges, axis=1)
            if self.radius != 0 or np.any(lengths <= 0):
                raise ValueError(f"{self.shape}: invalid box edge length or radius")
            axes = edges / lengths[:, None]
            expected = points[0] + np.array(list(product((0, 1), repeat=3))) @ edges
            if not (
                np.allclose(axes @ axes.T, np.eye(3), atol=1e-9, rtol=0)
                and np.allclose(points, expected, atol=1e-11 * max(1.0, lengths.max()), rtol=0)
            ):
                raise ValueError(f"{self.shape}: hull is not a product-ordered orthogonal box")
            object.__setattr__(self, "_axes", axes)
            object.__setattr__(self, "_half", 0.5 * lengths)
            object.__setattr__(self, "_center", points.mean(axis=0))
        elif self.radius <= 0 or (self.kind == "capsule" and np.all(points[0] == points[1])):
            raise ValueError(f"{self.shape}: invalid {self.kind} radius or endpoints")
        points.setflags(write=False)
        object.__setattr__(self, "vertices", points)
        # Exact solids are always enclosed inside these local axis bounds.
        object.__setattr__(self, "_low", points.min(axis=0) - self.radius)
        object.__setattr__(self, "_high", points.max(axis=0) + self.radius)

    def witness(self, center, sphere_radius):
        if np.any(center + sphere_radius <= self._low) or np.any(
            center - sphere_radius >= self._high
        ):
            return None
        if self.kind == "box":
            local = self._axes @ (center - self._center)
            clamped = np.clip(local, -self._half, self._half)
            delta = local - clamped
            distance = np.linalg.norm(delta)
            if distance > 0:
                normal = self._axes.T @ (delta / distance)
                feature = ("face", "edge", "corner")[np.count_nonzero(delta) - 1]
            else:
                margins = self._half - np.abs(local)
                axis = int(np.argmin(margins))
                if (
                    np.count_nonzero(np.isclose(margins, margins[axis], rtol=0, atol=1e-13)) != 1
                    or local[axis] == 0
                ):
                    raise ValueError(f"{self.shape}: no unique box separation normal")
                normal = self._axes[axis] * np.sign(local[axis])
                distance = -margins[axis]
                feature = "face"
            surface = center - distance * normal
        else:
            nearest = self.vertices[0]
            feature = "sphere"
            if self.kind == "capsule":
                line = self.vertices[1] - nearest
                u = np.clip((center - nearest) @ line / (line @ line), 0.0, 1.0)
                nearest = nearest + u * line
                feature = "end0" if u == 0 else "end1" if u == 1 else "shaft"
            delta = center - nearest
            length = np.linalg.norm(delta)
            # Endpoint interpolation may leave roundoff even when the centre lies on the spine.
            scale = max(self.radius, np.max(np.abs(self.vertices)), np.max(np.abs(center)))
            if length <= 32 * np.finfo(float).eps * scale:
                raise ValueError(
                    f"{self.shape}: coincident sphere or capsule spine has no unique normal"
                )
            normal = delta / length
            surface = nearest + self.radius * normal
            distance = length - self.radius
        gap = distance - sphere_radius
        return None if gap >= 0 else (gap, normal, surface, feature)


@dataclass
class SphereLinkLoads:
    contacts: list
    payload_wrench: np.ndarray
    link_wrenches: np.ndarray

    def joint_load(self, com_linear_jacobians, angular_jacobians):
        return np.einsum("nki,nk->i", com_linear_jacobians, self.link_wrenches[:, :3]) + np.einsum(
            "nki,nk->i", angular_jacobians, self.link_wrenches[:, 3:]
        )


class SphereLinkContacts:
    def __init__(self, solids):
        self.solids = tuple(sorted(solids, key=lambda s: (s.owner_index, s.shape, s.link)))
        names = [(s.owner_index, s.link, s.shape) for s in self.solids]
        if len(set(names)) != len(names):
            raise ValueError("duplicate structural shape on a link")

    @classmethod
    def from_asset(cls, asset, *, exclude_keys=(), authored_moving_frames=False):
        """Compile supported actual colliders. Authored moving frames suit URDF-native body states."""
        excluded = set(exclude_keys)
        solids = []
        convert = getattr(asset, "authored_to_model_points", None)
        for col in asset.collisions:
            if col.shape.key in excluded:
                continue
            owner = col.link_index
            points = np.asarray(col.local_vertices)
            if convert is not None:
                points = convert(col.link_name, points)
            elif owner < 0:
                root_r = quaternion_matrix(np.array([asset.root_xyzw]))[0]
                points = np.array(asset.root_xyz) + points @ root_r.T
            if authored_moving_frames and owner >= 0 and convert is not None:
                # The origin and basis of the moving authored link in the model-local frame.
                frame = convert(asset.link_names[owner], np.vstack([np.zeros(3), np.eye(3)]))
                points = (points - frame[0]) @ (frame[1:] - frame[0]).T
            kind = "box" if col.shape.kind == "hull" else col.shape.kind
            solids.append(LinkSolid(col.name, col.link_name, owner, kind, points, col.shape.radius))
        return cls(solids)

    def evaluate(
        self,
        positions,
        rotations,
        centers_of_mass,
        twists,
        center,
        radius,
        velocity,
        omega,
        material,
        *,
        fixed_reference_m=(0.0, 0.0, 0.0),
    ):
        """Evaluate one witness per penetrating registered solid at a single pre-step state.

        twists hold COM linear then angular velocity. The common action point is the midpoint
        of the two material witnesses. This preserves net moment and reciprocal power.
        """
        positions, rotations, centers_of_mass, twists, center, velocity, omega, fixed = (
            np.asarray(a, dtype=float)
            for a in (
                positions,
                rotations,
                centers_of_mass,
                twists,
                center,
                velocity,
                omega,
                fixed_reference_m,
            )
        )
        if not isinstance(material, RobotContactMaterial):
            raise TypeError("expected a structural RobotContactMaterial")
        n = len(positions)
        if (
            positions.shape != (n, 3)
            or rotations.shape != (n, 3, 3)
            or centers_of_mass.shape != (n, 3)
            or twists.shape != (n, 6)
            or any(a.shape != (3,) for a in (center, velocity, omega, fixed))
            or not np.isfinite(radius)
            or radius <= 0
            or not all(
                np.isfinite(a).all()
                for a in (
                    positions,
                    rotations,
                    centers_of_mass,
                    twists,
                    center,
                    velocity,
                    omega,
                    fixed,
                )
            )
            or any(s.owner_index >= n for s in self.solids)
        ):
            raise ValueError("invalid structural body poses, sphere or owner indices")
        if not np.allclose(
            rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-9, rtol=0
        ) or np.any(np.linalg.det(rotations) <= 0):
            raise ValueError("structural body orientation must be proper orthogonal")
        contacts = []
        body_load = np.zeros((n, 6))
        ball = np.zeros(6)
        for solid in self.solids:
            i = solid.owner_index
            local = (center - positions[i]) @ rotations[i] if i >= 0 else center
            w = solid.witness(local, radius)
            if w is None:
                continue
            gap, normal, surface, feature = w
            if i >= 0:
                normal = rotations[i] @ normal
                surface = positions[i] + rotations[i] @ surface
            skin = center - radius * normal
            point = 0.5 * (skin + surface)
            reference = centers_of_mass[i] if i >= 0 else fixed
            link_velocity = (
                twists[i, :3] + np.cross(twists[i, 3:], point - reference)
                if i >= 0
                else np.zeros(3)
            )
            relative = velocity + np.cross(omega, point - center) - link_velocity
            normal_load, force = sample_loads([-gap], normal[None], relative[None], material)
            force = force[0]
            sphere_torque = np.cross(point - center, force)
            link_torque = np.cross(point - reference, -force)
            ball += np.r_[force, sphere_torque]
            if i >= 0:
                body_load[i] += np.r_[-force, link_torque]
            contacts.append(
                dict(
                    shape=solid.shape,
                    link=solid.link,
                    owner_index=int(i),
                    feature=feature,
                    gap_m=float(gap),
                    normal_world=normal.tolist(),
                    link_surface_m=surface.tolist(),
                    sphere_surface_m=skin.tolist(),
                    point_world_m=point.tolist(),
                    link_com_world_m=reference.tolist(),
                    relative_velocity_m_s=relative.tolist(),
                    force_on_payload_N=force.tolist(),
                    force_on_link_N=(-force).tolist(),
                    torque_on_payload_Nm=sphere_torque.tolist(),
                    torque_on_link_Nm=link_torque.tolist(),
                    normal_load_N=float(normal_load[0]),
                    pair_power_W=float(force @ relative),
                )
            )
        return SphereLinkLoads(contacts, ball, body_load)
