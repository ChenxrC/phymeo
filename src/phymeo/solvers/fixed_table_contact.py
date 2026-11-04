"""Reciprocal forces between moving round robot links and one fixed finite cuboid support."""

from dataclasses import dataclass

import numpy as np

from ..constraints.robot_contact import RobotContactMaterial
from ..dynamics.robot_dynamics import robot_com_kinematics
from ..dynamics.rotation import quaternion_matrix
from .box_link_contact import BoxLinkContacts
from .sphere_link_contact import SphereLinkLoads


@dataclass
class FixedTableLoads:
    """Robot-side world COM wrenches and the opposite wrench about the fixed table centre."""

    contacts: list
    _source: SphereLinkLoads

    @property
    def table_wrench(self):
        return self._source.payload_wrench

    @property
    def link_wrenches(self):
        return self._source.link_wrenches

    def joint_load(self, com_linear_jacobians, angular_jacobians):
        return self._source.joint_load(com_linear_jacobians, angular_jacobians)


class FixedTableContacts:
    """A stationary oriented box supporting moving link capsules or spheres."""

    def __init__(
        self, solids, *, center_m, half_extents_m, material, orientation_xyzw=(0.0, 0.0, 0.0, 1.0)
    ):
        self._registry = BoxLinkContacts(solids)
        if any(s.owner_index < 0 for s in self._registry.solids):
            raise ValueError("a fixed table needs moving link solids")
        self.center = np.array(center_m, dtype=float, copy=True)
        self.half = np.array(half_extents_m, dtype=float, copy=True)
        q = np.asarray(orientation_xyzw, dtype=float)
        if (
            self.center.shape != (3,)
            or self.half.shape != (3,)
            or not np.isfinite(self.center).all()
            or not np.isfinite(self.half).all()
            or np.any(self.half <= 0)
            or q.shape != (4,)
            or not np.isfinite(q).all()
            or abs(np.linalg.norm(q) - 1) > 1e-6
        ):
            raise ValueError(
                "expected finite table centre, positive half-extents and a unit XYZW pose"
            )
        if not isinstance(material, RobotContactMaterial):
            raise TypeError("expected a table RobotContactMaterial")
        self.rotation = quaternion_matrix(q[None])[0]
        self.center.setflags(write=False)
        self.half.setflags(write=False)
        self.rotation.setflags(write=False)
        self.material = material

    @classmethod
    def from_asset(cls, asset, *, exclude_keys=(), authored_moving_frames=False, **table):
        # Static roots have no generalized motion. The caller must explicitly exclude
        # any unsupported moving shapes or channels it already solves.
        excluded = set(exclude_keys)
        excluded.update(c.shape.key for c in asset.collisions if c.link_index < 0)
        source = BoxLinkContacts.from_asset(
            asset, exclude_keys=excluded, authored_moving_frames=authored_moving_frames
        )
        return cls(source.solids, **table)

    def evaluate(self, positions, rotations, centers_of_mass, twists):
        zero = np.zeros(3)
        raw = self._registry.evaluate(
            positions,
            rotations,
            centers_of_mass,
            twists,
            self.center,
            self.half,
            zero,
            zero,
            self.material,
            payload_rotation=self.rotation,
        )
        contacts = []
        for hit in raw.contacts:
            row = dict(hit)
            row["table_surface_m"] = row.pop("payload_surface_m")
            row["force_on_table_N"] = row.pop("force_on_payload_N")
            row["torque_on_table_Nm"] = row.pop("torque_on_payload_Nm")
            row["table_center_world_m"] = self.center.tolist()
            # Use the table-to-link normal and material velocity of the moving link.
            row["normal_world"] = (-np.asarray(row["normal_world"])).tolist()
            row["relative_velocity_m_s"] = (-np.asarray(row["relative_velocity_m_s"])).tolist()
            row["link_power_W"] = row["pair_power_W"]
            row["table_power_W"] = 0.0
            contacts.append(row)
        return FixedTableLoads(contacts, raw)

    def evaluate_robot(self, model, state):
        """Return the pre-step load and J.T virtual-work effort of a fixed-base robot state."""
        geo, linear, angular = robot_com_kinematics(model, state.q)
        load = self.evaluate(
            geo.position,
            geo.rotation,
            geo.com_position,
            np.column_stack([linear @ state.v, angular @ state.v]),
        )
        return load, load.joint_load(linear, angular)
