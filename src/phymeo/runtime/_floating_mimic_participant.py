"""Pure HOST scene candidate for a physical floating tree with affine mimic joints.

Only state and bounded motor coordinates are reduced. Contact sampling retains
all full-coordinate Jacobians and actual body/material owners. Publication is
owned by MultiRobotScene; this adapter holds no accepted state or event history.
"""

from dataclasses import replace

import numpy as np

from ..core.validation import validate_dt
from ..dynamics.floating_mimic import FloatingMimicRobotAsset
from ..dynamics.floating_robot import FloatingRobotState
from ..solvers.floating_tree import FloatingRobotInputs
from ._floating_robot_participant import (
    FloatingRobotCommand,
    FloatingRobotObservation,
    FloatingRobotParticipant,
    FloatingRobotPreparation,
    FloatingRobotRegistration,
    _project,
    _resolve_floating_drive,
)
from .robot_payload_scene import _readonly, _vector
from .robot_scene import _asset_snapshot


class FloatingMimicParticipant:
    """Independent controls and full physical contacts in one scene transaction.

    velocity_count is 6+n, not 6+k. The geometry adapter below uses the same
    physical owner and allocates its collision keys once. Its zero controller
    metadata is never used to prepare or integrate a physical state.
    """

    is_floating = True

    def __init__(self, address, registration, occupied):
        if not isinstance(registration, FloatingRobotRegistration) or not isinstance(
            registration.asset, FloatingMimicRobotAsset
        ):
            raise TypeError("expected a floating mimic registration")
        if registration.pads:
            raise ValueError("floating robot pads are not implemented")
        self.address, self.spec = address, _asset_snapshot(registration)
        self.asset, self.tree = self.spec.asset, self.spec.asset.tree
        self.body_count = self.tree.model.body_count
        self.velocity_count = 6 + len(self.tree.joint_names)
        self.initial = FloatingRobotState(self.spec.q, self.spec.v)
        full = self.full(self.initial)
        count = self.body_count - 1
        geometry_registration = replace(
            self.spec,
            asset=self.tree,
            q=_readonly(full.q),
            v=_readonly(full.v),
            kp=_readonly(np.zeros(count)),
            kd=_readonly(np.zeros(count)),
            effort_limit=_readonly(np.zeros(count)),
        )
        self._geometry_adapter = FloatingRobotParticipant(address, geometry_registration, occupied)
        for name in ("contact_view", "key_map", "names", "named_keys", "shapes"):
            setattr(self, name, getattr(self._geometry_adapter, name))

    @staticmethod
    def copy_state(state):
        return FloatingRobotState(state.q, state.v)

    def validate(self, state):
        self.asset.full_state(state)

    def full(self, state):
        return self.asset.full_state(state)

    def terms(self, state):
        return self._geometry_adapter.terms(self.full(state))

    def kinematics(self, state):
        """Expand affine coordinates once for actual candidate body frames."""
        return self._geometry_adapter.kinematics(self.full(state))

    def poses(self, state):
        return self._geometry_adapter.poses(self.full(state))

    def _observation(
        self,
        state,
        full,
        entry,
        effort,
        contact,
        wrenches,
        external,
        applied,
        acceleration,
        *,
        command=None,
        requested=None,
        receipt=None,
    ):
        return FloatingRobotObservation(
            q=_readonly(state.q),
            v=_readonly(state.v),
            full_q=_readonly(full.q),
            full_v=_readonly(full.v),
            entry_q=_readonly(entry.q),
            entry_v=_readonly(entry.v),
            drive_effort=_readonly(effort),
            contact_effort=_readonly(self.asset.velocity_projection.T @ contact),
            contact_effort_full=_readonly(contact),
            link_wrenches_world=_readonly(wrenches),
            fixed_root_contact_wrench=_readonly(np.zeros(6)),
            command=_asset_snapshot(command),
            requested_command=_asset_snapshot(requested),
            external=_asset_snapshot(external),
            applied_effort_full=_readonly(applied),
            acceleration_full=_readonly(acceleration),
            mimic_loads=_asset_snapshot(receipt),
        )

    def initial_observation(self):
        zero = np.zeros(self.velocity_count)
        return self._observation(
            self.initial,
            self.full(self.initial),
            self.initial,
            np.zeros(len(self.asset.joint_names)),
            zero,
            np.zeros((self.body_count, 6)),
            FloatingRobotInputs.zeros(self.body_count - 1),
            zero,
            zero,
        )

    def prepare(self, state, command, dt):
        """Resolve existing bounded PD in the independent root-plus-joint basis.

        Full analytic M/b and declared COM loads belong to this same entry.
        Contact is sampled afterwards by the scene, not added to feedforward.
        """
        if not isinstance(command, FloatingRobotCommand):
            raise TypeError("floating mimic action must be a FloatingRobotCommand")
        h = validate_dt(dt, np.float64)
        entry, command = self.copy_state(state), _asset_snapshot(command)
        physical, geometry, linear, angular, mass, bias = self.terms(entry)
        count = self.body_count - 1
        external = (
            FloatingRobotInputs.zeros(count) if command.external is None else command.external
        )
        if (
            not isinstance(external, FloatingRobotInputs)
            or external.force.shape != (self.body_count, 3)
            or external.effort.shape != (count,)
            or np.any(external.effort)
        ):
            raise ValueError("external COM loads must cover the full tree with zero effort")
        external = _asset_snapshot(external)
        projection = self.asset.velocity_projection
        reduced_mass = projection.T @ mass @ projection
        reduced_bias = projection.T @ bias
        reduced_external = projection.T @ _project(linear, angular, external.force, external.torque)
        if not all(
            np.isfinite(value).all() for value in (reduced_mass, reduced_bias, reduced_external)
        ):
            raise ValueError("floating mimic drive projection exceeds finite range")
        effort, resolved = _resolve_floating_drive(
            self.spec,
            entry,
            command,
            h,
            reduced_mass,
            reduced_bias,
            reduced_external,
            external,
        )
        return FloatingRobotPreparation(
            entry,
            physical,
            geometry,
            linear,
            angular,
            mass,
            bias,
            _readonly(effort),
            external,
            resolved,
            command,
        )

    def integrate(self, prepared, measured, dt):
        """Apply full COM loads once and return one constrained candidate.

        The scene's contact_effort_full is collision-only. The mechanism receipt
        retains its existing meaning: contact_effort_full includes all external
        non-motor loads, including separately declared COM forces and torques.
        """
        h = validate_dt(dt, np.float64)
        contact = _vector(
            measured.joint_effort_full, self.velocity_count, "floating mimic contact effort"
        )
        wrenches = np.asarray(measured.link_wrenches_world)
        fixed_root = np.asarray(measured.fixed_root_contact_wrench)
        if (
            wrenches.shape != (self.body_count, 6)
            or wrenches.dtype.kind not in "fiu"
            or not np.isfinite(wrenches).all()
            or fixed_root.shape != (6,)
            or fixed_root.dtype.kind not in "fiu"
            or not np.isfinite(fixed_root).all()
            or np.any(fixed_root)
        ):
            raise ValueError("floating contact needs each physical COM wrench and no fixed root")
        projected = _project(prepared.linear, prepared.angular, wrenches[:, :3], wrenches[:, 3:])
        if not np.isfinite(projected).all() or not np.allclose(
            contact, projected, rtol=2e-10, atol=2e-10
        ):
            raise ValueError("full contact effort does not match its actual COM wrenches")
        external = FloatingRobotInputs(
            np.zeros(self.body_count - 1),
            prepared.external.force + wrenches[:, :3],
            prepared.external.torque + wrenches[:, 3:],
        )
        # The read-only full-tree adapter is never stepped. This is the only
        # physical advance: constrained solve followed by the existing one drift.
        result = self.asset.step(prepared.entry, effort=prepared.effort, dt_s=h, external=external)
        self.validate(result.state)
        applied = result.loads.actuator_effort_full + result.loads.contact_effort_full
        if not np.isfinite(applied).all():
            raise ValueError("floating mimic applied load exceeds finite range")
        observation = self._observation(
            result.state,
            result.full_state,
            prepared.entry,
            prepared.effort,
            contact,
            wrenches,
            prepared.external,
            applied,
            result.loads.acceleration_full,
            command=prepared.resolved,
            requested=prepared.requested,
            receipt=result.loads,
        )
        return result.state, observation
