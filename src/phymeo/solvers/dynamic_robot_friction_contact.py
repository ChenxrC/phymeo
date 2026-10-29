"""Owned bindings for the phasewise dynamic Coulomb law."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

import numpy as np

from ..constraints.robot_plane_collider import RobotPlaneCollider
from ..core.scene import HistoryField
from .dynamic_robot_normal_contact import DynamicRobotNormalContactProgram


class DynamicRobotFrictionContactProgram(DynamicRobotNormalContactProgram):
    """All-prismatic zero-thickness planes with native held/terminal friction.

    Friction belongs to this program, independently of the immutable geometry.
    The split law consumes held capacity only for initially closed rows.
    """

    def __init__(
        self,
        scene,
        base_program,
        colliders,
        *,
        friction_by_collider,
        materials,
        budgets,
        robot_domain="robot",
        registry=None,
        inner_settings=None,
        relaxation=1.0,
        history_prefix="friction_contact",
    ):
        colliders = tuple(colliders)
        if not colliders or any(not isinstance(c, RobotPlaneCollider) for c in colliders):
            raise ValueError("at least one RobotPlaneCollider is required")
        if not isinstance(friction_by_collider, Mapping):
            raise TypeError("friction_by_collider must be a mapping")
        if any(type(key) is not str for key in friction_by_collider):
            raise TypeError("friction collider keys must be builtin strings")
        if set(friction_by_collider) != {c.key for c in colliders}:
            raise ValueError("friction keys must exactly match collider keys")
        if any(
            type(mu) is not float or not np.isfinite(mu) or mu < 0
            for mu in friction_by_collider.values()
        ):
            raise ValueError("friction values must be finite nonnegative builtin floats")
        if any(c.radius_m != 0.0 for c in colliders):
            raise ValueError("split friction requires zero collider radius")
        super().__init__(
            scene,
            base_program,
            colliders,
            materials=materials,
            budgets=budgets,
            robot_domain=robot_domain,
            registry=registry,
            inner_settings=inner_settings,
            relaxation=relaxation,
            history_prefix=history_prefix,
        )
        self.friction_by_collider = MappingProxyType(dict(sorted(friction_by_collider.items())))
        count = len(self.row_keys)
        self._own_fields = (
            HistoryField(history_prefix + "/active", (count,), bool),
            HistoryField(history_prefix + "/initial_closed", (count,), bool),
            HistoryField(history_prefix + "/phase_world", (count, 2, 3), np.float64),
        )

    @property
    def capabilities(self):
        return replace(super().capabilities, features=frozenset({"dynamic_robot_friction_split"}))

    def problem_signature(self):
        result = super().problem_signature()
        result.update(
            law="dynamic-robot-friction-split-v1", friction_by_collider=self.friction_by_collider
        )
        return result

    def create_workspace(self, scene, state, inputs, h, settings, context):
        from ._dynamic_friction_workspace import _DynamicFrictionWorkspace

        if self._closed:
            raise RuntimeError("dynamic friction program is closed")
        self.scene.assert_unchanged()
        if (
            scene.identity != self.scene.identity
            or settings.tolerance <= 0
            or not np.isfinite(h)
            or h <= 0
        ):
            raise ValueError("bound scene and positive h/tolerance required")
        inner = (
            replace(settings, substeps=1, record_substeps=False)
            if self.inner_settings is None
            else self.inner_settings
        )
        if any(
            getattr(inner, key) != getattr(settings, key)
            for key in ("tolerance", "reference_length", "reference_speed")
        ):
            raise ValueError("inner tolerance and reference scales must match outer")
        return _DynamicFrictionWorkspace(self, state, inputs, h, settings, inner, context)
