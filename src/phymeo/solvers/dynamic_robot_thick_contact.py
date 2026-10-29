"""Owned finite-thickness contact binding; numerical work lives in its workspace."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

import numpy as np

from ..constraints.distance import DistanceConstraints
from ..constraints.robot_contact_patch import RobotContactPatch
from ..constraints.triangle_prism_surface import TrianglePrismSurface
from ..core.model import ParticleModel
from ..core.scene import HistoryField, _name
from ..geometry.prism_patch_manifold import validate_prismatic
from ..geometry.robot_patch_sweep import _positive_integer
from .initial_robot_material_contact import MaterialContactBudgets
from .robot_plane_contact import RobotPlaneContactProgram


class DynamicRobotThickContactProgram(RobotPlaneContactProgram):
    """All-prismatic, positive triangle thickness, finite two-sided patch friction."""

    def __init__(
        self,
        scene,
        base_program,
        surfaces,
        colliders,
        *,
        friction_by_collider,
        materials,
        budgets,
        robot_domain="robot",
        registry=None,
        inner_settings=None,
        relaxation=1.0,
        max_anchors_per_pair=64,
        max_manifold_rebuilds=8,
        max_ccd_intervals=4096,
        max_ccd_depth=32,
        history_prefix="thick_contact",
    ):
        super().__init__(
            scene,
            base_program,
            (),
            robot_domain=robot_domain,
            registry=registry,
            inner_settings=inner_settings,
            relaxation=relaxation,
        )
        validate_prismatic(self.model)
        if "terminal_point_impulses" not in self.base.capabilities.features:
            raise ValueError("base lacks terminal_point_impulses capability")
        for name, value in (
            ("max_anchors_per_pair", max_anchors_per_pair),
            ("max_manifold_rebuilds", max_manifold_rebuilds),
            ("max_ccd_intervals", max_ccd_intervals),
            ("max_ccd_depth", max_ccd_depth),
        ):
            setattr(self, name, _positive_integer(value, name))
        surfaces, colliders = tuple(surfaces), tuple(colliders)
        if not surfaces or any(not isinstance(s, TrianglePrismSurface) for s in surfaces):
            raise ValueError("nonempty TrianglePrismSurface required")
        if not colliders or any(not isinstance(c, RobotContactPatch) for c in colliders):
            raise ValueError("nonempty RobotContactPatch required")
        if len({s.key for s in surfaces}) != len(surfaces) or len(
            {c.key for c in colliders}
        ) != len(colliders):
            raise ValueError("surface and patch keys must be unique")
        if any(c.link_index >= self.model.dof_count for c in colliders):
            raise ValueError("patch link out of range")
        if len({(s.domain, tuple(sorted(s.node_indices))) for s in surfaces}) != len(surfaces):
            raise ValueError("duplicate unoriented triangle")
        if not isinstance(friction_by_collider, Mapping) or set(friction_by_collider) != {
            c.key for c in colliders
        }:
            raise ValueError("friction must exactly cover colliders")
        if any(
            type(v) is not float or not np.isfinite(v) or v < 0
            for v in friction_by_collider.values()
        ):
            raise ValueError("friction must be finite nonnegative builtin float")
        if not isinstance(materials, Mapping) or set(materials) != {s.domain for s in surfaces}:
            raise ValueError("materials must exactly cover surface domains")
        bindings = {d.name: d for d in scene.domains}
        owned = {}
        for name, material in sorted(materials.items()):
            if name not in bindings or bindings[name].family != "particles":
                raise ValueError("surface requires particle domain")
            model = bindings[name].model
            if (
                not isinstance(model, ParticleModel)
                or model.inverse_mass.dtype != np.float64
                or np.any(model.inverse_mass <= 0)
            ):
                raise ValueError("surface nodes must be dynamic FP64")
            if any(
                max(s.node_indices) >= model.particle_count for s in surfaces if s.domain == name
            ):
                raise ValueError("surface index outside domain")
            family = self.registry.get("particles")
            if (
                not {"position", "velocity"} <= set(family.state_fields)
                or "force" not in family.input_fields
            ):
                raise ValueError("particle family fields missing")
            if material is not None:
                if (
                    not isinstance(material, DistanceConstraints)
                    or material.rest_length.dtype != np.float64
                ):
                    raise TypeError("materials require FP64 DistanceConstraints or None")
                if np.any(material.indices >= model.particle_count):
                    raise ValueError("material index outside domain")
                material = replace(material)
            owned[name] = material
        if not isinstance(budgets, MaterialContactBudgets):
            raise TypeError("explicit MaterialContactBudgets required")
        self.materials, self.budgets = MappingProxyType(owned), replace(budgets)
        self.surfaces = tuple(sorted((replace(s) for s in surfaces), key=lambda s: s.key))
        self.colliders = tuple(sorted((replace(c) for c in colliders), key=lambda c: c.key))
        self.friction_by_collider = MappingProxyType(dict(sorted(friction_by_collider.items())))
        self.pairs = tuple((s, c) for s in self.surfaces for c in self.colliders)
        self.pair_keys = tuple((s.key, c.key) for s, c in self.pairs)
        _name(history_prefix, "history prefix")
        self.history_prefix = history_prefix
        k, r = len(self.pairs), self.max_anchors_per_pair
        self._own_fields = tuple(
            HistoryField(history_prefix + "/" + name, shape, dtype)
            for name, shape, dtype in (
                ("pair_active", (k,), bool),
                ("anchor_active", (k, r), bool),
                ("anchor_material", (k, r, 4), np.float64),
                ("phase_world", (k, r, 2, 3), np.float64),
            )
        )
        if any(
            f.name == history_prefix or f.name.startswith(history_prefix + "/")
            for f in base_program.history_fields
        ):
            raise ValueError("history prefix conflicts with base")

    @property
    def capabilities(self):
        return replace(
            self.base.capabilities,
            history="explicit",
            dtypes=frozenset({"float64"}),
            features=frozenset({"dynamic_robot_thick_split"}),
        )

    @property
    def history_fields(self):
        return tuple(self.base.history_fields) + self._own_fields

    def initial_history(self):
        return {f.name: np.zeros(f.shape, f.dtype) for f in self._own_fields}

    def problem_signature(self):
        return dict(
            super().problem_signature(),
            thick_law="owned-material-two-phase-v1",
            surfaces=self.surfaces,
            colliders=self.colliders,
            friction=self.friction_by_collider,
            materials=self.materials,
            budgets=self.budgets,
            max_anchors_per_pair=self.max_anchors_per_pair,
            max_manifold_rebuilds=self.max_manifold_rebuilds,
            max_ccd_intervals=self.max_ccd_intervals,
            max_ccd_depth=self.max_ccd_depth,
            history=tuple((f.name, f.shape, f.dtype.str) for f in self.history_fields),
        )

    def create_workspace(self, scene, state, inputs, h, settings, context):
        from ._thick_contact_workspace import ThickContactWorkspace

        if self._closed:
            raise RuntimeError("thick contact program closed")
        self.scene.assert_unchanged()
        if (
            scene.identity != self.scene.identity
            or not np.isfinite(h)
            or h <= 0
            or settings.tolerance <= 0
        ):
            raise ValueError("bound scene and positive h/tolerance required")
        inner = (
            replace(settings, substeps=1, record_substeps=False)
            if self.inner_settings is None
            else self.inner_settings
        )
        if any(
            getattr(inner, k) != getattr(settings, k)
            for k in ("tolerance", "reference_length", "reference_speed")
        ):
            raise ValueError("inner tolerance and scales must match outer")
        return ThickContactWorkspace(self, state, inputs, h, settings, inner, context)
