"""Bound public replay facade for nonincident finite-thickness self response."""

from dataclasses import replace

import numpy as np

from ..constraints.cloth_self_contact import ClothSelfContactTopology
from ..constraints.cloth_self_response import SelfContactResponseLaw, SelfContactResponseLimits, SelfContactBudgets
from ..constraints.distance import DistanceConstraints
from ..core.model import ParticleModel
from ..core.scene import HistoryField, StepSettings, fingerprint
from ..runtime.program import ProgramCapabilities, own_diagnostics
from ..runtime.registry import StateRegistry


class ClothSelfContactProgram:
    """Borrow one base factory; own only replay/workspace lifetimes and receipts.

    This is S2-L's nonincident, constant-normal loaded domain. It grants no
    incident-material exemption or rotating loaded continuation certificate.
    """
    def __init__(self, scene, base_program, topology, *, law, materials, budgets, limits,
                 registry, inner_settings, history_prefix="self_contact"):
        scene.assert_unchanged()
        if not isinstance(topology, ClothSelfContactTopology):
            raise TypeError("explicit immutable self topology required")
        if len(scene.domains) != 1 or scene.domains[0].name != topology.domain or scene.domains[0].family != "particles":
            raise ValueError("S2-L requires one declared physical cloth base")
        model = scene.domains[0].model
        if not isinstance(model, ParticleModel) or model.inverse_mass.dtype != np.float64 or np.any(model.inverse_mass <= 0):
            raise ValueError("S2-L requires positive finite FP64 particle masses")
        if np.any(topology.triangles >= model.particle_count):
            raise ValueError("topology node outside declared base")
        for value, cls in ((law, SelfContactResponseLaw), (budgets, SelfContactBudgets), (limits, SelfContactResponseLimits), (registry, StateRegistry), (inner_settings, StepSettings)):
            if not isinstance(value, cls):
                raise TypeError(f"explicit {cls.__name__} required")
        if inner_settings.substeps != 1 or inner_settings.position_iterations != 512:
            raise ValueError("inner replay requires one substep and original 512 material iterations")
        if not isinstance(history_prefix, str) or not history_prefix or history_prefix.strip() != history_prefix:
            raise ValueError("invalid contact history prefix")
        if materials is not None:
            if not isinstance(materials, DistanceConstraints) or materials.rest_length.dtype != np.float64:
                raise TypeError("S2-L supports only None or bound FP64 DistanceConstraints; composite/v2 requires later typed integration")
            if np.any(materials.indices >= model.particle_count):
                raise ValueError("material node outside declared base")
            materials = replace(materials)
        if "terminal_point_impulses" not in base_program.capabilities.features or base_program.capabilities.provider_kind != "mathematical":
            raise ValueError("public mathematical terminal participant program required")
        if base_program.scene.identity != scene.identity or set(base_program.participants) != {topology.domain}:
            raise ValueError("base participant binding mismatch")
        signature = base_program.participants[topology.domain].problem_signature()
        if signature.get("law") != "cloth-xpbd-distance-predicted-velocity-v1" or not signature.get("record_material_path"):
            raise ValueError("recorded legacy cloth material path required")
        if fingerprint(signature.get("ordered_constraints")) != fingerprint(materials):
            raise ValueError("base material differs from declared self-contact material")
        if registry.get("particles") != base_program.registry.get("particles"):
            raise ValueError("self response and base registries differ")
        self.scene, self.base, self.topology = scene, base_program, topology
        self.model, self.domain, self.materials = model, topology.domain, materials
        self.law, self.budgets, self.limits = law, budgets, limits
        self.registry, self.inner_settings, self.history_prefix = registry.copy(), inner_settings, history_prefix
        if any(f.name == history_prefix or f.name.startswith(history_prefix+"/") for f in base_program.history_fields):
            raise ValueError("self history prefix collides with base history")
        count = limits.max_contacts*limits.max_anchors_per_pair
        self.own_fields = tuple(HistoryField(history_prefix+"/"+name, shape, dtype) for name, shape, dtype in (
            ("active", (count,), bool), ("face_ids", (count, 2), np.int64),
            ("node_ids", (count, 2, 3), np.int64), ("counts", (count, 2, 3), np.int64),
            ("offsets", (count, 2), np.float64), ("offset_bits", (count, 2), np.int64),
            ("generations", (count,), np.int64), ("phase_normals", (count, 2, 3), np.float64),
            ("phase_world", (count, 2, 3), np.float64), ("epoch", (), np.int64),
            ("window_positive", (2,), np.float64)))
        self._closed = False
        self._attempt_receipts = ()
        self._base_signature = fingerprint(base_program.problem_signature())
        self._signature = fingerprint(self.problem_signature())

    @property
    def capabilities(self):
        return ProgramCapabilities(self.base.capabilities.backend, {"float64"}, {"particles"},
                                   {"cloth_self_response_unique_ray_v1"}, history="explicit")

    @property
    def history_fields(self):
        return tuple(self.base.history_fields)+self.own_fields

    @property
    def last_attempt_receipts(self):
        """Immutable same-run observations; never reused as physical state/capacity."""
        return self._attempt_receipts

    def initial_history(self):
        return {field.name: np.zeros(field.shape, field.dtype) for field in self.own_fields}

    def problem_signature(self):
        return dict(law=self.law, scene=self.scene.identity, base=self.base.problem_signature(),
                    topology=self.topology, material=self.materials, budgets=self.budgets, limits=self.limits,
                    inner=self.inner_settings, history=tuple((f.name, f.shape, f.dtype.str) for f in self.history_fields))

    def create_workspace(self, scene, state, inputs, h, settings, context):
        from ._cloth_self_contact_workspace import ClothSelfContactWorkspace
        if self._closed:
            raise RuntimeError("self-contact program closed")
        self.scene.assert_unchanged()
        if scene.identity != self.scene.identity or fingerprint(self.base.problem_signature()) != self._base_signature or fingerprint(self.problem_signature()) != self._signature:
            raise ValueError("bound self-contact/base problem changed")
        if context.backend != self.capabilities.backend or context.dtype != np.float64:
            raise ValueError("self-contact context mismatch")
        if not np.isfinite(h) or h <= 0 or settings.tolerance <= 0:
            raise ValueError("positive step and tolerance required")
        for key in ("tolerance", "reference_length", "reference_speed"):
            if getattr(settings, key) != getattr(self.inner_settings, key):
                raise ValueError("inner scales/tolerance must match outer")
        self._attempt_receipts = ()
        try:
            return ClothSelfContactWorkspace(self, state, inputs, float(h), settings, context)
        except BaseException as error:
            self._attempt_receipts = (own_diagnostics(dict(kind="construction-failure", reason=str(error), original_state=state, original_inputs=inputs)),)
            raise

    def close(self):
        self._closed = True
