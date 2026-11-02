"""Registered material-face forces and continuous checks for one actual elastic sheet."""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from itertools import combinations

import numpy as np

from ..backends.tyx_normal_layer import TyxNormalLayer
from ..constraints.collision_shapes import CollisionKey, CollisionModel, CollisionShape
from ..geometry.collision_pipeline import CollisionWorkspace
from .collision_loads import CollisionLoadSystem, NormalLayer, ParticleAttachment


class ClothSurfaceContact:
    def __init__(self, surfaces, layer, operator, geometry_mode):
        if geometry_mode not in ("host", "interpret", "cuda"):
            raise ValueError("unknown cloth self-contact geometry backend")
        if operator is not None and not isinstance(operator, TyxNormalLayer):
            raise TypeError("cloth self-contact force operator must be a TyxNormalLayer")
        self.geometry_mode = geometry_mode
        self.force_mode = "host" if operator is None else operator.mode
        self.names = tuple(s.key for s in surfaces)
        faces = tuple(tuple(map(int, s.node_indices)) for s in surfaces)
        if len({frozenset(face) for face in faces}) != len(faces):
            raise ValueError("duplicate material face in cloth self-contact")
        pairs = list(combinations(range(len(faces)), 2))
        keys = [tuple(sorted((self.names[i], self.names[j]))) for i, j in pairs]
        if isinstance(layer, NormalLayer):
            laws = dict.fromkeys(keys, layer)
        elif isinstance(layer, Mapping):
            laws = {}
            for names, law in layer.items():
                if (
                    not isinstance(names, tuple)
                    or len(names) != 2
                    or not all(isinstance(n, str) for n in names)
                ):
                    raise ValueError("self-contact calibrations need pairs of material face names")
                key = tuple(sorted(names))
                if key in laws or not isinstance(law, NormalLayer):
                    raise ValueError("invalid or duplicate cloth self-contact law")
                laws[key] = law
            if set(laws) != set(keys):
                raise ValueError("cloth self-contact requires every material face pair")
        else:
            raise TypeError("expected a NormalLayer or a complete face-pair calibration")
        shapes = tuple(
            CollisionShape(CollisionKey(0, 0, i), "triangle", face) for i, face in enumerate(faces)
        )
        model = CollisionModel(shapes)
        bindings = {s.key: ParticleAttachment(s.vertex_ids) for s in shapes}
        measured = {}
        for (i, j), key in zip(pairs, keys, strict=True):
            law = laws[key]
            if law.friction:
                raise ValueError("cloth self-contact currently requires frictionless normal layers")
            measured[shapes[i].key, shapes[j].key] = replace(
                law,
                width_m=law.width_m + surfaces[i].half_thickness_m + surfaces[j].half_thickness_m,
            )
        self.system = CollisionLoadSystem(model, bindings, measured, layer_operator=operator)
        self.calibrations = tuple((key, laws[key]) for key in keys)
        self.signature = (
            self.geometry_mode,
            self.force_mode,
            self.calibrations,
        )

    def workspace(self):
        return CollisionWorkspace(
            self.system.model, self.geometry_mode, broad_phase="bvh", sheet_bounds=True
        )

    def evaluate(self, cloth, workspace):
        loads = self.system.sample_particles(cloth, workspace)
        contacts = []
        for row in loads.contacts:
            first_nodes = self.system.bindings[row.first].rows
            second_nodes = self.system.bindings[row.second].rows
            sites = tuple(
                dict(
                    first_weights=site.material.first_weights,
                    second_weights=site.material.second_weights,
                    first_point_m=site.application_point_first_m,
                    second_point_m=site.application_point_second_m,
                    force_on_first_n=-np.asarray(site.force_on_second_n),
                    force_on_second_n=site.force_on_second_n,
                    fraction=site.fraction,
                    force_n=site.force_n,
                    normal_speed_m_s=site.rate_m_per_s,
                    power_w=site.power_w,
                )
                for site in row.sites
            )
            contacts.append(
                dict(
                    faces=(self.names[row.first.shape], self.names[row.second.shape]),
                    first_nodes=first_nodes,
                    second_nodes=second_nodes,
                    first_positions_m=cloth.position[list(first_nodes)].copy(),
                    second_positions_m=cloth.position[list(second_nodes)].copy(),
                    first_velocities_m_s=cloth.velocity[list(first_nodes)].copy(),
                    second_velocities_m_s=cloth.velocity[list(second_nodes)].copy(),
                    normal=row.normal,
                    middle_gap_m=row.gap_m,
                    force_n=row.force_n,
                    potential_j=row.potential_j,
                    power_w=row.power_w,
                    sites=sites,
                )
            )
        return dict(
            geometry_backend=self.geometry_mode,
            force_backend=self.force_mode,
            nodal_force_n=loads.inputs.particles.force,
            contacts=tuple(contacts),
            registered_pairs=len(self.system.layers),
            broad_candidates=sum(pair.candidate for pair in workspace.last_result.pairs),
        )

    def check_path(self, first, middle, last, workspace):
        with self.workspace() if workspace is None else nullcontext(workspace) as current:
            for start, end in ((first, middle), (middle, last)):
                self.system.check_particle_segment(start, end, current)
