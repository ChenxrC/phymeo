"""Finite convex tools attached to real robot links, acting on a movable cloth middle surface."""

from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from itertools import combinations
from numbers import Integral

import numpy as np

from ..backends.tyx_normal_layer import TyxNormalLayer
from ..constraints.collision_shapes import (
    CollisionKey,
    CollisionModel,
    CollisionMotion,
    CollisionShape,
)
from ..core.capabilities import SolverFailure
from ..core.coupled import CoupledState
from ..core.rigid import RigidBodyState
from ..core.scene import _name
from ..dynamics.robot_kinematics import forward_kinematics
from ..dynamics.rotation import matrix_quaternion
from ..geometry._convex_hull import HullShape
from ..geometry.articulated_hull_clearance import link_hull_motion, require_cloth_clearance
from ..geometry.collision_pipeline import CollisionWorkspace
from .collision_loads import CollisionLoadSystem, NormalLayer, ParticleAttachment, RigidAttachment


@dataclass(frozen=True)
class LinkHullTool:
    """A zero-rounding convex solid attached in a link frame. Layer stiffness is per cloth face."""

    key: str
    robot_domain: str
    link_index: int
    local_vertices: tuple
    layer: NormalLayer
    vertex_ids: tuple | None = field(default=None, kw_only=True)

    def __post_init__(self):
        _name(self.key, "tool key")
        _name(self.robot_domain, "robot domain")
        if (
            isinstance(self.link_index, bool)
            or not isinstance(self.link_index, Integral)
            or self.link_index < 0
        ):
            raise ValueError("tool link index must be a nonnegative integer")
        if not isinstance(self.layer, NormalLayer):
            raise TypeError("expected a NormalLayer for the convex link tool")
        vertices = RigidAttachment(0, self.local_vertices).vertices
        ids = tuple(range(len(vertices))) if self.vertex_ids is None else tuple(self.vertex_ids)
        shape = CollisionShape(CollisionKey(0, 0, 0), "hull", ids)
        if len(shape.vertex_ids) != len(vertices):
            raise ValueError("tool material ids must match its local vertices")
        HullShape(shape, CollisionMotion(vertices))
        object.__setattr__(self, "link_index", int(self.link_index))
        object.__setattr__(
            self, "local_vertices", tuple(tuple(float(x) for x in p) for p in vertices)
        )
        object.__setattr__(self, "vertex_ids", shape.vertex_ids)


class ClothToolContacts:
    """Geometry and forces are evaluated on FK snapshots, then mapped into the robot's real DOFs."""

    def __init__(self, cloth_domain, surfaces, robots, specs, operator):
        if not all(isinstance(tool, LinkHullTool) for tool in specs):
            raise TypeError("expected LinkHullTool registrations")
        if len({s.key for s in specs}) != len(specs):
            raise ValueError("duplicate convex tool key")
        if operator is not None and not isinstance(operator, TyxNormalLayer):
            raise TypeError("link_hull_operator must be a TyxNormalLayer")
        self.cloth_domain = cloth_domain
        self.models = {k: r.model for k, r in robots.items()}
        self.owners = tuple(sorted({(s.robot_domain, s.link_index) for s in specs}))
        if any(n not in self.models or i >= self.models[n].dof_count for n, i in self.owners):
            raise ValueError("convex tool is attached to an unknown robot link")
        self.specs = tuple(sorted(specs, key=lambda s: s.key))
        self.force_mode = "host" if operator is None else operator.mode
        self.faces = {}
        self.tools = {}
        bindings = {}
        for i, surface in enumerate(surfaces):
            key = CollisionKey(0, 0, i)
            face = CollisionShape(key, "triangle", surface.node_indices)
            self.faces[key] = (face, surface)
            bindings[key] = ParticleAttachment(surface.node_indices)
        if len({tuple(sorted(f.vertex_ids)) for f, _ in self.faces.values()}) != len(self.faces):
            raise ValueError("repeated material triangle in the convex tool registration")
        laws = {}
        for i, spec in enumerate(self.specs):
            owner = self.owners.index((spec.robot_domain, spec.link_index))
            key = CollisionKey(0, 1 + owner, i)
            tool = CollisionShape(key, "hull", spec.vertex_ids)
            self.tools[key] = (tool, spec, owner)
            # CollisionLoadSystem expects its rigid local vertices relative to the COM.
            model = self.models[spec.robot_domain]
            bindings[key] = RigidAttachment(
                owner, np.asarray(spec.local_vertices) - model.com[spec.link_index]
            )
            for face, surface in self.faces.values():
                laws[face.key, key] = replace(
                    spec.layer, width_m=spec.layer.width_m + surface.half_thickness_m
                )
        shapes = [*(f for f, _ in self.faces.values()), *(h for h, _, _ in self.tools.values())]
        excluded = [(a.key, b.key) for a, b in combinations(shapes, 2) if a.rigid == b.rigid]
        self.system = CollisionLoadSystem(
            CollisionModel(shapes, excluded), bindings, laws, layer_operator=operator
        )

    def workspace(self):
        return CollisionWorkspace(self.system.model)

    def evaluate(self, state, geo, work=None):
        cloth = state.domains[self.cloth_domain]
        pos = []
        quat = []
        vel = []
        omega = []
        for name, i in self.owners:
            _, rot, com, jl, ja = geo[name][:5]
            speed = state.domains[name].v
            pos.append(com[i])
            quat.append(matrix_quaternion(rot[i]))
            vel.append(jl[i] @ speed)
            omega.append(ja[i] @ speed)
        actual = CoupledState(
            cloth, RigidBodyState(*(np.asarray(v) for v in (pos, quat, vel, omega)))
        )
        with self.workspace() if work is None else nullcontext(work) as ws:
            loads = self.system.sample(actual, ws)
        efforts = {n: np.zeros(self.models[n].dof_count) for n in self.models}
        rows = []
        for raw in loads.contacts:
            _, surface = self.faces[raw.first]
            _, spec, owner = self.tools[raw.second]
            name, i = spec.robot_domain, spec.link_index
            gap = raw.gap_m - surface.half_thickness_m
            if gap <= 0:
                raise SolverFailure(
                    f"convex tool {spec.key} intersects the material thickness of {surface.key}"
                )
            normal = np.asarray(raw.normal)
            sites = []
            F = np.zeros(3)
            M = np.zeros(3)
            for site in raw.sites:
                f = np.asarray(site.force_on_second_n)
                t = np.cross(np.asarray(site.application_point_second_m) - pos[owner], f)
                F += f
                M += t
                mat = site.material
                mid = np.asarray(mat.first_point)
                sites.append(
                    dict(
                        fraction=site.fraction,
                        force_on_tool_n=tuple(f),
                        rate_m_s=site.rate_m_per_s,
                        cloth_middle_point_m=mat.first_point,
                        cloth_surface_point_m=tuple(mid + surface.half_thickness_m * normal),
                        tool_point_m=mat.second_point,
                        cloth_weights=mat.first_weights,
                        tool_weights=mat.second_weights,
                        applied_cloth_point_m=site.application_point_first_m,
                        applied_tool_point_m=site.application_point_second_m,
                    )
                )
            *_, jl, ja = geo[name][:5]
            tau = jl[i].T @ F + ja[i].T @ M
            efforts[name] += tau
            rows.append(
                dict(
                    tool=spec.key,
                    face=surface.key,
                    robot_domain=name,
                    link_index=i,
                    middle_gap_m=raw.gap_m,
                    gap_m=gap,
                    half_thickness_m=surface.half_thickness_m,
                    normal=raw.normal,
                    force_n=raw.force_n,
                    force_on_tool_n=F,
                    torque_about_com_nm=M,
                    generalized_effort=tau,
                    power_w=raw.power_w,
                    potential_j=raw.potential_j,
                    sites=tuple(sites),
                )
            )
        return dict(
            contacts=tuple(rows),
            nodal_force_n=loads.inputs.particles.force,
            efforts=efforts,
            geometry_backend="host",
            force_backend=self.force_mode,
        )

    def check_path(self, first, last):
        cloth0 = first.domains[self.cloth_domain].position
        cloth1 = last.domains[self.cloth_domain].position
        poses = {}
        for name, _ in self.owners:
            if name not in poses:
                q0, q1 = first.domains[name].q, last.domains[name].q
                poses[name] = (
                    q0,
                    q1,
                    forward_kinematics(self.models[name], q0),
                    forward_kinematics(self.models[name], q1),
                )
        for shape, spec, _ in self.tools.values():
            tool, slack = link_hull_motion(
                self.models[spec.robot_domain], spec, shape, *poses[spec.robot_domain]
            )
            for face, surface in self.faces.values():
                nodes = list(surface.node_indices)
                path = CollisionMotion(cloth0[nodes], cloth1[nodes])
                require_cloth_clearance(tool, face, path, surface.half_thickness_m, slack)
