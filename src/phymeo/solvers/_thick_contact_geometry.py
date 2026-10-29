"""Fixed B/s material anchors and full-prism geometry for one native replay step."""

from dataclasses import dataclass
from fractions import Fraction as F
from math import fsum

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.scene import fingerprint
from ..coupling.offset_port import TriangleOffsetPointPort
from ..coupling.ports import ParticlePointPort
from ..dynamics.robot_kinematics import point_jacobians
from ..geometry._prism_exact import Quadratic as Q
from ..geometry._prism_exact import vector
from ..geometry.prism_patch_manifold import prefer_initial_axis, triangle_prism_patch_manifold
from ..geometry.robot_patch_sweep import patch_geometry
from ..validation._material_path_math import q
from ..validation.thick_contact_mapping import ThickContactMappingLedger


@dataclass(frozen=True)
class MaterialAnchor:
    pair: int
    material: tuple
    ideal_material: tuple
    feature: tuple
    terminal_only: bool = False
    representation_error_m: float = 0.0

    @property
    def token(self):
        return (self.pair,) + tuple(self.material)


def convert_material(material):
    # A public 52-bit binary simplex grid has exact sum one and is idempotent
    # under the offset port's normalization, including its first-coordinate repair.
    weights = [float(v) for v in material[:3]]
    grid = 2**52
    if min(weights) < 0 or abs(fsum(weights) - 1) > 1e-12:
        raise SolverFailure("invalid ideal convex material coordinates")
    counts = [max(0, int(round(v * grid))) for v in weights]
    counts[max(range(3), key=weights.__getitem__)] += grid - sum(counts)
    if min(counts) < 0:
        raise SolverFailure("unrepresentable convex material weights")
    result = tuple(v / grid for v in counts) + (float(material[3]),)
    if fsum(result[:3]) != 1:
        raise SolverFailure("non-idempotent material representation")
    return result


def geometry_key(w, state):
    return fingerprint(
        (
            state.domains[w.domain].q,
            tuple((name, state.domains[name].position) for name in w.node_domains),
            tuple(a.material for a in w.anchors),
        )
    )


def manifolds(w, state):
    key = geometry_key(w, state)
    cache = getattr(w, "_manifold_cache", {})
    if key in cache:
        return cache[key]
    results = []
    for s, c in w.program.pairs:
        try:
            results.append(
                triangle_prism_patch_manifold(
                    w.model,
                    state.domains[w.domain].q,
                    c,
                    state.domains[s.domain].position[list(s.node_indices)],
                    half_thickness_m=s.half_thickness_m,
                    geometric_budget_m=w.bound,
                )
            )
        except (ValueError, ZeroDivisionError) as error:
            raise SolverFailure("whole-prism geometry unresolved: " + str(error)) from error
    if w.initial_manifolds is not None:
        results = [
            prefer_initial_axis(
                m,
                w.initial_manifolds[i].selected_axis.direction,
                w.program.pairs[i][1],
                w.program.pairs[i][0].half_thickness_m,
                w.bound,
            )
            for i, m in enumerate(results)
        ]
    if len(cache) > 6:
        cache.clear()
    cache[key] = tuple(results)
    w._manifold_cache = cache
    return cache[key]


def discover(w, state, terminal_only=False, penetrating_only=False):
    result = manifolds(w, state)
    added = []
    old = {a.token for a in w.anchors}
    for pair, m in enumerate(result):
        if penetrating_only and m.exact_gap_bounds[0] >= -w.bound:
            continue
        # A free trial may cross the patch. Keep the initial support side for
        # held root generation; final whole-body geometry remains independent.
        if (
            not terminal_only
            and w.initial_manifolds is not None
            and np.dot(m.normal, w.initial_manifolds[pair].normal) < 0
        ):
            continue
        for p in m.points:
            mat = convert_material(p.material)
            surface = w.program.pairs[pair][0]
            x = tuple(
                vector(row)
                for row in state.domains[surface.domain].position[list(surface.node_indices)]
            )
            normal = tuple(
                (a - b) / (2 * q(surface.half_thickness_m))
                for a, b in zip(m.exact_vertices[3], m.exact_vertices[0], strict=True)
            )
            ideal = tuple(
                sum((q(mat[i]) * x[i][k] for i in range(3)), Q(0)) + q(mat[3]) * normal[k]
                for k in range(3)
            )
            error = sum(
                (abs(a - b).bounds()[1] for a, b in zip(ideal, p.exact_prism_point, strict=True)),
                F(0),
            )
            if error > w.bound:
                raise SolverFailure("material conversion exceeds geometry budget")
            anchor = MaterialAnchor(
                pair,
                mat,
                p.material,
                (p.prism_feature, p.patch_feature),
                terminal_only,
                float(error),
            )
            if anchor.token not in old:
                added.append(anchor)
                old.add(anchor.token)
    count = {i: sum(a.pair == i for a in (*w.anchors, *added)) for i in range(len(w.program.pairs))}
    if any(n > w.program.max_anchors_per_pair for n in count.values()):
        raise SolverFailure("anchor capacity budget exhausted")
    return result, tuple(added)


def maps(w, state):
    key = geometry_key(w, state)
    cache = getattr(w, "_maps_cache", {})
    if key in cache:
        return cache[key]
    blocks = np.zeros((len(w.anchors), 3, 3, 3))
    points = np.zeros((len(w.anchors), 3))
    local = np.zeros_like(points)
    samples = []
    mapping = {}
    for name in w.node_domains:
        ids = [i for i, a in enumerate(w.anchors) if w.program.pairs[a.pair][0].domain == name]
        triangles = np.array(
            [w.program.pairs[w.anchors[i].pair][0].node_indices for i in ids], int
        ).reshape(-1, 3)
        weights = np.array([w.anchors[i].material[:3] for i in ids], float).reshape(-1, 3)
        offsets = np.array([w.anchors[i].material[3] for i in ids], float)
        base = ParticlePointPort(
            w.models[name],
            state.domains[name],
            np.arange(w.models[name].particle_count),
            domain=name,
        )
        port = TriangleOffsetPointPort(base, triangles, weights, offsets)
        if not np.array_equal(port.weights, weights):
            raise SolverFailure("owned material weights are not idempotent")
        blocks[ids] = port.jacobians
        points[ids] = port.positions()
        mapping[name] = (
            ids,
            ThickContactMappingLedger(state.domains[name].position, triangles, weights, offsets),
        )
    for i, a in enumerate(w.anchors):
        s, c = w.program.pairs[a.pair]
        sample = patch_geometry(w.model, state.domains[w.domain].q, c, points[i])
        samples.append(sample)
        local[i] = sample.local_point
        if q(sample.world_error_m) > w.bound:
            raise SolverFailure("patch representation exceeds geometry budget")
    indices = np.array([w.program.pairs[a.pair][1].link_index for a in w.anchors], int)
    jac = point_jacobians(w.model, state.domains[w.domain].q, indices, local)[0]
    if len(cache) > 6:
        cache.clear()
    cache[key] = (blocks, points, local, jac, tuple(samples), mapping)
    w._maps_cache = cache
    return cache[key]


def response(w, blocks, robot_jac, inverse):
    count = len(w.anchors)
    result = np.zeros((count, count, 3, 3))
    for i, a in enumerate(w.anchors):
        si = w.program.pairs[a.pair][0]
        for j, b in enumerate(w.anchors):
            sj = w.program.pairs[b.pair][0]
            block = robot_jac[i] @ inverse @ robot_jac[j].T
            if si.domain == sj.domain:
                for u, nu in enumerate(si.node_indices):
                    for v, nv in enumerate(sj.node_indices):
                        if nu == nv:
                            block = (
                                block
                                + w.models[si.domain].inverse_mass[nu]
                                * blocks[i, u]
                                @ blocks[j, v].T
                            )
            result[i, j] = block
    return result


def relative(w, state, blocks, jac):
    output = []
    for a, j, rj in zip(w.anchors, blocks, jac, strict=True):
        s = w.program.pairs[a.pair][0]
        velocities = state.domains[s.domain].velocity[list(s.node_indices)]
        output.append(
            [
                sum(
                    (
                        q(j[i, k, component]) * q(velocities[i, component])
                        for i in range(3)
                        for component in range(3)
                    ),
                    F(0),
                )
                - sum(
                    (
                        q(rj[k, d]) * q(state.domains[w.domain].v[d])
                        for d in range(w.model.dof_count)
                    ),
                    F(0),
                )
                for k in range(3)
            ]
        )
    return np.array(output, object).reshape(-1, 3)
