"""Run the frozen two-U input through the public Phymeo mesh collision query."""

from fractions import Fraction as F

import numpy as np


def _record_support(support, reverse, lookup, tolerance):
    w = support.witness
    first, second = (1, 0) if reverse else (0, 1)
    points = (w.first_point, w.second_point)
    weights = (w.first_weights, w.second_weights)
    faces = (support.first_faces, support.second_faces)
    for side, owner in enumerate((first, second)):
        material = weights[owner]
        assert faces[owner] and all({i for i, v in material if v} <= set(f) for f in faces[owner])
        restored = sum(v * lookup[side][i] for i, v in material)
        np.testing.assert_allclose(restored, points[owner], rtol=0, atol=tolerance)
    normal = w.normal
    if reverse and normal is not None:
        normal = tuple(-n for n in normal)
    return dict(
        gap_m=w.gap_m,
        normal_a_to_b=normal,
        a_support_world_m=points[first],
        b_support_world_m=points[second],
        a_weights=weights[first],
        b_weights=weights[second],
        a_faces=faces[first],
        b_faces=faces[second],
    )


def _first_observation(case, model, keys, vertices, sources, paths, reverse, work):
    import phymeo as ph

    start = [v + path[0] for v, path in zip(vertices, paths, strict=True)]
    end = [v + path[1] for v, path in zip(vertices, paths, strict=True)]
    motion = {key: ph.CollisionMotion(a, b) for key, a, b in zip(keys, start, end, strict=True)}
    (pair,) = ph.detect(model, motion, ph.CollisionPolicy(), work).pairs
    assert pair.status == "hit" and pair.mesh_mesh.entry is not None
    trace, sweep = pair.mesh_mesh, pair.sweep
    expected, tol = case["expected"], case["tolerances"]
    exact = F(*trace.first_ratio)
    assert exact == F(str(expected["first_fraction"]))
    lo, hi = sweep.first_t
    assert F(lo) <= exact <= F(hi) and hi - lo <= tol["time_fraction"]
    assert lo <= sweep.witness_time <= hi
    assert trace.start.relation == "separated" and trace.entry.relation == "touching"
    assert trace.end.relation == "boundary_overlap" and trace.entry.boundary
    # MeshMeshTrace.entry is the exact rational instant; its public points and
    # weights are FP64. Reconstruct them from each body's independently posed vertices.
    t = float(exact)
    origins = [np.asarray(a) + t * (np.asarray(b) - a) for a, b in paths]
    lookup = [
        dict(zip(ids, v + o, strict=True))
        for ids, v, o in zip(sources, vertices, origins, strict=True)
    ]
    records = [_record_support(s, reverse, lookup, tol["length_m"]) for s in trace.entry.boundary]
    matches = 0
    for s in records:
        world_a, world_b = np.asarray(s["a_support_world_m"]), np.asarray(s["b_support_world_m"])
        if (
            s["normal_a_to_b"] is not None
            and np.allclose(world_a, world_b, rtol=0, atol=tol["length_m"])
            and np.allclose(
                s["normal_a_to_b"],
                expected["first_normal_a_to_b"],
                rtol=0,
                atol=tol["normal_component"],
            )
            and abs(world_a[0] - origins[0][0] - expected["first_a_local_x_m"]) <= tol["length_m"]
        ):
            matches += 1
    duration = case["inputs"]["duration_s"]
    report = dict(
        backend=work.stats()["backend"],
        first_ratio=trace.first_ratio,
        first_interval_fraction=sweep.first_t,
        first_interval_s=[lo * duration, hi * duration],
        first_time_s=t * duration,
        stored_witness_fraction=sweep.witness_time,
        start_relation=trace.start.relation,
        end_relation=trace.end.relation,
        entry_relation=trace.entry.relation,
        entry_origins_world_m=[o.tolist() for o in origins],
        entry_boundary=records,
        strict_matches=matches,
    )
    if matches != len(records) or not lo * duration <= expected["first_time_s"] <= hi * duration:
        error = AssertionError(
            f"{matches}/{len(records)} first-touch records match the frozen wall"
        )
        error.observed = report
        raise error
    return report


def run_case(case, inputs, assets, device):
    import phymeo as ph

    from . import _u_geometry

    if device != "cpu" or [a["kind"] for a in assets] != ["u_mesh", "u_mesh"]:
        raise ValueError("The closed-mesh comparison requires two actual triangle meshes and HOST")
    reverse = inputs.get("reverse_registration", False)
    keys = [ph.CollisionKey(0, i, 0) for i in ((2, 1) if reverse else (1, 2))]
    vertices, sources, shapes = [], [], []
    for key, asset in zip(keys, assets, strict=True):
        points, indices = _u_geometry(asset["scale_xyz"], inputs.get("flip_quad_diagonal", False))
        points = np.asarray(points, dtype=float)
        ids = tuple(range(len(points)))
        faces = tuple(map(tuple, indices.reshape(-1, 3).tolist()))
        vertices.append(points)
        sources.append(ids)
        shapes.append(ph.CollisionShape(key, "mesh", ids, triangles=faces))
    paths = inputs["a_path_m"], inputs["b_path_m"]
    model = ph.CollisionModel(shapes)
    length = case["tolerances"].get("length_m", 1e-10)
    samples = []
    failures = []
    with ph.CollisionWorkspace(model, "host") as work:
        if inputs["query"] == "first_touch":
            return _first_observation(case, model, keys, vertices, sources, paths, reverse, work)
        for t, wanted in zip(inputs["sample_fractions"], case["expected"]["samples"], strict=True):
            origins = [np.asarray(p0) + t * (np.asarray(p1) - p0) for p0, p1 in paths]
            coords = [p + o for p, o in zip(vertices, origins, strict=True)]
            motion = {k: ph.CollisionMotion(c) for k, c in zip(keys, coords, strict=True)}
            (pair,) = ph.detect(model, motion, ph.CollisionPolicy(), work).pairs
            assert pair.family == "RR" and pair.status in ("separated", "touching", "overlap")
            result = dict(
                fraction=t,
                prescribed_time_s=t * inputs["duration_s"],
                status=pair.status,
                contact=pair.status != "separated",
                origins_world_m=[p.tolist() for p in origins],
            )
            assert pair.candidate and pair.mesh_mesh is not None, "Both U-shape AABBs overlap"
            pose = pair.mesh_mesh.end
            lookup = [
                dict(zip(ids, x, strict=True)) for ids, x in zip(sources, coords, strict=True)
            ]
            deepest = [_record_support(s, reverse, lookup, length) for s in pose.deepest]
            boundary = [_record_support(s, reverse, lookup, length) for s in pose.boundary]
            result.update(
                relation=pose.relation,
                boundary_gap_m=pose.boundary_gap_m,
                vertex_intrusion_m=pose.vertex_intrusion_m,
                deepest=deepest,
                boundary=boundary,
                boundary_facet_pairs_tested=pose.boundary_facet_pairs_tested,
            )
            if wanted["contact"]:
                lo, hi = wanted["gap_range_m"]
                normal = np.asarray(wanted["normal_a_to_b"])
                wall = wanted["a_local_x_m"]
                correct = [
                    lo - length <= s["gap_m"] <= hi + length
                    and s["normal_a_to_b"] is not None
                    and np.allclose(
                        s["normal_a_to_b"],
                        normal,
                        rtol=0,
                        atol=case["tolerances"]["normal_component"],
                    )
                    and abs(s["a_support_world_m"][0] - origins[0][0] - wall) <= length
                    for s in deepest
                ]
                result["strict_matches"] = int(sum(correct))
                if not boundary or not deepest or not all(correct):
                    failures.append(
                        f"t={t} {sum(correct)}/{len(deepest)} complete intrusion witnesses, {len(boundary)} boundary intersections"
                    )
                if deepest:
                    w = pose.witness
                    direction = w.normal
                    if reverse and direction is not None:
                        direction = tuple(-n for n in direction)
                    result.update(
                        gap_m=w.gap_m,
                        normal_a_to_b=direction,
                        a_support_world_m=w.second_point if reverse else w.first_point,
                        b_support_world_m=w.first_point if reverse else w.second_point,
                    )
            samples.append(result)
        observed = dict(backend=work.stats()["backend"], samples=samples)
    if failures:
        error = AssertionError(" | ".join(failures))
        error.observed = observed
        raise error
    return observed
