"""Ordinary Phymeo DD face distances and the concrete two-sided particle load system."""

import os
from contextlib import ExitStack
from itertools import combinations

import numpy as np

import phymeo as ph

from .oracles import check_forces, weights_check


def observe(case, device):
    inp = case["inputs"]
    shapes = []
    paths = {}
    bindings = {}
    ids = []
    positions = []
    face_keys = {}
    for sheet in inp["sheets"]:
        lookup = {}
        for ident, *xyz in sheet["vertices"]:
            lookup[ident] = len(positions)
            ids.append(ident)
            positions.append(xyz)
        for index, face in enumerate(sheet["faces"]):
            key = ph.CollisionKey(0, sheet["body"], index)
            shape = ph.CollisionShape(key, "triangle", tuple(face))
            shapes.append(shape)
            face_keys[tuple(sorted(face))] = key
            rows = tuple(lookup[n] for n in face)
            bindings[key] = ph.ParticleAttachment(rows)
            paths[key] = ph.CollisionMotion([positions[i] for i in rows])
    model = ph.CollisionModel(tuple(shapes))
    force_layers = {
        tuple(sorted((a.key, b.key))): ph.NormalLayer(
            inp["self_margin_m"], inp["stiffness_n_per_m"]
        )
        for a, b in combinations(shapes, 2)
    }
    focus = [face_keys[tuple(sorted(f))] for f in inp["focus_faces"]]
    mode = "host" if device == "cpu" else "cuda"
    cache = os.environ.get("PHY_DD_REFERENCE_CACHE")
    with ExitStack() as stack:
        work = stack.enter_context(ph.CollisionWorkspace(model, mode, cache_root=cache))
        force_op = (
            stack.enter_context(ph.TyxNormalLayer("cuda", cache_root=cache))
            if mode == "cuda"
            else None
        )
        plain = ph.detect(
            model,
            paths,
            ph.CollisionPolicy(margin=inp["phymeo_query_margin_m"], proximity_patches=True),
            work,
        )
        assert plain.status == "ok"
        row = next(v for v in plain.pairs if {v.first, v.second} == set(focus))
        witness = row.witness
        forward = row.first == focus[0]
        positions = np.asarray(positions, dtype=float)
        sys = ph.CollisionLoadSystem(model, bindings, force_layers, layer_operator=force_op)
        loads = sys.sample_particles(ph.ParticleState(positions, np.zeros_like(positions)), work)
        geometry = work.stats()
        counted = None if force_op is None else force_op.stats()
    first_weights = () if witness is None else witness.first_weights
    second_weights = () if witness is None else witness.second_weights
    a_weights, b_weights = (
        (first_weights, second_weights) if forward else (second_weights, first_weights)
    )
    native_faces = {key: list(face) for face, key in face_keys.items()}
    return dict(
        device=device,
        backend=mode,
        material_order=ids,
        focus=dict(
            status=row.status,
            reason=row.reason,
            candidate=row.candidate,
            gap_m=None if witness is None else witness.gap_m,
            weights_a=a_weights,
            weights_b=b_weights,
            point_a_m=None
            if witness is None
            else (witness.first_point if forward else witness.second_point),
            point_b_m=None
            if witness is None
            else (witness.second_point if forward else witness.first_point),
            shared_material=[]
            if witness is None
            else [
                dict(point_m=s.point, first=s.first_weights, second=s.second_weights)
                for s in witness.shared_material_points
            ],
        ),
        geometry_pairs=len(plain.pairs),
        geometry_candidates=plain.required_candidates,
        forces_n={
            str(i): v.tolist() for i, v in zip(ids, loads.inputs.particles.force, strict=True)
        },
        loads=[
            dict(
                faces=[native_faces[load.first], native_faces[load.second]],
                gap_m=load.gap_m,
                force_n=load.force_n,
                sites=[
                    dict(
                        weights_a=site.material.first_weights,
                        weights_b=site.material.second_weights,
                        force_on_second_n=site.force_on_second_n,
                    )
                    for site in load.sites
                ],
            )
            for load in loads.contacts
        ],
        geometry_stats={
            k: v
            for k, v in geometry.items()
            if k
            in (
                "backend",
                "completed_launches",
                "downloads",
                "uploaded_bytes",
                "downloaded_bytes",
                "synchronizations",
            )
        },
        force_stats=counted,
    )


def validate(data, case):
    tol, exp = case["tolerances"], case["expected"]
    row = data["focus"]
    assert row["candidate"] and row["gap_m"] is not None
    assert abs(row["gap_m"] - exp["focus_distance_m"]) <= tol["host_m"]
    if case["family"] == "adjacent":
        assert row["status"] == "adjacent" and row["shared_material"]
    else:
        assert row["status"] == "separated"
        if exp["focus_weights_a"]:
            weights_check(row["weights_a"], exp["focus_weights_a"], tol["weight"])
            weights_check(row["weights_b"], exp["focus_weights_b"], tol["weight"])
    if data["backend"] == "cuda":
        assert data["geometry_stats"]["completed_launches"] > 0
        if data["loads"]:
            assert (
                data["force_stats"]["completed_launches"] > 0
                and data["force_stats"]["downloads"] > 0
            )
    check_forces(data, case, "phymeo")
