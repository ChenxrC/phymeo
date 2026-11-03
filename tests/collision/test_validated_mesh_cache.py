"""Closed-mesh certification reuse without cross-shape, cross-motion or failure aliasing."""

from dataclasses import replace

import numpy as np
import pytest
from closed_mesh_cases import authored

import phymeo as ph
from phymeo.geometry import collision_pipeline as pipeline


def duplicate(paths):
    return {
        k: ph.CollisionMotion(p.start.copy(), None if p.end is None else p.end.copy(), p.turn)
        for k, p in paths.items()
    }


@pytest.mark.parametrize("backend", ["host", "cuda"])
def test_equal_material_bytes_are_prepared_once_but_consumers_are_fresh(
    backend, tmp_path, monkeypatch
):
    model, paths, _ = authored("base")
    mesh, hull = model.shapes
    original = pipeline.ClosedMesh
    builds = []

    def counted(shape, motion):
        builds.append(shape.key)
        return original(shape, motion)

    monkeypatch.setattr(pipeline, "ClosedMesh", counted)
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model, backend, cache_root=tmp_path) as work:
        a = ph.detect(model, paths, policy, work)
        independent = duplicate(paths)
        assert independent[mesh.key] is not paths[mesh.key]
        assert independent[mesh.key].start is not paths[mesh.key].start
        assert ph.detect(model, independent, policy, work) == a
        assert len(builds) == 1, "identical closed-mesh geometry was certified again"
        moved = dict(independent)
        moved[hull.key] = ph.CollisionMotion(
            paths[hull.key].start, turn=ph.RigidTurn((0, 0, 0), (0.6, 0, 0), (0, 0, 1), 0)
        )
        b = ph.detect(model, moved, policy, work)
        assert b.pairs[0].status == "separated" and a.pairs[0].status == "hit"
        assert len(builds) == 1 and work.stats()["validated_mesh_hits"] == 2
        assert work.stats()["validated_mesh_builds"] == work.stats()["validated_mesh_attempts"] == 1


def test_bad_fold_and_replacement_never_reuse_the_previous_mesh(tmp_path):
    model, paths, _ = authored("base")
    mesh = model.shapes[0]
    key = mesh.key
    bad = paths[key].start.copy()
    bad[10] = [-1.75, 1.5, 0.25]
    invalid = dict(paths)
    invalid[key] = ph.CollisionMotion(bad)
    change = dict(paths)
    changed = paths[key].start.copy()
    changed[:, 0] += 0.125
    change[key] = ph.CollisionMotion(changed)
    p = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model, cache_root=tmp_path) as work:
        first = ph.detect(model, paths, p, work)
        for _ in range(2):
            broken = ph.detect(model, invalid, p, work)
            assert broken.pairs[0].status == "invalid_geometry"
        assert work.stats()["validated_mesh_builds"] == 1
        assert work.stats()["validated_mesh_attempts"] == 3
        other = ph.detect(model, change, p, work)
        assert other.pairs[0].mesh.first_ratio != first.pairs[0].mesh.first_ratio
        assert ph.detect(model, paths, p, work) == first
        assert work.stats()["validated_mesh_builds"] == 3
        assert work.stats()["validated_mesh_hits"] == 0


@pytest.mark.parametrize("backend", ["host", "cuda"])
def test_capacity_and_world_boundaries_keep_publication_and_validation_separate(backend, tmp_path):
    from dataclasses import replace

    model, paths, _ = authored("base")
    group = {}
    for w in (27, 5):
        ids = {s.key: replace(s.key, world=w) for s in model.shapes}
        group[w] = (
            ph.CollisionModel(tuple(replace(s, key=ids[s.key]) for s in model.shapes)),
            {ids[k]: v for k, v in paths.items()},
        )
    rules = dict.fromkeys(group, ph.CollisionPolicy())
    frames = {w: p for w, (_, p) in group.items()}
    with ph.WorldCollisionWorkspace(
        [(w, m) for w, (m, _) in group.items()], backend, cache_root=tmp_path
    ) as work:
        rows = work.detect(frames, rules)
        assert all(r.pairs[0].status == "hit" for r in rows.values())
        assert work.stats()["validated_mesh_builds"] == 2
        limited = dict(rules)
        limited[27] = ph.CollisionPolicy(candidate_capacity=0)
        trial = work.detect(frames, limited)
        assert trial[27].status == "capacity_exceeded" and work.last_result(27) is rows[27]
        assert trial[5] == rows[5] and work.stats()["validated_mesh_hits"] == 1
        assert work.detect(frames, rules) == rows
        assert (
            work.stats()["validated_mesh_builds"] == 2 and work.stats()["validated_mesh_hits"] == 3
        )


def test_shape_identity_motion_representation_and_exact_step_are_cache_inputs():
    from dataclasses import FrozenInstanceError

    from phymeo.geometry._closed_mesh import ClosedMesh
    from phymeo.geometry._validated_mesh_cache import ValidatedMeshCache

    model, paths, _ = authored("base")
    mesh = model.shapes[0]
    base = paths[mesh.key].start
    cache = ValidatedMeshCache()
    first = cache.get(mesh, ph.CollisionMotion(base.copy()), ClosedMesh)
    node = mesh.vertex_ids[0]
    with pytest.raises(TypeError):
        first.points[node] = np.zeros(3)
    with pytest.raises(ValueError):
        first.points[node][0] = 1
    with pytest.raises(ValueError):
        first.facets[0].xyz[0, 0] = 10
    with pytest.raises(ValueError):
        first.facets[0].face.normal[0] = 11
    with pytest.raises(FrozenInstanceError):
        first.step = np.zeros(3)
    assert cache.get(mesh, ph.CollisionMotion(base.copy()), ClosedMesh) is first
    assert cache.get(mesh, ph.CollisionMotion(base.copy(), base.copy()), ClosedMesh) is not first
    one = ph.CollisionMotion(base, base + [0.125, 0, 0])
    a = cache.get(mesh, one, ClosedMesh)
    two = ph.CollisionMotion(base, base + [0.25, 0, 0])
    b = cache.get(mesh, two, ClosedMesh)
    assert a is not b and float(a.step[0]) == 0.125 and float(b.step[0]) == 0.25
    t1 = ph.CollisionMotion(base, turn=ph.RigidTurn((0, 0, 0), (0.25, 0, 0), (0, 0, 1), 0))
    t2 = ph.CollisionMotion(base, turn=ph.RigidTurn((1, 0, 0), (1.25, 0, 0), (0, 0, 1), 0))
    t3 = ph.CollisionMotion(base, turn=ph.RigidTurn((1, 0, 0), (1.25, 0, 0), (0, 1, 0), 0))
    r1 = cache.get(mesh, t1, ClosedMesh)
    r2 = cache.get(mesh, t2, ClosedMesh)
    r3 = cache.get(mesh, t3, ClosedMesh)
    assert r1 is not b and r2 is not r1 and r3 is not r2
    assert cache.get(mesh, ph.CollisionMotion(base.copy(), turn=t3.turn), ClosedMesh) is r3
    changed_angle = ph.CollisionMotion(base, turn=replace(t3.turn, angle=0.125))
    assert cache.get(mesh, changed_angle, ClosedMesh) is not r3
    revision = replace(mesh, revision=mesh.revision + 1)
    assert cache.get(revision, t3, ClosedMesh) is not r3
    bad = replace(revision, triangles=revision.triangles[:-1])
    with pytest.raises(ValueError, match="mesh_open_or_misoriented_edge"):
        cache.get(bad, t3, ClosedMesh)
    assert len(cache) == 0
    cache.get(revision, t3, ClosedMesh)
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0


def test_solver_failure_is_retried_and_never_used_as_validated_geometry(tmp_path, monkeypatch):
    from phymeo.core.capabilities import SolverFailure

    original = pipeline.ClosedMesh
    attempts = []

    def temporary(shape, path):
        attempts.append(shape.key)
        if len(attempts) <= 2:
            raise SolverFailure("injected certificate failure")
        return original(shape, path)

    monkeypatch.setattr(pipeline, "ClosedMesh", temporary)
    model, paths, _ = authored("base")
    rule = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model, cache_root=tmp_path) as work:
        for _ in range(2):
            assert ph.detect(model, paths, rule, work).pairs[0].status == "unresolved"
        accepted = ph.detect(model, paths, rule, work)
        assert accepted.pairs[0].status == "hit"
        assert ph.detect(model, duplicate(paths), rule, work) == accepted
        stats = work.stats()
        assert (
            stats["validated_mesh_attempts"] == 3
            and stats["validated_mesh_builds"] == 1
            and stats["validated_mesh_hits"] == 1
        )
    assert len(work._validated_meshes) == 0


def test_cuda_fault_clears_closed_workspace_certificate_before_rebuilding(tmp_path, monkeypatch):
    model, paths, _ = authored("base", world=7)
    policies = {7: ph.CollisionPolicy()}
    with ph.WorldCollisionWorkspace([(7, model)], "cuda", cache_root=tmp_path) as batch:
        old = batch.detect({7: paths}, policies)[7]
        stale = batch._work._validated_meshes
        assert len(stale) == 1

        def broken(*args):
            raise RuntimeError("injected classification download failure")

        monkeypatch.setattr(batch._work._engine, "_read", broken)
        with pytest.raises(RuntimeError, match="classification download"):
            batch.detect({7: paths}, policies)
        assert batch.last_result(7) is old and len(stale) == 0
        recovered = batch.detect({7: duplicate(paths)}, policies)[7]
        assert recovered == old and batch._work._validated_meshes is not stale
        assert (
            batch.stats()["validated_mesh_builds"] == 2
            and batch.stats()["validated_mesh_hits"] == 0
        )
