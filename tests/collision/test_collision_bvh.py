"""Stackless AABB traversal, frame refit, and true mesh occupancy after pruning."""

from dataclasses import replace

import pytest
from m1_cases import mixed_scene
from spatial_index_cases import mesh_scene, sheet
from test_spatial_broad_phase import balls

import phymeo as ph


@pytest.mark.parametrize(
    "make", [lambda: sheet(3), lambda: sheet(6), lambda: (balls(34, True)[0], [balls(34, True)[1]])]
)
def test_refit_matches_all_pairs_on_every_frame(make):
    model, frames = make()
    with (
        ph.CollisionWorkspace(model) as baseline,
        ph.CollisionWorkspace(model, broad_phase="bvh") as tree,
    ):
        for paths in frames:
            assert ph.detect(model, paths, ph.CollisionPolicy(), tree) == ph.detect(
                model, paths, ph.CollisionPolicy(), baseline
            )
        counts = tree.stats()
        assert counts["shape_bvh_builds"] == 1 and counts["shape_bvh_refits"] == len(frames)
        assert counts["shape_bvh_node_visits"] > 0


def test_first_capacity_failure_does_not_publish_layout_or_pair_history():
    model, frames = sheet(3)
    with ph.CollisionWorkspace(model, broad_phase="bvh") as tree:
        no = ph.detect(model, frames[0], ph.CollisionPolicy(candidate_capacity=0), tree)
        assert no.status == "capacity_exceeded" and tree.last_result is None
        first = ph.detect(model, frames[1], ph.CollisionPolicy(), tree)
        assert tree.stats()["shape_bvh_builds"] == 2
        no = ph.detect(model, frames[2], ph.CollisionPolicy(candidate_capacity=0), tree)
        assert no.status == "capacity_exceeded" and tree.last_result is first
        restored = ph.detect(model, frames[1], ph.CollisionPolicy(), tree)
        assert restored == first and tree.stats()["shape_bvh_builds"] == 2
        changed = ph.CollisionModel([replace(s, revision=1) for s in model.shapes])
        with pytest.raises(ValueError, match="different collision model"):
            ph.detect(changed, frames[1], ph.CollisionPolicy(), tree)
        with ph.CollisionWorkspace(changed, broad_phase="bvh") as new:
            update = ph.detect(changed, frames[1], ph.CollisionPolicy(), new)
            assert (
                all(b.revision == 1 for b in update.bounds) and new.stats()["shape_bvh_builds"] == 1
            )


@pytest.mark.parametrize("refine", [False, True])
def test_mesh_facet_candidates_preserve_clearance_and_filled_relations(refine):
    model, frames = mesh_scene(refine)
    with (
        ph.CollisionWorkspace(model) as brute,
        ph.CollisionWorkspace(model, broad_phase="sap", mesh_midphase="bvh") as work,
    ):
        counts = []
        for paths in frames:
            prev = work.stats()["mesh_facet_pairs_tested"]
            ref = ph.detect(model, paths, ph.CollisionPolicy(), brute)
            result = ph.detect(model, paths, ph.CollisionPolicy(), work)
            assert result == ref
            counts.append(work.stats()["mesh_facet_pairs_tested"] - prev)
        assert 0 < counts[0] < len(model.shapes[0].triangles) and counts[1:] == [0, 0]
        assert work.stats()["mesh_bvh_builds"] == 1 and work.stats()["mesh_bvh_refits"] == 3
        assert brute.stats()["mesh_facet_pairs_tested"] == 3 * len(model.shapes[0].triangles)


def test_unbounded_planes_multiple_worlds_and_empty_model():
    model, paths, _, _ = mixed_scene()
    k = ph.CollisionKey(0, 80, 0)
    model = ph.CollisionModel(
        [*model.shapes, ph.CollisionShape(k, "plane", (1,), normal=(0, 0, 1))], model.excluded
    )
    paths[k] = ph.CollisionMotion([[1000, 2000, 3000]])
    with (
        ph.CollisionWorkspace(model) as baseline,
        ph.CollisionWorkspace(model, broad_phase="bvh") as work,
    ):
        assert ph.detect(model, paths, ph.CollisionPolicy(), work) == ph.detect(
            model, paths, ph.CollisionPolicy(), baseline
        )
    empty = ph.CollisionModel(())
    with ph.CollisionWorkspace(empty, broad_phase="bvh") as work:
        assert ph.detect(empty, {}, ph.CollisionPolicy(), work).pairs == ()
