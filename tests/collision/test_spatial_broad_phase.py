"""The spatial broad phase preserves every public pair, not only positive cases."""

import pytest
from m1_cases import mixed_scene

import phymeo as ph


def run_both(model, paths, rule, **options):
    with ph.CollisionWorkspace(model) as exhaustive:
        ref = ph.detect(model, paths, rule, exhaustive)
    with ph.CollisionWorkspace(model, **options) as spatial:
        actual = ph.detect(model, paths, rule, spatial)
        assert actual == ref
        return actual, spatial.stats()


def balls(n, dense=False):
    shapes, paths = [], {}
    for i in range(n):
        k = ph.CollisionKey(3, i, 0)
        shapes.append(ph.CollisionShape(k, "sphere", (0,), 0.125))
        y = i / (8 * n) if dense else 2 * i
        paths[k] = ph.CollisionMotion([[0, y, 0]])
    return ph.CollisionModel(shapes), paths


@pytest.mark.parametrize("count,dense", [(1, False), (34, False), (65, False), (34, True)])
def test_sparse_dense_world_and_bit_word_boundaries(count, dense):
    model, paths = balls(count, dense)
    ref, stats = run_both(model, paths, ph.CollisionPolicy(), broad_phase="sap")
    slots = count * (count - 1) // 2
    assert len(ref.pairs) == slots
    assert ref.required_candidates == (slots if dense else 0)
    assert stats["sap_window_pairs"] == (slots if dense else 0)
    assert stats["sap_box_tests"] == stats["sap_window_pairs"]


def test_filters_planes_closed_ties_and_new_motion_reuse():
    model, paths, _, _ = mixed_scene(reverse=True)
    plane = ph.CollisionKey(0, 50, 1)
    shape = ph.CollisionShape(plane, "plane", (7,), normal=(0, 1, 0))
    model = ph.CollisionModel((*model.shapes, shape), model.excluded)
    paths[plane] = ph.CollisionMotion([[1000, 2000, 3000]])
    with (
        ph.CollisionWorkspace(model) as reference,
        ph.CollisionWorkspace(model, broad_phase="sap") as work,
    ):
        expected = ph.detect(model, paths, ph.CollisionPolicy(margin=0.25), reference)
        first = ph.detect(model, paths, ph.CollisionPolicy(margin=0.25), work)
        assert first == expected
        failed = ph.detect(
            model, paths, ph.CollisionPolicy(candidate_capacity=0, margin=0.25), work
        )
        assert failed.status == "capacity_exceeded" and work.last_result is first
        assert failed.required_candidates == first.required_candidates
        assert ph.detect(model, paths, ph.CollisionPolicy(margin=0.25), work) == first
    with pytest.raises(ValueError, match="broad phase"):
        ph.CollisionWorkspace(model, broad_phase="octree")


def test_closed_endpoints_outward_fp32_and_empty_model():
    model, paths = balls(2)
    a, b = (shape.key for shape in model.shapes)
    paths[a] = ph.CollisionMotion([[2**24 + 1, 0, 0]])
    paths[b] = ph.CollisionMotion([[2**24 + 1.25, 0, 0]])
    result, _ = run_both(model, paths, ph.CollisionPolicy(), broad_phase="sap")
    assert result.pairs[0].status == "touching"
    empty = ph.CollisionModel(())
    assert run_both(empty, {}, ph.CollisionPolicy(), broad_phase="sap")[0].pairs == ()
