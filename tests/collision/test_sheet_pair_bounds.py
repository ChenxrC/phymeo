"""Exact supporting directions may simplify ordinary sheet pairs without hiding remote events."""

import numpy as np
import pytest
from adjacent_triangle_cases import hinge_path, mesh, vertex_path

import phymeo as ph


def query(model, paths, fast, margin=0.0):
    with ph.CollisionWorkspace(model, broad_phase="bvh", sheet_bounds=fast) as work:
        result = ph.detect(
            model, paths, ph.CollisionPolicy(margin=margin, proximity_patches=True), work
        )
        return result, work.stats()


@pytest.mark.parametrize("remote", [False, True])
def test_shared_edge_has_identical_material_and_first_event(remote):
    model, paths = hinge_path(remote)
    exact, _ = query(model, paths, False)
    fast, stats = query(model, paths, True)
    assert fast.pairs == exact.pairs
    assert fast.pairs[0].status == ("hit" if remote else "adjacent")
    if not remote:
        assert stats["sheet_pairs_adjacent"] == 1


def test_shared_vertex_cannot_mask_the_birth_of_a_nonmaterial_intersection():
    model, paths = vertex_path()
    exact, _ = query(model, paths, False)
    fast, _ = query(model, paths, True)
    assert exact.pairs == fast.pairs and fast.pairs[0].status == "hit"


@pytest.mark.parametrize("shift", [(0.0, 0.0, 0.0), (2.0**24, -(2.0**20), 8.0)])
def test_separated_vertex_fans_are_certified_in_both_storage_frames(shift):
    x = np.array([[0, 0, 0], [1.0, 0.0, 0], [0.0, 1, 0], [-1, 0, 0], [0, -1, 0]]) + shift
    y = x.copy()
    y[4, 2] += 0.25
    model, paths = mesh(x, y, [(0, 1, 2), (0, 3, 4)])
    exact, _ = query(model, paths, False)
    fast, stats = query(model, paths, True)
    assert exact.pairs == fast.pairs and fast.pairs[0].status == "adjacent"
    assert stats["sheet_pairs_adjacent"] == 1


def test_diagonal_gap_outside_skin_is_pruned_but_a_pair_at_the_margin_survives():
    a = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    b = np.array([[0.65, 0.65, 0], [1, 0.65, 0], [0.65, 1, 0]])
    model, paths = mesh(np.vstack([a, b]), None, [(0, 1, 2), (3, 4, 5)])
    fast, stats = query(model, paths, True, 0.125)
    assert (
        fast.pairs[0].status == "separated"
        and not fast.pairs[0].candidate
        and stats["sheet_pairs_far"] == 1
    )
    above, _ = query(model, paths, True, 0.25)
    exact, _ = query(model, paths, False, 0.25)
    assert above.pairs == exact.pairs and above.pairs[0].witness.proximity_patch is not None
    b = a + np.array([0.0, 0.0, 0.125])
    model, paths = mesh(np.vstack([a, b]), None, [(0, 1, 2), (3, 4, 5)])
    equal, _ = query(model, paths, True, 0.125)
    assert equal.pairs[0].candidate and equal.pairs[0].witness.gap_m == 0.125


def test_degenerate_interior_and_conflicting_material_path_use_original_geometry():
    x = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    y = x.copy()
    y[1, 0] = -1
    model, paths = mesh(x, y, [(0, 1, 2), (0, 3, 4)])
    exact, _ = query(model, paths, False)
    fast, _ = query(model, paths, True)
    assert exact.pairs == fast.pairs and fast.pairs[0].status in ("invalid_geometry", "unresolved")
    model, paths = mesh(x, None, [(0, 1, 2), (0, 3, 4)])
    key = model.shapes[-1].key
    moved = paths[key].start.copy()
    moved[0, 2] += 0.01
    paths[key] = ph.CollisionMotion(moved)
    for fast in (False, True):
        with pytest.raises(ValueError, match="shared material vertex has inconsistent motion"):
            query(model, paths, fast)


def test_sheet_filter_is_explicit_and_only_accepts_one_zero_thickness_sheet():
    sphere = ph.CollisionShape(ph.CollisionKey(0, 1, 0), "sphere", (0,), 1.0)
    with pytest.raises(ValueError, match="sheet"):
        ph.CollisionWorkspace(ph.CollisionModel([sphere]), sheet_bounds=True)
