"""Exact ownership, tie and continuation controls with explicit geometry."""

from dataclasses import replace

import numpy as np
import pytest
from test_cloth_self_response import BUDGETS, LIMITS

from phymeo.core.capabilities import SolverFailure
from phymeo.geometry.cloth_self_anchors import (
    make_self_contact_anchors,
    require_area_path,
    require_fixed_support,
)
from phymeo.geometry.prism_prism_manifold import PrismPairIdentity, prism_prism_manifold


def contact(shift=(0.0, 0.0, 0.0), swap=False, cycle=False):
    a = np.array([[0.0, 0.0, 0.25], [2.0, 0.0, 0.25], [0.0, 2.0, 0.25]])
    b = a - np.array([0.0, 0.0, 0.5]) + np.array(shift)
    nodes = np.arange(6, dtype=np.int64).reshape(2, 3)
    faces = np.array([11, 29], np.int64)
    if cycle:
        a, b, nodes = np.roll(a, 1, axis=0), np.roll(b, 1, axis=0), np.roll(nodes, 1, axis=1)
    if swap:
        a, b, nodes, faces = b, a, nodes[::-1].copy(), faces[::-1].copy()
    identity = PrismPairIdentity("cloth", faces, nodes)
    manifold = prism_prism_manifold(
        a, b, identity=identity, half_thickness_m=np.array([0.25, 0.25])
    )
    anchors = make_self_contact_anchors(
        (manifold,), identities=(identity,), history={}, limits=LIMITS, budgets=BUDGETS
    )
    return np.vstack((a, b)), manifold, identity, anchors


def test_complete_extrema_owned_dyadic_counts_and_bits():
    _, manifold, _, anchors = contact()
    assert len(anchors.owner_tokens) == 3
    assert sum(len(axis.points) for axis in manifold.maximal_axes) >= 3
    np.testing.assert_array_equal(anchors.counts.sum(axis=1), np.full(6, 2**52, np.int64))
    with pytest.raises(ValueError):
        anchors.counts.flags.writeable = True


def test_canonical_exchange_and_cyclic_node_order_preserve_owners():
    ordinary = contact()[3]
    for swapped, cycle in ((True, False), (False, True), (True, True)):
        changed = contact(swap=swapped, cycle=cycle)[3]
        assert set(changed.owner_tokens) == set(ordinary.owner_tokens)


def test_loaded_noncollinear_point_tie_rejects_without_first_axis():
    with pytest.raises(SolverFailure, match="unique oriented ray"):
        contact(shift=(2.0, 0.0, 0.0))


def test_duplicate_pair_and_reversed_identity_reject():
    _, manifold, identity, _ = contact()
    with pytest.raises(ValueError, match="duplicate"):
        make_self_contact_anchors(
            (manifold, manifold),
            identities=(identity, identity),
            history={},
            limits=LIMITS,
            budgets=BUDGETS,
        )
    forged = PrismPairIdentity("cloth", identity.face_ids, identity.node_ids[:, ::-1].copy())
    with pytest.raises(ValueError, match="identity"):
        make_self_contact_anchors(
            (manifold,), identities=(forged,), history={}, limits=LIMITS, budgets=BUDGETS
        )


def test_unknown_and_exhausted_clipping_do_not_allocate_capacity():
    _, manifold, identity, _ = contact()
    with pytest.raises(SolverFailure, match="unresolved"):
        make_self_contact_anchors(
            (replace(manifold, status="unresolved"),),
            identities=(identity,),
            history={},
            limits=LIMITS,
            budgets=BUDGETS,
        )
    with pytest.raises(SolverFailure, match="budget"):
        make_self_contact_anchors(
            (manifold,),
            identities=(identity,),
            history={},
            limits=replace(LIMITS, max_anchors_per_pair=2),
            budgets=BUDGETS,
        )


def test_fixed_normal_support_continuation_and_rotating_negative():
    positions, _, _, anchors = contact()
    require_fixed_support(anchors, positions, positions.copy(), np.ones(3, bool))
    rotated = positions.copy()
    rotated[1, 2] += 0.125
    with pytest.raises(SolverFailure, match="rotating"):
        require_fixed_support(anchors, positions, rotated, np.ones(3, bool))


def test_interior_area_collapse_is_not_endpoint_proof():
    start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    end = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    with pytest.raises(SolverFailure, match="area"):
        require_area_path(start, end)
