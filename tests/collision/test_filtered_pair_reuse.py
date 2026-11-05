"""Complete ordinary detect records with reusable immutable filter rows."""

from dataclasses import FrozenInstanceError
from itertools import combinations

import numpy as np
import pytest

import phymeo as ph
import phymeo.geometry.collision_pipeline as pipeline


def _filter_scene():
    """One real touching pair, an AABB miss, and all four filter priorities."""
    keys = tuple(
        ph.CollisionKey(*key)
        for key in ((0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 3, 0), (0, 4, 0), (0, 5, 0), (1, 1, 0))
    )
    a, same, mask, excluded, touching, far, other_world = keys
    shapes = [
        ph.CollisionShape(key, "sphere", (7,), 0.125, group=1, mask=1) for key in keys
    ]
    shapes[1] = ph.CollisionShape(same, "sphere", (7,), 0.125, group=1, mask=0)
    shapes[2] = ph.CollisionShape(mask, "sphere", (7,), 0.125, group=2, mask=2)
    model = ph.CollisionModel(
        tuple(reversed(shapes)),
        ((a, same), (a, mask), (a, excluded), (a, other_world)),
    )
    centers = (0, 4, 2, 6, 0.25, 8, 0)
    paths = {key: ph.CollisionMotion([[x, 0, 0]]) for key, x in zip(keys, centers, strict=True)}
    return model, paths, keys


def _by_pair(result):
    return {(row.first, row.second): row for row in result.pairs}


def _assert_filter_scene(result, keys):
    a, same, mask, excluded, touching, far, other_world = keys
    assert result.status == "ok" and result.required_candidates == 1
    assert [(row.first, row.second) for row in result.pairs] == list(combinations(keys, 2))
    rows = _by_pair(result)
    for second, reason in (
        (other_world, "different_world"),
        (same, "same_rigid_body"),
        (mask, "mask"),
        (excluded, "excluded"),
    ):
        row = rows[a, second]
        assert (row.candidate, row.status, row.reason, row.family) == (
            False, "filtered", reason, "RR"
        )
        assert row.witness is None and row.sweep is None and not row.witness_shapes
    miss = rows[a, far]
    assert (miss.candidate, miss.status, miss.reason) == (False, "separated", "aabb")
    contact = rows[a, touching]
    assert contact.candidate and contact.status == "touching"
    assert contact.witness.gap_m == 0
    assert contact.witness.first_weights == ((7, 1.0),)
    assert contact.witness.second_weights == ((7, 1.0),)
    np.testing.assert_array_equal(contact.witness.first_point, (0.125, 0, 0))
    np.testing.assert_array_equal(contact.witness.second_point, (0.125, 0, 0))
    np.testing.assert_array_equal(contact.witness.normal, (1, 0, 0))


def test_repeat_public_detect_does_not_dispatch_invariant_filtered_rows(monkeypatch):
    """RED on the old path: its second public query redispatches codes 1--4."""
    model, paths, keys = _filter_scene()
    original_dispatch, original_pair = pipeline._detect_pairs, pipeline._pair
    dispatched, constructed = [], []

    def dispatch(*args, **kwargs):
        dispatched.extend(int(code) for code in args[2])
        return original_dispatch(*args, **kwargs)

    def pair(*args, **kwargs):
        constructed.append(int(args[4]))
        return original_pair(*args, **kwargs)

    with ph.CollisionWorkspace(model) as work:
        before = ph.detect(model, paths, ph.CollisionPolicy(), work)
        _assert_filter_scene(before, keys)
        saved_pairs = before.pairs
        monkeypatch.setattr(pipeline, "_detect_pairs", dispatch)
        monkeypatch.setattr(pipeline, "_pair", pair)
        after = ph.detect(model, paths, ph.CollisionPolicy(), work)
        _assert_filter_scene(after, keys)
        assert after == before and before.pairs is saved_pairs
        assert work.last_result is after
        assert 0 in dispatched and 5 in dispatched
        repeated = [code for code in dispatched if 1 <= code <= 4]
        assert not repeated, f"invariant filter rows still reached narrow dispatch: {repeated}"
        assert not [code for code in constructed if 1 <= code <= 4]
        for old, new in zip(before.pairs, after.pairs, strict=True):
            if old.status == "filtered":
                assert old is new
                with pytest.raises(FrozenInstanceError):
                    new.reason = "changed"