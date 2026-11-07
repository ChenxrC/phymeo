"""Capability facts must not confuse geometry, owners, or execution backends."""

from dataclasses import FrozenInstanceError
from itertools import combinations_with_replacement

import pytest
from convex_family_cases import KINDS, PAIRS, convex_case

from phymeo.geometry.collision_capabilities import (
    CollisionCapability,
    collision_capabilities,
    get_collision_capability,
)


def test_convex_matrix_covers_every_operation_without_duplicate_keys():
    rows = collision_capabilities()
    keys = [(r.first_kind, r.second_kind, r.operation, r.owner_mode, r.backend) for r in rows]
    assert len(keys) == len(set(keys))
    for a, b in combinations_with_replacement(KINDS, 2):
        for operation in ("static", "manifold", "translation_ccd", "rotation_ccd", "response"):
            row = get_collision_capability(a, b, operation)
            assert row == get_collision_capability(b, a, operation)
            assert row in rows
            assert row.domain
    assert any(r.first_kind == "mesh" or r.second_kind == "mesh" for r in rows)
    assert any(r.first_kind == "triangle" or r.second_kind == "triangle" for r in rows)


def test_verified_rows_bind_existing_source_and_evidence():
    for row in collision_capabilities():
        if row.status == "verified":
            assert any(item.startswith("source:") for item in row.evidence)
            assert any(item.startswith("docs/") for item in row.evidence)
        if row.backend.startswith("hybrid"):
            assert "host" in row.route.lower()
    row = get_collision_capability("box", "box", "response", owner_mode="robot_robot")
    assert row.status == "missing"  # The new adapter has not passed yet.
    with pytest.raises(FrozenInstanceError):
        row.status = "verified"


def test_unknown_scope_does_not_silently_inherit_host_or_geometry_support():
    for kwargs in ({}, {"backend": "cuda"}, {"owner_mode": "arbitrary_robot"}):
        row = get_collision_capability("unregistered", "sphere", "response", **kwargs)
        assert row.status == "missing"
        assert not row.evidence
    assert get_collision_capability("box", "hull", "static", backend="cuda").status == "missing"


@pytest.mark.parametrize("status", ["all_green", "", "gpu"])
def test_invalid_registration_status_is_rejected(status):
    with pytest.raises(ValueError):
        CollisionCapability(
            "sphere", "box", "static", "geometry", "host", status, "route", "domain", ()
        )


def test_verified_without_revision_or_evidence_is_rejected():
    with pytest.raises(ValueError):
        CollisionCapability(
            "sphere", "box", "static", "geometry", "host", "verified", "route", "domain", ()
        )


@pytest.mark.parametrize("kinds", PAIRS)
@pytest.mark.parametrize("gap", [0.02, 0.0, -0.01])
def test_input_library_preserves_registered_materials_and_independent_gap(kinds, gap):
    shapes, paths = convex_case(*kinds, gap)
    a, b = shapes
    assert a.key.world == b.key.world and a.key.body != b.key.body
    assert set(a.vertex_ids).isdisjoint(b.vertex_ids)
    assert all(paths[s.key].start.shape == (len(s.vertex_ids), 3) for s in shapes)
    first_top = paths[a.key].start[:, 2].max() + a.radius
    second_bottom = paths[b.key].start[:, 2].min() - b.radius
    assert second_bottom - first_top == pytest.approx(gap, abs=1e-15)
    for kind, shape in zip(kinds, shapes, strict=True):
        if kind == "hull":
            assert len(shape.vertex_ids) == 10  # Non-box pentagonal prism.
