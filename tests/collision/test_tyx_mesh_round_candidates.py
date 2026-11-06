"""Real TYX face traversal and public readback-failure transaction boundaries."""

from dataclasses import asdict

import pytest
from mesh_round_candidate_cases import FACE, build

import phymeo as ph
from phymeo.core.capabilities import SolverFailure


def test_public_face_readback_failure_is_atomic_and_closes_workspace(monkeypatch):
    """A backend validation error must not become a newly published narrow unknown."""
    model, paths, _, _ = build(FACE)
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        previous = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert previous.pairs[0].mesh_round.end.axis_gap_m == -1 / 32
        assert work.stats()["mesh_bvh_refits"] > 0
        before = asdict(previous)
        layouts = work._trees.layouts.copy()
        called = []

        def broken_readback(*args):
            called.append(args)
            raise SolverFailure("invalid BVH output")

        monkeypatch.setattr(work._trees, "facets", broken_readback)
        with pytest.raises(RuntimeError, match="face"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert called and work.last_result is previous
        assert asdict(previous) == before and work._trees.layouts == layouts
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
