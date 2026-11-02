"""Physical and geometry publications share one commit boundary."""

import numpy as np
import pytest
from load_cases import scene

import phymeo as ph


def test_rejected_physical_path_retains_caller_visible_geometry_and_state():
    system, state, models, constraints = scene("RD")
    solvers = ph.XPBDSolver(constraints), ph.RigidSemiImplicitSolver()
    loads = ph.CoupledInputs.zeros(3, 1)
    with ph.CollisionWorkspace(system.model, broad_phase="bvh") as ws:
        good = system.step(state, loads, models, solvers, 1 / 64, ws)
        assert ws.last_result is good.path
        state.rigid.position[0, 2] = 0.5
        state.rigid.velocity[0, 2] = -64.0
        before = state.rigid.position.copy()
        with pytest.raises(ph.SolverFailure, match="core"):
            system.step(state, loads, models, solvers, 1 / 64, ws)
        assert ws.last_result is good.path
        assert system.last_frame is good
        np.testing.assert_array_equal(state.rigid.position, before)


def test_nested_query_publication_preserves_outer_commit_and_does_not_refund_work():
    system, state, _, _ = scene("RD")
    motions = system._paths(state)
    with ph.CollisionWorkspace(system.model, broad_phase="bvh") as ws:
        with pytest.raises(RuntimeError, match="abort outer"):
            with ws.transaction():
                ph.detect(system.model, motions, ph.CollisionPolicy(margin=0.2), ws)
                assert ws.last_result is None
                raise RuntimeError("abort outer")
        assert ws.last_result is None
        assert ws.stats()["shape_bvh_builds"] == 1
        with ws.transaction():
            keep = ph.detect(system.model, motions, ph.CollisionPolicy(margin=0.2), ws)
            with pytest.raises(RuntimeError, match="abort inner"):
                with ws.transaction():
                    ph.detect(system.model, motions, ph.CollisionPolicy(), ws)
                    raise RuntimeError("abort inner")
            assert ws.last_result is None
        assert ws.last_result is keep
        assert ws.stats()["shape_bvh_builds"] == 2
        latest = ph.detect(system.model, motions, ph.CollisionPolicy(), ws)
        assert ws.last_result is latest


def test_outer_rejection_keeps_both_prior_publications_even_after_an_inner_step_completes():
    system, state, models, constraints = scene("RD")
    solvers = ph.XPBDSolver(constraints), ph.RigidSemiImplicitSolver()
    inputs = ph.CoupledInputs.zeros(3, 1)
    with ph.CollisionWorkspace(system.model) as workspace:
        accepted = system.step(state, inputs, models, solvers, 1 / 64, workspace)
        with pytest.raises(ValueError, match="reject group"):
            with workspace.transaction():
                candidate = system.step(accepted.state, inputs, models, solvers, 1 / 64, workspace)
                assert candidate is not accepted
                assert system.last_frame is accepted
                assert workspace.last_result is accepted.path
                raise ValueError("reject group")
        assert system.last_frame is accepted
        assert workspace.last_result is accepted.path
        with workspace.transaction():
            next_frame = system.step(accepted.state, inputs, models, solvers, 1 / 64, workspace)
            assert system.last_frame is accepted
        assert system.last_frame is next_frame
        assert workspace.last_result is next_frame.path


def test_failed_standalone_force_observation_does_not_publish_partial_geometry():
    system, state, _, _ = scene("RD")
    with ph.CollisionWorkspace(system.model) as workspace:
        previous = ph.detect(system.model, system._paths(state), ph.CollisionPolicy(), workspace)
        state.rigid.position[0, 2] = 0
        with pytest.raises(ph.SolverFailure, match="core"):
            system.sample(state, workspace)
        assert workspace.last_result is previous
        assert system.last_frame is None
