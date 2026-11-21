"""Release reference and real candidate rollback, without replaying old motion."""

from copy import deepcopy

import numpy as np
import pytest

from experiments.area_pressure.pusht_loaded_release import (
    attempt_release_candidate,
    release_reference,
)
from scripts.run_manipulation_control import reference

NODES = ((0.0, "withdraw", np.array([-0.2, 0.45])), (0.25, "withdraw", np.array([-0.24, 0.46])))


@pytest.mark.parametrize("elapsed", (0.0, 0.001, 0.125, 0.25, 0.3))
def test_baseline_is_the_existing_reference_without_new_gains_or_time_scale(elapsed):
    actual = release_reference(NODES, elapsed, mode="smooth")
    expected = reference(NODES, elapsed)
    for a, b in zip(actual[:2], expected[:2], strict=True):
        np.testing.assert_array_equal(a, b)
    assert actual[2] == expected[2] == "withdraw"


def test_single_fallback_keeps_the_same_path_and_duration_and_only_changes_time_parameter():
    q0, q1 = NODES[0][2], NODES[1][2]
    q, v, stage = release_reference(NODES, 0.0, mode="constant_rate")
    np.testing.assert_array_equal(q, q0)
    np.testing.assert_allclose(v, (q1 - q0) / 0.25, atol=0, rtol=0)
    assert stage == "withdraw"
    for elapsed in (0.001, 0.1, 0.2):
        q, v, _ = release_reference(NODES, elapsed, mode="constant_rate")
        np.testing.assert_array_equal(q, q0 + elapsed / 0.25 * (q1 - q0))
        np.testing.assert_array_equal(v, (q1 - q0) / 0.25)
    for elapsed in (0.25, 0.3):
        q, v, _ = release_reference(NODES, elapsed, mode="constant_rate")
        np.testing.assert_array_equal(q, q1)
        np.testing.assert_array_equal(v, np.zeros_like(v))


@pytest.mark.parametrize(
    "nodes,elapsed,mode",
    (
        (NODES, -0.001, "smooth"),
        (NODES, float("nan"), "smooth"),
        (NODES, 0.0, "adaptive"),
        ((NODES[0], (0.0, "withdraw", NODES[1][2])), 0.0, "smooth"),
        ((NODES[0], (0.25, "push", NODES[1][2])), 0.0, "smooth"),
        (NODES + ((0.5, "approach", np.zeros(2)),), 0.0, "smooth"),
    ),
)
def test_release_reference_rejects_hidden_replan_or_invalid_domain(nodes, elapsed, mode):
    with pytest.raises((ValueError, TypeError)):
        release_reference(nodes, elapsed, mode=mode)


class _Batch:
    def __init__(self):
        self.checkpoint_revision = 0
        self.state = {6: dict(q=0.0, v=1.0, tick=0, held_control=None)}

    def integration_snapshot(self):
        return deepcopy(self.state)

    def set_controls(self, controls):
        self.state[6]["held_control"] = deepcopy(controls[6])
        self.checkpoint_revision += 1


class _Scene:
    def __init__(self, batch):
        self.batch = batch
        self.last_geometry = {6: "previous-geometry"}
        self.calls = 0

    def checkpoint(self):
        return deepcopy((self.batch.state, self.last_geometry))

    def restore(self, capsule, *, expected_revision):
        assert expected_revision == self.batch.checkpoint_revision
        self.batch.state, self.last_geometry = deepcopy(capsule)
        self.batch.checkpoint_revision += 1

    def step(self, h, *, expected_revision):
        assert h == 0.001 and expected_revision == self.batch.checkpoint_revision
        self.calls += 1
        self.batch.state[6]["q"] += h
        self.batch.state[6]["tick"] += 1
        self.batch.checkpoint_revision += 1
        self.last_geometry = {6: "candidate-geometry"}
        return {6: True}


@pytest.mark.parametrize("safe", (True, False))
def test_candidate_selection_retains_or_restores_but_never_erases_the_attempt(safe):
    batch, controls = _Batch(), {6: "candidate-control"}
    scene = _Scene(batch)
    previous = batch.integration_snapshot()
    observed = []

    def assess(before, after, geometry):
        observed.append((before, after, geometry))
        assert before[6]["q"] == 0.0 and after[6]["q"] == 0.001
        assert geometry == {6: "candidate-geometry"}
        return dict(safe=safe, reason="allowed" if safe else "load_increased")

    result = attempt_release_candidate(scene, batch, controls, assess)
    assert scene.calls == 1 and len(observed) == 1
    assert result["physics_accepted"] is True
    assert result["selected"] is safe and result["restored"] is (not safe)
    assert result["assessment"]["reason"] == ("allowed" if safe else "load_increased")
    assert result["candidate_output"][6]["tick"] == 1
    if safe:
        assert batch.integration_snapshot()[6]["tick"] == 1
    else:
        assert batch.integration_snapshot() == previous
        assert scene.last_geometry == {6: "previous-geometry"}


@pytest.mark.cuda
def test_real_loaded_public_candidate_can_be_recorded_then_restored_without_second_step():
    from test_mujoco_area_friction_lifecycle import _same
    from test_pusht_public_mixed_area import KEYS, _asset, _controls, _declaration

    from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaPressureScene,
    )

    model, spec, worlds, entries = _declaration()
    key = KEYS[2]
    world = next(world for world in worlds if world.world == key)
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (_asset(model),)})
        before = batch.integration_snapshot()
        observed = []

        def reject_real_candidate(initial, candidate, geometry):
            receipt = candidate[key].last_step
            (pair,) = receipt.raw_contact_fields["area_friction"]
            assert candidate[key].tick == initial[key].tick + 1
            np.testing.assert_array_equal(receipt.input_q, initial[key].q)
            np.testing.assert_array_equal(receipt.input_v, initial[key].v)
            assert len(pair["sample_ids"]) > 0 and pair["normal_force"].sum() > 0
            np.testing.assert_array_equal(pair["force_on_b"], -pair["force_on_a"])
            assert receipt.constraints and receipt.contacts and geometry[key]
            observed.append(receipt)
            # Explicit test selection rejection, not a claim that this real
            # candidate violates the future release policy.
            return dict(safe=False, reason="test_selection_rejection")

        result = attempt_release_candidate(
            scene, batch, {key: _controls(entries)[key]}, reject_real_candidate
        )
        assert len(observed) == 1
        assert result["physics_accepted"] and not result["selected"] and result["restored"]
        assert result["rollback_verification"] == dict(passed=True, worlds=[key], mismatches=[])
        assert result["candidate_output"][key].last_step is observed[0]
        _same(batch.integration_snapshot(), before, fresh_stamp=True)
        assert not scene.last_geometry


def test_evidence_error_restores_and_exposes_the_real_accepted_trial():
    batch = _Batch()
    scene = _Scene(batch)
    before = batch.integration_snapshot()

    def invalid_evidence(initial, candidate, geometry):
        raise ValueError("missing declared pair evidence")

    with pytest.raises(ValueError, match="missing declared pair") as caught:
        attempt_release_candidate(scene, batch, {6: "new-held-control"}, invalid_evidence)
    assert scene.calls == 1 and batch.integration_snapshot() == before
    trial = caught.value.release_trial
    assert trial["physics_accepted"] is True and trial["restored"] is True
    assert trial["output"][6]["tick"] == 1
    assert trial["geometry"] == {6: "candidate-geometry"}
    assert trial["input"] == before


def test_physics_error_restores_the_outer_held_control_without_claiming_acceptance():
    batch = _Batch()

    class RejectingScene(_Scene):
        def step(self, h, *, expected_revision):
            self.calls += 1
            raise ValueError("physical solve rejected")

    scene = RejectingScene(batch)
    before = batch.integration_snapshot()
    with pytest.raises(ValueError, match="physical solve rejected") as caught:
        attempt_release_candidate(
            scene, batch, {6: "candidate"}, lambda *args: pytest.fail("no receipt")
        )
    assert scene.calls == 1 and batch.integration_snapshot() == before
    assert caught.value.release_trial["physics_accepted"] is False
    assert caught.value.release_trial["restored"] is True
    assert caught.value.release_trial["output"] is None


def test_restore_is_read_back_before_a_rejected_candidate_is_called_rolled_back():
    batch = _Batch()

    class BrokenRestore(_Scene):
        def restore(self, capsule, *, expected_revision):
            super().restore(capsule, expected_revision=expected_revision)
            self.batch.state[6]["v"] = 99.0

    scene = BrokenRestore(batch)
    with pytest.raises(RuntimeError, match="rollback readback") as caught:
        attempt_release_candidate(
            scene, batch, {6: "candidate"}, lambda *args: dict(safe=False, reason="test")
        )
    failure = caught.value.release_trial
    assert failure["physics_accepted"] is True
    assert failure["restored"] is False
    assert failure["rollback_verification"]["passed"] is False
    assert "6.v" in failure["rollback_verification"]["mismatches"]


def test_successful_restore_reports_the_actual_readback_guard():
    batch = _Batch()
    result = attempt_release_candidate(
        _Scene(batch), batch, {6: "candidate"}, lambda *args: dict(safe=False, reason="test")
    )
    assert result["rollback_verification"] == dict(passed=True, worlds=[6], mismatches=[])
