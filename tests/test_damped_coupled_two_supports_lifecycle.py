"""Real dual-support capacity rollback and source-identity restore rejection."""

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest
from test_damped_coupled_mixed_area_friction import _observe, _prescribed
from test_damped_coupled_two_supports import KEY, H, _declaration, _drive, _source
from test_mujoco_area_friction_lifecycle import _evidence, _same

from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
from phymeo.backends.mujoco_runtime import MuJoCoControl
from phymeo.training.resident_stepper import PhysicsStepFailure


def _assert_two_supports(spec, before, after):
    """Establish genuine native rows from source input, not capacity metadata."""
    _model, data, _mass, pairs, sources = _source(spec, before)
    assert data.ncon == 2 and data.nefc == 8 and set(sources) == set(pairs)
    receipt = after.last_step
    assert len(receipt.contacts) == 2 and all(contact.active for contact in receipt.contacts)
    assert len(receipt.constraints) == 8
    native = receipt.raw_contact_fields["native_contacts"]
    np.testing.assert_array_equal(native["counts"], (2, 2, 8))
    np.testing.assert_array_equal(native["contact_map"][:2], ((0, 0, 4), (1, 4, 4)))
    assert tuple(contact.geometry_ids for contact in receipt.contacts) == pairs
    assert tuple(
        (row.constraint_type, row.source_id, row.side_or_edge, row.geometry_slot)
        for row in receipt.constraints
    ) == tuple((6, ci, edge, ci) for ci in range(2) for edge in range(4))
    # The two supports must remain distinguishable even on the same moving body.
    assert not np.array_equal(receipt.contacts[0].friction, receipt.contacts[1].friction)
    assert not np.array_equal(receipt.contacts[0].solref, receipt.contacts[1].solref)
    (area,) = receipt.raw_contact_fields["area_friction"]
    assert len(area["sample_ids"]) == 3 and area["normal_force"].sum() > 0
    assert after.tick == before.tick + 1


@pytest.mark.cuda
@pytest.mark.parametrize("empty", [True, False])
def test_separated_ground_pairs_keep_external_and_area_loads_distinct(empty):
    spec, world = _declaration(q=(-0.005, 0.0), v=(0.04, 0.15))
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        applied = np.array([2.0, -0.25])
        batch.set_controls({KEY: MuJoCoControl((), qfrc_applied=applied)})
        before = batch.integration_snapshot()[KEY]
        model, source, mass, _, contacts = _source(spec, before)
        assert not contacts
        surface = _prescribed(batch, KEY, empty=empty)
        assert batch.step_area_pressure(
            H, {KEY: (surface,)}, expected_revision=batch.checkpoint_revision
        ) == {KEY: True}
        after = batch.integration_snapshot()[KEY]
        step = after.last_step
        assert not step.contacts and not step.constraints
        (area,) = step.raw_contact_fields["area_friction"]
        solver = step.raw_contact_fields["area_friction_solver"]
        assert len(area["sample_ids"]) == (0 if empty else 3)
        area_load = np.einsum(
            "sij,si->j", area["jacobian_a"] - area["jacobian_b"], area["force_on_a"]
        )
        if empty:
            np.testing.assert_array_equal(area_load, np.zeros(2))
        else:
            assert np.linalg.norm(area_load) > 0.1
        np.testing.assert_allclose(
            solver["applied_with_contact"], applied + area_load, atol=2e-5, rtol=2e-5
        )
        rhs = applied + area_load + source.qfrc_passive - source.qfrc_bias
        expected = before.v + H * np.linalg.solve(mass + H * np.diag(model.dof_damping), rhs)
        np.testing.assert_allclose(after.v, expected, atol=2e-6, rtol=2e-5)


def _accepted_step(batch, spec):
    _drive(batch, spec, (0.04, 0.15))
    before = batch.integration_snapshot()[KEY]
    surface = _prescribed(batch, KEY)
    assert batch.step_area_pressure(
        H, {KEY: (surface,)}, expected_revision=batch.checkpoint_revision
    ) == {KEY: True}
    after = batch.integration_snapshot()[KEY]
    _assert_two_supports(spec, before, after)
    return before, after, surface


@pytest.mark.cuda
def test_eight_native_edges_and_area_overflow_roll_back_and_same_world_retries(tmp_path):
    spec, world = _declaration()
    spec = replace(spec, constraint_capacity=11)
    with _evidence(tmp_path / "two-support-capacity.json.gz") as evidence:
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:
                _accepted_step(batch, spec)
                _drive(batch, spec, (0.04, 0.15))
                before = batch.integration_snapshot()[KEY]
                # The next input still contains both physical contacts. Six area
                # samples fit the packet alone but 8+6 exceeds the common 11 rows.
                _model, source, _mass, _pairs, _contacts = _source(spec, before)
                assert source.ncon == 2 and source.nefc == 8
                frame, arrays = _observe(ws, batch, (spec,), {KEY: before})
                revision, publication = batch.checkpoint_revision, batch.last_step
                checkpoint = batch.checkpoint().to_bytes()
                oversized = _prescribed(batch, KEY, copies=2)
                evidence.update(before=before, arrays=arrays, oversized=oversized)
                with pytest.raises(PhysicsStepFailure):
                    batch.step_area_pressure(H, {KEY: (oversized,)}, expected_revision=revision)
                assert batch.checkpoint_revision == revision and batch.last_step is publication
                assert batch.checkpoint().to_bytes() == checkpoint
                _same(batch.integration_snapshot()[KEY], before)
                _same(ws.readback(frame), arrays)
                # Reuse the same world, device workspace, controls and revision.
                surface = _prescribed(batch, KEY)
                assert batch.step_area_pressure(
                    H, {KEY: (surface,)}, expected_revision=revision
                ) == {KEY: True}
                after = batch.integration_snapshot()[KEY]
                _assert_two_supports(spec, before, after)
                _observe(ws, batch, (spec,), {KEY: after})
                evidence["retry"] = after


@pytest.mark.cuda
@pytest.mark.parametrize("fault", ("slot", "compact-id", "net"))
def test_restore_rejects_swapped_support_identity_before_device_writes(monkeypatch, fault):
    from phymeo.backends import mujoco_checkpoint as codec

    spec, world = _declaration()
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        with batch.prepare_device_workspace(include_bodies=True, include_area_pressure=True) as ws:
            _before, accepted, _surface = _accepted_step(batch, spec)
            frame, arrays = _observe(ws, batch, (spec,), {KEY: accepted})
            owner = batch._buckets[batch.layout[KEY].bucket]
            capsule, owner_revision = owner.checkpoint(), owner.checkpoint_revision
            revision, publication = batch.checkpoint_revision, batch.last_step
            checkpoint = batch.checkpoint().to_bytes()
            state = dict(capsule._read(capsule.binding))
            saved = dict(state["worlds"][0])
            receipt = saved["receipt"]
            raw = dict(receipt.raw_contact_fields)
            typed = list(receipt.constraints)
            if fault == "net":
                solver = dict(raw["area_friction_solver"])
                original = solver["applied_with_contact"]
                altered = original.copy()
                # Keep the actual FP32 value represented in the original HOST
                # dtype so this reaches the physical sum gate, not a dtype gate.
                altered[0] = np.float32(altered[0] + 1.0)
                assert altered.dtype == original.dtype and altered[0] != original[0]
                np.testing.assert_array_equal(altered[1:], original[1:])
                solver["applied_with_contact"] = altered
                raw["area_friction_solver"] = MappingProxyType(solver)
            else:
                native = dict(raw["native_contacts"])
                ids = native["efc_ids"].copy()
                column, attribute = (3, "geometry_slot") if fault == "slot" else (1, "source_id")
                # Change all eight typed/raw row identities coherently. A generic
                # raw-versus-typed comparison alone cannot detect this forgery.
                ids[:8, column] = 1 - ids[:8, column]
                for index, row in enumerate(typed):
                    typed[index] = replace(row, **{attribute: int(ids[index, column])})
                native["efc_ids"] = ids
                if fault == "slot":
                    mapping = native["contact_map"].copy()
                    mapping[:2, 0] = 1 - mapping[:2, 0]
                    native["contact_map"] = mapping
                raw["native_contacts"] = MappingProxyType(native)
            saved["receipt"] = replace(
                receipt, constraints=tuple(typed), raw_contact_fields=MappingProxyType(raw)
            )
            state["worlds"] = (saved,)
            forged = codec._make_checkpoint(capsule.binding, capsule.source_revision, state)
            expected_error = {
                "slot": "mixed contact map does not cover exact active rows",
                "compact-id": "mixed rows omit, reorder or replace source constraints",
                "net": "physical load before cancellation",
            }[fault]
            with monkeypatch.context() as patch:
                patch.setattr(
                    owner,
                    "_build_device",
                    lambda *args, **kwargs: pytest.fail("forged support receipt reached device"),
                )
                with pytest.raises((ValueError, RuntimeError), match=expected_error):
                    owner.restore(forged, expected_revision=owner_revision)
            assert owner.checkpoint_revision == owner_revision
            assert owner.checkpoint().to_bytes() == capsule.to_bytes()
            assert batch.checkpoint_revision == revision and batch.last_step is publication
            assert batch.checkpoint().to_bytes() == checkpoint
            _same(batch.integration_snapshot()[KEY], accepted)
            _same(ws.readback(frame), arrays)
            _accepted_step(batch, spec)
