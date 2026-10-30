"""Actual represented execution controls; complete frozen windows stay in place."""

from dataclasses import replace
from fractions import Fraction as F
from math import atan2, sqrt

import numpy as np
import pytest
from test_composite_cloth import BUDGETS, case, program_for, settings_for, settled_workspace

from phymeo.constraints.cloth_material import ClothMaterial
from phymeo.constraints.material_representation import (
    NativeCompositeRepresentation,
    deserialize_evidence,
    serialize_evidence,
)
from phymeo.core.capabilities import SolverFailure
from phymeo.core.model import ParticleModel
from phymeo.core.scene import fingerprint
from phymeo.core.state import ParticleState, StepInputs
from phymeo.validation.material_representation import projection_charge

pytestmark = pytest.mark.tyx
REPRESENTATION = NativeCompositeRepresentation()


def _workspace(mode, name="A", h=1 / 8, **kwargs):
    model, material, state, inputs = case(name, **kwargs)
    return program_for(model, material, mode, representation=REPRESENTATION).workspace(
        "cloth", state, inputs, h, settings_for(mode)
    )


def test_actual_represented_unit_hinge_literal_first_sweep(tyx_mode):
    ws = _workspace(tyx_mode)
    try:
        ws.predict()
        ws.position_sweep()
        ws.prepare_velocity()
        batch = ws.frames[0].blocks[1]
        assert batch.rows.dtype == np.float32 and batch.rows.shape == (1, 45)
        assert batch.launches == batch.completed_launches == 1
        np.testing.assert_array_equal(batch.rows[0, :12], np.zeros(12))
        assert batch.rows[0, 42] == pytest.approx(1 / 56, abs=2e-8)
        expected = ws.initial.position.copy()
        expected[:, 2] = (1 / 28, 0.0, -1 / 56, -1 / 56)
        np.testing.assert_allclose(ws.candidate.position, expected, rtol=0, atol=2e-7)
        theta = atan2(6 / 56 * sqrt(1 + 4 / 56**2), 1 - 5 / 56**2)
        assert batch.rows[0, 37] == pytest.approx(theta, abs=2e-6)
        assert np.sum(ws.candidate.velocity**2) / 2 == pytest.approx(3 / 49, abs=2e-7)
        assert ws.position_metrics().max_residual > 2e-5
        assert (
            ws.prediction_operation.completed_launches
            == ws.reconstruction_operation.completed_launches
            == 1
        )
        assert len(ws.alpha_operations) == 2
        with pytest.raises(RuntimeError, match="already"):
            ws.prepare_velocity()
    finally:
        ws.close()


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("shift", [0.0, 1024.0])
def test_actual_small_displacement_translation_and_scale(tyx_mode, scale, shift):
    model = ParticleModel(np.ones(1), np.zeros(3))
    initial = ParticleState(
        np.array([[scale + shift, 0.0, 0.0]]), np.array([[scale * 2**-25, 0.0, 0.0]])
    )
    program = program_for(model, ClothMaterial(), tyx_mode, representation=REPRESENTATION)
    ws = program.workspace("cloth", initial, StepInputs.zeros(1), 1.0, settings_for(tyx_mode))
    try:
        reference_bytes = ws.reference.tobytes()
        ws.predict()
        ws.position_sweep()
        ws.prepare_velocity()
        data = ws.diagnostics()
        assert F(ws.candidate.position[0, 0]) == F(scale) * (1 + F(1, 2**25)) + F(shift)
        assert ws.displacement[0, 0] == scale * 2**-25
        assert ws.reference.tobytes() == reference_bytes
        assert not ws.reference.flags.writeable
        assert data["material_audit"]["cumulative_projection_error"] == 0
        assert [b.completed_launches for b in ws.frames[0].blocks] == [0, 0]
    finally:
        ws.close()


def test_actual_decoder_loss_rejects_without_host_world_repair(tyx_mode):
    model = ParticleModel(np.ones(1), np.zeros(3))
    initial = ParticleState(np.array([[2.0**100, 0, 0]]), np.array([[1.0, 0, 0]]))
    ws = program_for(model, ClothMaterial(), tyx_mode, representation=REPRESENTATION).workspace(
        "cloth", initial, StepInputs.zeros(1), 1.0, settings_for(tyx_mode)
    )
    try:
        ws.predict()
        assert ws.displacement[0, 0] == 1
        assert ws.candidate.position[0, 0] == initial.position[0, 0]
        ws.position_sweep()
        ws.prepare_velocity()
        with pytest.raises(SolverFailure, match="equation"):
            ws.result()
        assert initial.position[0, 0] == 2.0**100
    finally:
        ws.close()


def test_default_v1_preserves_original_interpret_rejection(tyx_mode):
    if tyx_mode != "interpret":
        pytest.skip("original fixed trace receipt was interpret; no cross-mode impossibility claim")
    model, material, state, inputs = case("B")
    program = program_for(model, material, tyx_mode)
    assert program.representation is None and "representation" not in program.problem_signature()
    ws = program.workspace("cloth", state, inputs, 1 / 128, settings_for(tyx_mode))
    try:
        ws.predict()
        before = fingerprint((ws.candidate, ws.multipliers, tuple(ws.frames)))
        with pytest.raises(SolverFailure, match="cumulative projection/conversion"):
            ws.position_sweep()
        assert fingerprint((ws.candidate, ws.multipliers, tuple(ws.frames))) == before
    finally:
        ws.close()


def test_actual_represented_path_forbids_host_solver_and_reconstruction(tyx_mode, monkeypatch):
    import phymeo.constraints.bending as bending
    import phymeo.constraints.distance as distance
    import phymeo.solvers.composite_cloth as host

    def forbidden(*args, **kwargs):
        raise AssertionError("host solve/gradient/reconstruction is forbidden")

    monkeypatch.setattr(bending, "project_bending", forbidden)
    monkeypatch.setattr(bending, "bending_angle_gradient", forbidden)
    monkeypatch.setattr(distance, "project_distances", forbidden)
    monkeypatch.setattr(host, "project_bending", forbidden)
    monkeypatch.setattr(host, "project_distances", forbidden)
    monkeypatch.setattr(host.CompositeClothWorkspace, "_reconstruct", forbidden)
    ws = _workspace(tyx_mode, "B", h=1 / 128)
    try:
        ws.predict()
        ws.position_sweep()
        ws.prepare_velocity()
        assert all(b.completed_launches == 1 for b in ws.frames[0].blocks)
    finally:
        ws.close()


@pytest.mark.parametrize("phase", ["bending", "audit"])
def test_actual_late_failure_rolls_back_encoded_state_and_retry(tyx_mode, monkeypatch, phase):
    ws = _workspace(tyx_mode, "B", h=1 / 128)
    fresh = _workspace(tyx_mode, "B", h=1 / 128)
    try:
        ws.predict()
        before = fingerprint(
            (
                ws.reference,
                ws.displacement,
                ws.predicted_displacement,
                ws.candidate,
                ws.multipliers,
                tuple(ws.frames),
            )
        )
        target, name = (
            (ws.engine, "project_bending") if phase == "bending" else (ws.ledger, "audit_sweep")
        )
        original = getattr(target, name)

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise SolverFailure("late represented failure")

        with monkeypatch.context() as patch:
            patch.setattr(target, name, fail)
            with pytest.raises(SolverFailure, match="late represented"):
                ws.position_sweep()
        assert (
            fingerprint(
                (
                    ws.reference,
                    ws.displacement,
                    ws.predicted_displacement,
                    ws.candidate,
                    ws.multipliers,
                    tuple(ws.frames),
                )
            )
            == before
        )
        ws.position_sweep()
        fresh.predict()
        fresh.position_sweep()
        assert fingerprint((ws.displacement, ws.multipliers, tuple(ws.frames))) == fingerprint(
            (fresh.displacement, fresh.multipliers, tuple(fresh.frames))
        )
    finally:
        ws.close()
        fresh.close()


def test_actual_represented_hidden_mutation_and_raw_wire_round_trip(tyx_mode):
    ws = settled_workspace("B", mode=tyx_mode, representation=REPRESENTATION, h=1 / 128)
    try:
        diagnostic = ws.diagnostics()
        path = diagnostic["material_path"]
        assert fingerprint(deserialize_evidence(diagnostic["represented_wire"])) == fingerprint(
            path
        )
        restored = deserialize_evidence(serialize_evidence(path))
        assert fingerprint(path) == fingerprint(restored)
        original = ws.displacement.copy()
        ws.displacement[0, 0] += np.float32(0.125)
        with pytest.raises(SolverFailure):
            ws.diagnostics()
        ws.displacement[:] = original
        assert fingerprint(ws.diagnostics()["material_path"]) == fingerprint(path)
    finally:
        ws.close()


def test_actual_represented_bounded_mixed_sweep_and_reconstruction(tyx_mode):
    """One bounded complete mixed sweep, also the explicit future cpu_cpp node."""
    ws = _workspace(tyx_mode, "B", h=1 / 128)
    try:
        ws.predict()
        ws.position_sweep()
        ws.prepare_velocity()
        assert (
            ws.prediction_operation.completed_launches
            == ws.reconstruction_operation.completed_launches
            == 1
        )
        assert len(ws.frames) == 1 and len(ws.alpha_operations) == 4
        for batch in ws.frames[0].blocks:
            assert batch.completed_launches == 1 and not np.any(batch.status)
            arity = batch.node_indices.shape[1]
            size = 3 * arity
            for k, nodes in enumerate(batch.node_indices):
                raw = batch.rows[k]
                delta = raw[17] if batch.kind == "distance" else raw[42]
                gradients = (
                    np.array([raw[14:17], -raw[14:17]])
                    if batch.kind == "distance"
                    else raw[24:36].reshape((4, 3))
                )
                for local, node in enumerate(nodes):
                    for c in range(3):
                        before = raw[3 * local + c]
                        after = raw[size + 3 * local + c]
                        charge, direct, terms = projection_charge(
                            ws.reference[node, c],
                            before,
                            after,
                            batch.products[k, local, c],
                            ws.model.inverse_mass[node],
                            delta,
                            gradients[local, c],
                        )
                        independent = abs(
                            F(float(after))
                            - F(float(before))
                            - F(float(batch.products[k, local, c]))
                        )
                        assert charge >= independent and charge >= direct
                        assert len(terms) == 4
        expected_v = ws.predicted.velocity + (ws.candidate.position - ws.predicted.position) / ws.h
        np.testing.assert_allclose(ws.candidate.velocity, expected_v, rtol=0, atol=2e-5)
        assert ws.engine.stats()["mode"] == tyx_mode
        if tyx_mode == "cuda":
            assert ws.engine.stats()["uploads"] > 0 and ws.engine.stats()["downloads"] > 0
        before = fingerprint((ws.candidate, ws.displacement, ws.multipliers, tuple(ws.frames)))
        broken_block = replace(ws.frames[0].blocks[1], reference_fingerprint="stale")
        broken_sweep = replace(ws.frames[0], blocks=(ws.frames[0].blocks[0], broken_block))
        with pytest.raises(SolverFailure, match="reference"):
            ws.ledger.audit_sweep(
                broken_sweep,
                ws.predicted.position,
                {name: np.zeros(ws.material.count(name)) for name in ws.material.order},
                ws.alphas,
                displacement=ws.predicted_displacement,
            )
        assert (
            fingerprint((ws.candidate, ws.displacement, ws.multipliers, tuple(ws.frames))) == before
        )
        # A completed nonconverged first sweep is not a physical time-step pass.
        with pytest.raises(RuntimeError, match="after reconstruction"):
            ws.position_sweep()
    finally:
        ws.close()


def test_actual_second_hinge_native_failure_retains_raw_receipt_and_rolls_back(tyx_mode):
    ws = _workspace(tyx_mode, copies=2)
    try:
        ws.predict()
        before = fingerprint((ws.candidate, ws.displacement, ws.multipliers, tuple(ws.frames)))
        original = ws.native_rest["bending"].copy()
        ws.native_rest["bending"][1] = 4.0
        with pytest.raises(SolverFailure, match="row rejected"):
            ws.position_sweep()
        assert (
            fingerprint((ws.candidate, ws.displacement, ws.multipliers, tuple(ws.frames))) == before
        )
        assert ws.engine.failed_operations
        receipt = ws.engine.failed_operations[-1]
        assert receipt.completed_launches == 1
        assert receipt.outputs[4][0] == 2
        assert receipt.outputs[3][1] != 0
        ws.native_rest["bending"][:] = original
        ws.position_sweep()
        assert len(ws.frames) == 1
    finally:
        ws.close()


def test_actual_reconstruction_failure_keeps_hidden_publication_atomic(tyx_mode, monkeypatch):
    ws = _workspace(tyx_mode)
    try:
        ws.predict()
        ws.position_sweep()
        before = fingerprint(
            (
                ws.candidate,
                ws.displacement,
                ws.predicted_displacement,
                ws.reconstructed_velocity,
                ws.reconstruction_operation,
                ws.velocity_ready,
            )
        )
        original = ws.engine.reconstruct_displacement

        def late(*args, **kwargs):
            original(*args, **kwargs)
            raise SolverFailure("late native reconstruction")

        with monkeypatch.context() as patch:
            patch.setattr(ws.engine, "reconstruct_displacement", late)
            with pytest.raises(SolverFailure, match="late native reconstruction"):
                ws.prepare_velocity()
        assert (
            fingerprint(
                (
                    ws.candidate,
                    ws.displacement,
                    ws.predicted_displacement,
                    ws.reconstructed_velocity,
                    ws.reconstruction_operation,
                    ws.velocity_ready,
                )
            )
            == before
        )
        ws.prepare_velocity()
        assert ws.velocity_ready and ws.reconstruction_operation.completed_launches == 1
    finally:
        ws.close()


def test_actual_terminal_candidate_failure_rolls_back_receipts_and_retry(tyx_mode, monkeypatch):
    from phymeo.backends.tyx_composite_cloth import TyxCompositeClothParticipant
    from phymeo.backends.tyx_participants import _NodePort
    from phymeo.core.scene import BackendContext

    model, material, initial, inputs = case()
    factory = TyxCompositeClothParticipant(
        model, material, tyx_mode, budgets=BUDGETS, representation=REPRESENTATION
    )
    ws = factory.create_workspace(
        "cloth", initial, inputs, 1 / 64, settings_for(tyx_mode), BackendContext("tyx:" + tyx_mode)
    )
    try:
        ws.predict()
        for _ in range(512):
            ws.position_sweep()
            if ws.position_metrics().max_residual <= 2e-5:
                break
        else:
            pytest.fail("frozen position budget exhausted")
        ws.prepare_velocity()
        native = ws.native
        before = fingerprint(
            (
                native.candidate,
                native.displacement,
                native.terminal_pre,
                native.terminal_requests,
                tuple(native.engine.terminal_operations),
            )
        )
        original = _NodePort.validate_candidate

        def reject(self, state):
            original(self, state)
            raise SolverFailure("late terminal candidate validation")

        impulse = np.array([[1 / 64, 0.0, 0.0]])
        with monkeypatch.context() as patch:
            patch.setattr(_NodePort, "validate_candidate", reject)
            with pytest.raises(SolverFailure, match="late terminal"):
                ws.apply_velocity_impulses(np.array([0]), None, impulse)
        assert (
            fingerprint(
                (
                    native.candidate,
                    native.displacement,
                    native.terminal_pre,
                    native.terminal_requests,
                    tuple(native.engine.terminal_operations),
                )
            )
            == before
        )
        assert len(native.engine.failed_operations) >= 2
        ws.apply_velocity_impulses(np.array([0]), None, impulse)
        data = ws.diagnostics()
        assert [o.name for o in data["terminal_operations"]] == [
            "terminal_response",
            "terminal_apply",
        ]
        assert all(o.completed_launches == 1 for o in data["terminal_operations"])
    finally:
        ws.close()
