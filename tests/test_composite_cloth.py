"""Public composite ownership and frozen independently authored configurations."""

from math import atan2, sqrt

import numpy as np
import pytest

from phymeo.constraints.bending import BendingConstraints
from phymeo.constraints.cloth_material import (
    ClothMaterial,
    CompositeMaterialAuditSettings,
    CompositeMaterialBudgets,
)
from phymeo.constraints.distance import DistanceConstraints
from phymeo.core.model import ParticleModel
from phymeo.core.scene import BackendContext, SceneInputs, SceneState, StepSettings, fingerprint
from phymeo.core.state import ParticleState, StepInputs
from phymeo.coupling.composite_cloth import CompositeClothParticipant
from phymeo.runtime.scene_stepper import SceneStepper
from phymeo.solvers.composite_cloth import CompositeClothProgram

BUDGETS = CompositeMaterialBudgets(1e-3, 1e-3)
POLICY = CompositeMaterialAuditSettings()


def case(name="A", *, mass_scale=1.0, perturb=0.0, rotation=None, translation=None, copies=1):
    if name == "A":
        x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        w, velocity, force = np.ones(4), np.zeros((4, 3)), np.zeros((4, 3))
        edges, rest, dc = np.empty((0, 2), np.int64), np.zeros(0), np.zeros(0)
        hinges, angles, bc = np.array([[0, 1, 2, 3]]), np.array([0.25]), np.array([0.125])
    elif name == "B":
        x = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1 / 16],
                [1.0, 1.0, -1 / 32],
                [0.0, 2.0, 1 / 8],
                [1.0, 2.0, 0.0],
            ]
        )
        x[4, 2] += perturb
        w = np.array([1.0, 1 / 2, 1 / 4, 2.0, 4.0, 1.0])
        masses = 1 / w
        centroid = np.sum(masses[:, None] * x, axis=0) / np.sum(masses)
        velocity = np.array([1 / 16, -1 / 32, 0.0]) + np.cross(
            np.array([0.0, 0.0, 1 / 8]), x - centroid
        )
        force = masses[:, None] * np.array([0.0, 0.0, 1 / 64])
        edges = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5]])
        rest = np.array([1.0, 1.0, 1.4142135623730951, 1.0, 1.0, 1.0, 1.4142135623730951, 1.0, 1.0])
        dc = np.full(9, 1 / 64)
        hinges = np.array([[1, 2, 0, 3], [3, 2, 1, 4], [3, 4, 2, 5]])
        angles, bc = np.zeros(3), np.full(3, 1 / 8)
    else:
        raise ValueError("unknown authored fixture")
    w, force = w / mass_scale, force * mass_scale
    if rotation is not None:
        x, velocity, force = (a @ rotation.T for a in (x, velocity, force))
    if translation is not None:
        x = x + translation
    n = len(x)
    if copies != 1:
        x = np.concatenate([x + [4 * k, 0.0, 0.0] for k in range(copies)])
        velocity, force = np.tile(velocity, (copies, 1)), np.tile(force, (copies, 1))
        w = np.tile(w, copies)
        edges = np.concatenate([edges + n * k for k in range(copies)])
        hinges = np.concatenate([hinges + n * k for k in range(copies)])
        rest, dc, angles, bc = (np.tile(a, copies) for a in (rest, dc, angles, bc))
    material = ClothMaterial(
        DistanceConstraints(edges, rest, dc), BendingConstraints(hinges, angles, bc)
    )
    return ParticleModel(w, np.zeros(3)), material, ParticleState(x, velocity), StepInputs(force)


def program_for(model, material, mode=None, *, policy=POLICY):
    if mode is None:
        return CompositeClothProgram(model, material, audit_settings=policy, budgets=BUDGETS)
    from phymeo.backends.tyx_composite_cloth import TyxCompositeClothProgram

    return TyxCompositeClothProgram(model, material, mode, audit_settings=policy, budgets=BUDGETS)


def settings_for(mode=None, **kwargs):
    return StepSettings(
        substeps=1,
        position_iterations=512,
        velocity_iterations=1,
        tolerance=1e-8 if mode is None else 2e-5,
        **kwargs,
    )


def settled_workspace(name="A", *, mode=None, h=1 / 64, **kwargs):
    model, material, state, inputs = case(name, **kwargs)
    program = program_for(model, material, mode)
    settings = settings_for(mode)
    ws = program.workspace("cloth", state, inputs, h, settings)
    try:
        ws.predict()
        for _ in range(settings.position_iterations):
            ws.position_sweep()
            if ws.position_metrics().max_residual <= settings.tolerance:
                break
        else:
            pytest.fail("frozen position iteration budget exhausted")
        ws.prepare_velocity()
        return ws
    except BaseException:
        ws.close()
        raise


def test_owned_material_empty_blocks_and_positive_mass_scope():
    model, material, _, _ = case()
    original = material.bending.rest_angle.copy()
    copied = ClothMaterial(material.distance, material.bending)
    assert copied.identity == material.identity
    assert not np.shares_memory(copied.bending.rest_angle, material.bending.rest_angle)
    assert copied.order == ("distance", "bending")
    np.testing.assert_array_equal(copied.bending.rest_angle, original)
    assert not copied.bending.indices.flags.writeable
    with pytest.raises(ValueError, match="all-dynamic"):
        CompositeClothProgram(
            ParticleModel(np.array([0.0, 1.0, 1.0, 1.0])), material, budgets=BUDGETS
        )
    CompositeClothProgram(model, ClothMaterial(), budgets=BUDGETS)


def test_first_unit_hinge_sweep_matches_literal_nonconverged_equations():
    model, material, state, inputs = case()
    ws = program_for(model, material).workspace("cloth", state, inputs, 1 / 8, settings_for())
    try:
        ws.predict()
        ws.position_sweep()
        row = ws.frames[0].blocks[1].rows[0]
        assert row[42] == pytest.approx(1 / 56, rel=0, abs=2e-17)
        assert row[44] == pytest.approx(14.0, rel=0, abs=2e-14)
        expected = state.position.copy()
        expected[:, 2] = (1 / 28, 0.0, -1 / 56, -1 / 56)
        np.testing.assert_allclose(ws.candidate.position, expected, rtol=0, atol=2e-17)
        delta = 1 / 56
        theta = atan2(6 * delta * sqrt(1 + 4 * delta**2), 1 - 5 * delta**2)
        assert row[37] == pytest.approx(theta, abs=2e-15)
        ws.prepare_velocity()
        assert np.sum(ws.candidate.velocity**2) / 2 == pytest.approx(3 / 49, abs=2e-15)
        np.testing.assert_allclose(np.sum(ws.candidate.velocity, axis=0), 0.0, atol=1e-15)
        assert ws.position_metrics().max_residual > 1e-8
        with pytest.raises(RuntimeError, match="already"):
            ws.prepare_velocity()
    finally:
        ws.close()


def test_complete_order_owned_trace_both_residual_families_and_one_reconstruction():
    ws = settled_workspace("B", h=1 / 128)
    try:
        metrics = ws.position_metrics()
        assert set(metrics.residuals) == {"distance", "bending"}
        data = ws.diagnostics()
        assert data["angular_error_rad"] >= 0 and metrics.geometric_error >= 0
        assert data["material_path"].material_fingerprint == ws.material.identity
        for i, sweep in enumerate(data["material_path"].sweeps):
            assert sweep.sweep_index == i
            assert tuple(b.kind for b in sweep.blocks) == ("distance", "bending")
            assert not sweep.blocks[0].rows.flags.writeable
            np.testing.assert_array_equal(
                sweep.blocks[0].positions_out, sweep.blocks[1].positions_in
            )
        expected = ws.predicted.velocity + (ws.candidate.position - ws.predicted.position) / ws.h
        np.testing.assert_array_equal(ws.candidate.velocity, expected)
        assert data["material_audit"]["physical_affine_domain"].status == "certified"
    finally:
        ws.close()


def test_participant_binds_material_identity_and_rejects_unrecorded_load():
    model, material, state, inputs = case()
    factory = CompositeClothParticipant(model, material, budgets=BUDGETS)
    assert factory.material.identity == material.identity
    ws = factory.create_workspace("cloth", state, inputs, 1 / 64, settings_for(), BackendContext())
    try:
        ws.predict()
        from phymeo.core.capabilities import SolverFailure

        with pytest.raises(SolverFailure, match="unrecorded"):
            ws.apply_position_load(np.array([0]), None, np.zeros((1, 3)))
    finally:
        ws.close()


def test_empty_material_uses_public_runtime_and_complete_empty_observations():
    model = ParticleModel(np.ones(1), np.zeros(3))
    state = SceneState({"cloth": ParticleState(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]))})
    inputs = SceneInputs({"cloth": StepInputs.zeros(1)})
    program = program_for(model, ClothMaterial())
    with SceneStepper(program.scene, program, settings_for()) as runtime:
        result = runtime.step(state, inputs, 1 / 128)
    assert result.report.accepted, result.report.reason
    np.testing.assert_array_equal(result.state.domains["cloth"].position, [[1 / 128, 0.0, 0.0]])
    blocks = result.report.substeps[0].diagnostics["material_path"].sweeps[0].blocks
    assert [len(b.rows) for b in blocks] == [0, 0]
    assert fingerprint(state) != fingerprint(result.state)
