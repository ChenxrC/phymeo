"""An independently written exact distance path exercises the new reader.

No producer kernel, projection, codec result, or solver supplies this fixture's
positions, gradients, multiplier, products, geometry, or ideal equations.
"""

from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from phymeo.constraints.cloth_material import ClothMaterial, CompositeMaterialAuditSettings
from phymeo.constraints.distance import DistanceConstraints
from phymeo.constraints.material_representation import (
    NativeCompositeRepresentation, NativeRepresentationOperation, RepresentationBoundary,
    RepresentedMaterialBatch, RepresentedMaterialPath, RepresentedMaterialSweep,
    RoundingWitness, deserialize_evidence, serialize_evidence,
)
from phymeo.constraints.material_trace import CompositeMaterialPath
from phymeo.core.capabilities import SolverFailure
from phymeo.core.model import ParticleModel
from phymeo.core.scene import StepSettings, fingerprint
from phymeo.core.state import ParticleState, StepInputs
from phymeo.validation.material_representation import RepresentedMaterialPathLedger, projection_charge


def _boundary(name, values):
    values = np.asarray(values, np.float32)
    witnesses = tuple(RoundingWitness(F(float(v)).numerator, F(float(v)).denominator, 32, float(v)) for v in values.flat)
    return RepresentationBoundary(name, values, witnesses)


def _witnesses(values):
    return tuple(RoundingWitness(F(float(v)).numerator, F(float(v)).denominator, 64, float(v)) for v in values.flat)


def literal_distance_path():
    # h=1/2, compliance=1/2 -> alpha=2; C=1, denominator=4,
    # delta=-1/4, gradients=(-ex,+ex), corrected distance=3/2.
    model = ParticleModel(np.ones(2), np.zeros(3))
    material = ClothMaterial(DistanceConstraints(np.array([[0, 1]]), np.array([1.0]), np.array([0.5])))
    policy, representation = CompositeMaterialAuditSettings(), NativeCompositeRepresentation()
    reference = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    initial = ParticleState(reference.copy(), np.zeros((2, 3)))
    inputs = StepInputs.zeros(2)
    settings = StepSettings(tolerance=2e-5, position_iterations=512)
    ledger = RepresentedMaterialPathLedger(model, material, initial, inputs, 0.5, settings,
        audit_settings=policy, backend="tyx:interpret", representation=representation)
    identity = fingerprint((reference, material.identity, representation, "cloth", 0.5, fingerprint(initial), fingerprint(inputs)))
    zero = np.zeros((2, 3), np.float32)
    displacement = np.array([[0.25, 0, 0], [-0.25, 0, 0]], np.float32)
    world = np.array([[0.25, 0, 0], [1.75, 0, 0]], np.float64)
    raw = np.zeros((1, 18), np.float32)
    raw[0, 6:12] = displacement.ravel()
    raw[0, 13], raw[0, 14], raw[0, 17] = -0.25, -1, -0.25
    d_batch = RepresentedMaterialBatch("distance", np.array([[0, 1]]), identity, zero, displacement,
        reference, world, np.zeros(1, np.float32), np.array([-0.25], np.float32), np.ones(2, np.float32),
        np.ones(1, np.float32), np.array([2], np.float32), np.array([[[-2, 0, 0]]], np.float32), raw, displacement[None],
        np.array([[[-2, 0, 0]]], np.float32), np.array([[[-1.5, 0, 0]]], np.float32),
        reference[None], world[None], _witnesses(reference), _witnesses(world), np.zeros(1, np.int32), 1, 1)
    empty = np.zeros(0, np.float32)
    b_batch = RepresentedMaterialBatch("bending", np.empty((0, 4), np.int64), identity, displacement, displacement,
        world, world, empty, empty, np.ones(2, np.float32), empty, empty, np.empty((0, 3, 3), np.float32),
        np.empty((0, 45), np.float32), np.empty((0, 4, 3), np.float32),
        np.empty((0, 3, 3), np.float32), np.empty((0, 3, 3), np.float32),
        np.empty((0, 4, 3)), np.empty((0, 4, 3)), (), (), np.zeros(0, np.int32), 0, 0)
    boundaries = (
        _boundary("predict/velocity", zero), _boundary("predict/force", zero),
        _boundary("predict/weights", np.ones(2, np.float32)), _boundary("predict/gravity", np.zeros(3, np.float32)),
        _boundary("predict/h", 0.5), _boundary("alpha/inverse_h", 2.0),
        _boundary("distance/rest", [1]), _boundary("distance/compliance", [0.5]),
        _boundary("reference/distance", [[[-2, 0, 0]]]),
        _boundary("bending/rest", empty), _boundary("bending/compliance", empty),
        _boundary("reference/bending", np.empty((0, 3, 3), np.float32)),
    )
    predict = NativeRepresentationOperation("predict", tuple(b.computed for b in boundaries[:5]), (zero, zero), 1, 1)
    inverse = np.array(2, np.float32)
    alpha_ops = (
        NativeRepresentationOperation("distance/alpha/0", (np.zeros(1, np.float32), np.array([0.5], np.float32), inverse), (np.ones(1, np.float32),), 1, 1),
        NativeRepresentationOperation("distance/alpha/1", (np.zeros(1, np.float32), np.ones(1, np.float32), inverse), (np.array([2], np.float32),), 1, 1),
    )
    velocity = np.array([[0.5, 0, 0], [-0.5, 0, 0]], np.float32)
    reconstruct = NativeRepresentationOperation("reconstruct", (zero, zero, displacement, np.array(0.5, np.float32)), (velocity,), 1, 1)
    path = RepresentedMaterialPath(material.identity, fingerprint(policy), fingerprint(inputs), fingerprint(initial),
        "tyx:interpret", 0.5, representation, "cloth", reference, identity, zero, displacement,
        reference, zero.astype(np.float64), velocity, np.array([2], np.float32), empty,
        (RepresentedMaterialSweep(0, (d_batch, b_batch)),), boundaries, predict, reconstruct, alpha_ops)
    final = ParticleState(world.copy(), velocity.astype(np.float64))
    diagnostics = dict(material_path=path, represented_reference=reference.copy(), represented_displacement=displacement.copy(),
        distance_multipliers=np.array([-0.25]), bending_multipliers=np.zeros(0))
    return ledger, path, final, inputs, diagnostics


def _audit(ledger, final, inputs, diagnostics):
    return ledger.require(final, inputs, np.zeros_like(final.position), diagnostics, sweeps=1)


def test_literal_complete_world_path_zero_charge_and_original_units():
    ledger, path, final, inputs, diagnostics = literal_distance_path()
    result = _audit(ledger, final, inputs, diagnostics)
    assert result["cumulative_projection_error"] == 0
    assert result["physical_affine_domain"].status == "certified"
    assert F(final.position[1, 0]) - F(final.position[0, 0]) == F(3, 2)
    assert F(3, 2) - 1 + 2 * F(-1, 4) == 0
    assert result["ordered_material_work"][0]["rows"][0]["arithmetic_work_discrepancy"] == 0
    assert path.sweeps[0].blocks[0].rows.dtype == np.float32


@pytest.mark.parametrize("field", ["rows", "reference_vectors", "products", "geometry_before", "geometry_after", "world_before", "world_after", "lambda_out", "compute_rest", "compute_alpha", "compute_weights"])
def test_each_raw_and_derived_family_is_independently_bound(field):
    ledger, path, final, inputs, diagnostics = literal_distance_path()
    batch = path.sweeps[0].blocks[0]
    changed = getattr(batch, field).copy()
    changed.flat[0] += 0.125
    forged = replace(batch, **{field: changed})
    diagnostics["material_path"] = replace(path, sweeps=(RepresentedMaterialSweep(0, (forged, path.sweeps[0].blocks[1])),))
    with pytest.raises(SolverFailure):
        _audit(ledger, final, inputs, diagnostics)


@pytest.mark.parametrize("change", ["identities", "launches", "reference", "hidden_d", "prediction", "reconstruction", "boundary", "extra_alpha", "missing_sweep"])
def test_sequence_native_operations_and_hidden_provenance_reject(change):
    ledger, path, final, inputs, diagnostics = literal_distance_path()
    batch = path.sweeps[0].blocks[0]
    if change in ("identities", "launches"):
        forged = replace(batch, node_indices=np.array([[1, 0]])) if change == "identities" else replace(batch, completed_launches=0)
        path = replace(path, sweeps=(RepresentedMaterialSweep(0, (forged, path.sweeps[0].blocks[1])),))
    elif change == "reference":
        reference = path.reference.copy()
        reference[0, 0] = 2**-100
        assert float(F(reference[0, 0]) + F(1, 4)) == final.position[0, 0]
        path = replace(path, reference=reference)
    elif change == "hidden_d":
        diagnostics["represented_displacement"][0, 0] += 0.125
    elif change == "prediction":
        path = replace(path, prediction_operation=replace(path.prediction_operation, completed_launches=0))
    elif change == "reconstruction":
        path = replace(path, reconstruction_operation=replace(path.reconstruction_operation, name="predict"))
    elif change == "boundary":
        path = replace(path, compute_boundaries=path.compute_boundaries[:-1])
    elif change == "extra_alpha":
        path = replace(path, alpha_operations=path.alpha_operations + path.alpha_operations[:1])
    else:
        path = replace(path, sweeps=())
    diagnostics["material_path"] = path
    with pytest.raises(SolverFailure):
        _audit(ledger, final, inputs, diagnostics)


def test_unknown_v1_schema_and_mutable_aliases_do_not_become_v2():
    ledger, path, final, inputs, diagnostics = literal_distance_path()
    old = CompositeMaterialPath(path.material_fingerprint, path.policy_fingerprint, path.input_fingerprint,
        path.initial_fingerprint, path.backend, path.duration, path.predicted_position, path.predicted_velocity,
        path.distance_alpha, path.bending_alpha, (), 1, 1)
    diagnostics["material_path"] = old
    with pytest.raises(SolverFailure, match="v2"):
        _audit(ledger, final, inputs, diagnostics)
    with pytest.raises(ValueError):
        replace(path, schema="xpbd-composite-material-path-v1")
    restored = deserialize_evidence(serialize_evidence(path))
    assert fingerprint(restored) == fingerprint(path)
    assert not np.shares_memory(restored.reference, path.reference)
    assert not restored.sweeps[0].blocks[0].rows.flags.writeable


def test_conservative_charge_uses_both_decoder_residuals_without_double_count():
    charge, direct, terms = projection_charge(2**100, 1, 2, 1, 1, 1, 1)
    assert terms == (F(0), F(0), F(-1), F(-2))
    assert (charge, direct) == (F(3), F(1))
    assert charge != direct and charge >= direct


def test_complete_sweep_reader_has_no_acceptance_side_effect_on_failure():
    ledger, path, final, inputs, diagnostics = literal_distance_path()
    before = fingerprint((final, diagnostics))
    broken = dict(diagnostics, bending_multipliers=np.array([1.0]))
    with pytest.raises(SolverFailure):
        _audit(ledger, final, inputs, broken)
    assert fingerprint((final, diagnostics)) == before
    assert _audit(ledger, final, inputs, diagnostics)["cumulative_projection_error"] == 0


def test_success_only_status_and_complete_order_types_reject_prefixes():
    _, path, _, _, _ = literal_distance_path()
    distance, bending = path.sweeps[0].blocks
    with pytest.raises(ValueError, match="successful"):
        replace(distance, status=np.ones(1, np.int32))
    with pytest.raises(ValueError, match="complete"):
        RepresentedMaterialSweep(0, (distance,))
    with pytest.raises(ValueError, match="complete"):
        RepresentedMaterialSweep(0, (bending, distance))
    with pytest.raises(ValueError):
        replace(distance, rows=np.zeros((0, 18), np.float32))
