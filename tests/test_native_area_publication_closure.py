"""Final native row law must close on the actual area load selected for publication."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import test_resident_native_receipt_forgery as receipt_fixtures

accepted_peak = receipt_fixtures.accepted_peak
_solver_receipt = receipt_fixtures._solver_receipt


@pytest.mark.cuda
@pytest.mark.parametrize("area_offset", (0.0, 2.0**-17))
def test_dual_area_closure_does_not_excuse_final_native_row_violation(area_offset):
    import tyx as tx

    from phymeo.backends.mujoco_tyx._area_native_coupled_kernels import qualify_publication
    from phymeo.backends.mujoco_tyx._resources import DeviceResources

    # A real fixed-point candidate with six DOFs, M=I, Qtrial=100, native
    # J=4/R=1/f=1/aref=1, external=-104. The native trial acceleration is zero.
    # One ULP more area load passes both original area closure gates but yields
    # final g=4*2^-17 > 2e-5. No absolute/world coordinate special case is needed.
    n, width, iteration, h, tolerance = 6, 6, 1, 0.001, 2e-5
    packet = {
        "counts": np.array([[0, 0, 1]], np.int32),
        "physical_factor": np.eye(8, dtype=np.float32)[None],
        "native_constraint": np.zeros((1, 8), np.float32),
        "efc_data": np.zeros((1, 1, 14), np.float32),
        "efc_jacobian": np.zeros((1, 1, 8), np.float32),
        "applied": np.zeros((1, 8), np.float32),
        "passive": np.zeros((1, 8), np.float32),
        "bias": np.zeros((1, 8), np.float32),
        "vectors": np.zeros((1, 8, width), np.float32),
        "trial_applied": np.zeros((1, 8), np.float32),
        "meta": np.array([[0, iteration, iteration + 1, 1]], np.int32),
        "trace_meta": np.zeros((1, 25, 3), np.int32),
        "trace": np.zeros((1, 25, 14 + 2 * width), np.float32),
        "publication": np.zeros((1, 25, 1 + width + 1), np.float32),
        "status": np.zeros(1, np.int32),
    }
    packet["native_constraint"][0, 0] = 4
    packet["efc_data"][0, 0, [3, 4, 5, 9, 11]] = 1
    packet["efc_jacobian"][0, 0, 0] = 4
    packet["applied"][0, 0] = -104
    packet["vectors"][0, 4, 0] = 100
    packet["vectors"][0, 5, 0] = 100 + area_offset
    packet["vectors"][0, 3, 0] = area_offset
    packet["trial_applied"][0, 0] = -4
    packet["trace_meta"][0, iteration] = (1, 0, 1)
    packet["trace"][0, iteration, 10:14] = (
        area_offset / (101 + area_offset),
        h * area_offset,
        area_offset,
        h * area_offset,
    )
    packet["trace"][0, iteration, 14 + width] = area_offset
    assert packet["trace"][0, iteration, 10] <= 1e-6
    assert packet["trace"][0, iteration, 11] <= 1e-7
    resource = DeviceResources()
    runtime = tx.Runtime(backend="cuda")
    resource.set_runtime(runtime)
    try:
        with tx.use_runtime(runtime):
            buffers = []
            for values in packet.values():
                buffer = tx.cuda_array(values)
                resource.add_buffer(buffer)
                buffers.append(buffer)
            qualify_publication(1, n, width, iteration, tolerance, *buffers, mode="cuda")
            runtime.sync()
            actual = {key: buffer.to_numpy() for key, buffer in zip(packet, buffers, strict=True)}
    finally:
        resource.close()
    assert actual["status"][0] == 0
    np.testing.assert_array_equal(actual["efc_data"], packet["efc_data"])
    np.testing.assert_array_equal(actual["vectors"][0, 5], packet["vectors"][0, 5])
    assert actual["publication"][0, iteration, 1] == np.float32(area_offset)
    if area_offset:
        assert actual["publication"][0, iteration, 0] > 1
        assert actual["meta"][0, 3] == actual["trace_meta"][0, iteration, 2] == 0
        assert actual["trial_applied"][0, 0] == np.float32(-4 + area_offset)
    else:
        assert actual["publication"][0, iteration, 0] == 0
        assert actual["meta"][0, 3] == actual["trace_meta"][0, iteration, 2] == 1
        assert actual["trial_applied"][0, 0] == -4


@pytest.mark.cuda
def test_saved_rr_1584_step_keeps_original_trial_and_publication_row_gates():
    import mujoco

    from experiments.area_pressure.pusht_active_checkpoint import require_saved_state
    from experiments.area_pressure.pusht_public_scene import (
        H,
        build_scene_spec,
        build_side_asset,
        control_from_record,
    )
    from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaPressureScene,
        MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
    )
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    from phymeo.training.manipulation_tasks import manipulation_scenes

    folder = Path(__file__).resolve().parents[1] / "artifacts/rr-pose-replan-20261121-01"
    source = folder / "run-B.json"
    if not source.is_file():
        pytest.skip("requires the separately archived real RR 1.584 s failure checkpoint")
    report = json.loads(source.read_bytes())
    cap = report["final_checkpoint"]
    if not all((folder / cap[name]).is_file() for name in ("file", "metadata_file")):
        pytest.skip("requires the separately archived failure capsule and metadata")
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "2161ae007060700a084cb1fce053f72c28f3b64d45cd7e67156522e0e40db5df"
    )
    capsule, metadata_bytes = (
        (folder / cap["file"]).read_bytes(),
        (folder / cap["metadata_file"]).read_bytes(),
    )
    metadata = json.loads(metadata_bytes)
    assert hashlib.sha256(capsule).hexdigest() == cap["sha256"] == metadata["checkpoint_sha256"]
    assert hashlib.sha256(metadata_bytes).hexdigest() == cap["metadata_sha256"]
    assert cap["source_tick"] == 1584 and report["failed_candidate"]["restored"]
    control = report["failed_candidate"]["control"]
    assert control["tick"] == 1584
    case = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    spec, model = build_scene_spec(case.mjcf), mujoco.MjModel.from_xml_string(case.mjcf)
    declaration = report["provenance"]["reset_declaration"]
    initial = MuJoCoInitialState(
        np.asarray(declaration["q"]), np.asarray(declaration["v"]), time=declaration["time"]
    )
    with MuJoCoTyxHeterogeneousBatch(
        (spec,), (MuJoCoTyxWorldSpec(6, spec.key, initial),), mode="cuda"
    ) as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(
            batch, {spec.key: (build_side_asset(model),)}
        )
        assert scene.asset_sha256 == metadata["asset_sha256"]
        scene.restore(
            MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(capsule),
            expected_revision=batch.checkpoint_revision,
        )
        require_saved_state(batch.integration_snapshot()[6], metadata["state"])
        batch.set_controls({6: control_from_record(control)})
        before = batch.integration_snapshot()[6]
        assert scene.step(H, expected_revision=batch.checkpoint_revision) == {6: True}
        after = batch.integration_snapshot()[6]
        assert after.tick == before.tick + 1
        receipt = after.last_step
        raw = receipt.raw_contact_fields["area_friction_solver"]["raw_diagnostics"]
        calls = int(raw["meta"][2])
        assert 8 < calls <= 25
        assert raw["publication"][calls - 1, 0] <= 1
        np.testing.assert_array_equal(
            raw["publication"][calls - 1, 1:9].astype(float), receipt.qacc
        )
        old_closed = (raw["trace"][:calls, 10] <= 1e-6) & (raw["trace"][:calls, 11] <= 1e-7)
        assert np.any(
            old_closed & (raw["publication"][:calls, 0] > 1) & (raw["trace_meta"][:calls, 2] == 0)
        )
        for row in receipt.constraints:
            gradient = (
                row.jacobian @ receipt.qacc
                - row.reference_acceleration
                + row.regularization * row.force
            )
            violation = abs(gradient) if row.force > 0 else max(0.0, -gradient)
            assert violation <= float(np.float32(2e-5)) * max(
                1.0, abs(row.regularization * row.force)
            )
        # Removing the new evidence cannot turn a delayed publication check
        # into an old full/half/quarter-only history.
        from phymeo.backends.mujoco_area_friction_state import validate_friction_receipt

        solver = receipt.raw_contact_fields["area_friction_solver"]
        old_claim = {name: value for name, value in raw.items() if name != "publication"}
        forged = _solver_receipt(receipt, {**solver, "raw_diagnostics": old_claim})
        owner = batch._buckets[batch.layout[6].bucket]
        with pytest.raises(ValueError, match="resident actual closure/exit status"):
            validate_friction_receipt(owner, 6, forged, owner._data[6])


@pytest.mark.cuda
@pytest.mark.parametrize("fault", ("ratio", "acceleration", "force", "missing", "padding"))
def test_publication_row_evidence_is_bound_to_its_actual_source(accepted_peak, fault):
    from phymeo.backends.mujoco_area_friction_state import validate_friction_receipt

    # This fixture uses portable shipped original-model inputs, not ignored
    # continuation artifacts. Its actual row/force/pose source is audited too.
    owner, key, receipt = accepted_peak.owner, accepted_peak.key, accepted_peak.receipt
    state = owner._data[key]
    solver = dict(receipt.raw_contact_fields["area_friction_solver"])
    raw = {name: value.copy() for name, value in solver["raw_diagnostics"].items()}
    evidence = raw["publication"]
    last, width = int(raw["meta"][2]) - 1, 8
    assert evidence[last, 0] >= 0 and receipt.constraints
    if fault == "ratio":
        evidence[last, 0] += np.float32(0.01)
    elif fault == "acceleration":
        evidence[last, 1] += np.float32(0.125)
    elif fault == "force":
        evidence[last, 1 + width] += np.float32(1.0)
    elif fault == "missing":
        evidence[last] = 0
        evidence[last, 0] = -1
    else:
        evidence[last, -1] = 1
    forged = _solver_receipt(receipt, {**solver, "raw_diagnostics": raw})
    with pytest.raises(ValueError, match="resident publication|resident missing"):
        validate_friction_receipt(owner, key, forged, state)


@pytest.mark.parametrize("native_count", (0, 1))
@pytest.mark.parametrize("new", (False, True))
def test_zero_native_rows_and_legacy_three_field_histories_keep_their_meaning(native_count, new):
    from types import SimpleNamespace

    from phymeo.backends.mujoco_area_friction_state import (
        _resident_diagnostics,
        _resident_trace_sources,
    )

    plan = SimpleNamespace(nv=6, solver_iterations=128, solver_tolerance=2e-5)
    raw = dict(
        meta=np.array([0, 0, 1, 1], np.int32),
        trace_meta=np.zeros((25, 3), np.int32),
        trace=np.zeros((25, 26), np.float32),
    )
    raw["trace_meta"][0] = (0, 0, 1)
    raw["trace"][0, :10] = (0, 0, 0, 0, native_count, 0, 1, 128, 0, 1)
    zero = np.zeros(6)
    rows = (
        ()
        if not native_count
        else (
            SimpleNamespace(
                jacobian=zero,
                regularization=1.0,
                reference_acceleration=-1.0,
                lower=0.0,
                upper=None,
                force=0.0,
            ),
        )
    )
    if new:
        raw["publication"] = np.zeros((25, 7 + 512), np.float32)
        raw["publication"][0, 0] = 0 if native_count else -1
    _resident_diagnostics(raw, zero, plan, native_count, publication_capacity=512)
    _resident_trace_sources(
        raw,
        (),
        np.eye(6),
        zero,
        zero,
        0.001,
        zero,
        native_rows=rows,
        mass=np.eye(6),
        passive=zero,
        bias=zero,
        physical_acceleration=zero,
        native_tolerance=2e-5,
    )
