"""One real public RR peak step; historical receipt forgeries never reach a device."""

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from test_mujoco_area_friction_lifecycle import _same
from test_pusht_public_mixed_area import (
    KEYS,
    H,
    _asset,
    _controls,
    _declaration,
    _native,
    _observe,
)

from phymeo.backends import mujoco_checkpoint as codec
from phymeo.backends.mujoco_area_friction_state import validate_friction_receipt
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def accepted_peak():
    model, spec, worlds, entries = _declaration()
    key = KEYS[2]
    world = next(world for world in worlds if world.world == key)
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (_asset(model),)})
        batch.set_controls({key: _controls(entries)[key]})
        before = batch.integration_snapshot()[key]
        source = _native(model, before)
        assert source["mass"].shape == (8, 8) and source["owners"]  # No CPU integration.
        with batch.prepare_device_workspace(
            include_bodies=True, include_contacts=True, include_area_pressure=True
        ) as workspace:
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == {key: True}
            accepted = batch.integration_snapshot()[key]
            (area,) = accepted.last_step.raw_contact_fields["area_friction"]
            assert len(area["sample_ids"]) > 0 and area["normal_force"].sum() > 0
            assert accepted.last_step.constraints
            owner = batch._buckets[batch.layout[key].bucket]
            assert owner._plan.resident_native_contacts
            receipt = accepted.last_step
            diagnostic = receipt.raw_contact_fields["area_friction_solver"]["raw_diagnostics"]
            assert diagnostic["meta"][2] > 1 and diagnostic["trace_meta"][0, 2] == 0
            validate_friction_receipt(owner, key, receipt, owner._data[key])
            frame, arrays = _observe(workspace, batch, spec, {key: accepted})
            yield SimpleNamespace(
                model=model,
                spec=spec,
                key=key,
                batch=batch,
                scene=scene,
                owner=owner,
                accepted=accepted,
                capsule=owner.checkpoint(),
                receipt=receipt,
                workspace=workspace,
                frame=frame,
                arrays=arrays,
                revision=batch.checkpoint_revision,
                owner_revision=owner.checkpoint_revision,
                checkpoint=batch.checkpoint().to_bytes(),
                scene_checkpoint=scene.checkpoint().to_bytes(),
                publication=batch.last_step,
                geometry=scene.last_geometry,
            )


def _assert_unchanged(case):
    assert case.owner.checkpoint_revision == case.owner_revision
    assert case.owner.checkpoint().to_bytes() == case.capsule.to_bytes()
    assert case.batch.checkpoint_revision == case.revision
    assert case.batch.last_step is case.publication and case.scene.last_geometry is case.geometry
    assert case.batch.checkpoint().to_bytes() == case.checkpoint
    assert case.scene.checkpoint().to_bytes() == case.scene_checkpoint
    _same(case.batch.integration_snapshot()[case.key], case.accepted)
    _same(case.workspace.readback(case.frame), case.arrays)
    validate_friction_receipt(case.owner, case.key, case.receipt, case.owner._data[case.key])


def _reject(case, receipt, expected, monkeypatch):
    saved = dict(case.capsule._read(case.capsule.binding))
    world = dict(saved["worlds"][0])
    world["receipt"] = receipt
    saved["worlds"] = (world,)
    forged = codec._make_checkpoint(case.capsule.binding, case.capsule.source_revision, saved)
    with monkeypatch.context() as patch:
        patch.setattr(
            case.owner,
            "_build_device",
            lambda *a, **kw: pytest.fail(
                "forged resident receipt reached replacement device allocation"
            ),
        )
        with pytest.raises(ValueError, match=expected):
            validate_friction_receipt(case.owner, case.key, receipt, case.owner._data[case.key])
        with pytest.raises((ValueError, RuntimeError), match=expected):
            case.owner.restore(forged, expected_revision=case.owner_revision)
    _assert_unchanged(case)


def _solver_receipt(receipt, solver):
    raw = dict(receipt.raw_contact_fields)
    raw["area_friction_solver"] = MappingProxyType(solver)
    return replace(receipt, raw_contact_fields=MappingProxyType(raw))


@pytest.mark.parametrize(
    "fault",
    (
        "first-kkt",
        "final-kkt",
        "baseline-load-gate",
        "baseline-velocity-gate",
        "relabel-final-as-baseline",
        "first-load-source",
        "early-exit",
        "native-trial",
    ),
)
def test_working_set_history_and_actual_trial_input_are_bound(accepted_peak, monkeypatch, fault):
    case = accepted_peak
    solver = dict(case.receipt.raw_contact_fields["area_friction_solver"])
    raw = {name: value.copy() for name, value in solver["raw_diagnostics"].items()}
    last = int(raw["meta"][2]) - 1
    trace = raw["trace"]
    if fault == "native-trial":
        trial = solver["native_trial_applied"].copy()
        trial[0] = np.float32(trial[0] + 1.0)
        assert trial.dtype == solver["native_trial_applied"].dtype
        solver["native_trial_applied"] = trial
        expected = "resident final load-defect/source binding"
    elif fault in ("first-kkt", "final-kkt"):
        trace[0 if fault == "first-kkt" else last, 1] = np.float32(
            1.0 + case.owner._plan.solver_tolerance
        )
        expected = "actual native working-set diagnostics"
    elif fault.startswith("baseline-") or fault == "relabel-final-as-baseline":
        # Preserve a real successful native record and final v while claiming
        # the very first Qtrial=0 call was enough. Zero-filled unused slots and
        # valid call metadata make this pass mere shape/status checks.
        final = trace[last].copy()
        trace[:] = 0
        trace[0] = final
        raw["meta"][:] = (0, 0, 1, 1)
        raw["trace_meta"][:] = 0
        raw["trace_meta"][0] = (0, 0, 1)
        # Relabel every per-call record together. Leaving the original later
        # publication rows would test unused padding, not the claimed baseline.
        publication = raw.get("publication")
        if publication is not None:
            final_publication = publication[last].copy()
            publication[:] = 0
            publication[0] = final_publication
        if fault != "relabel-final-as-baseline":
            trace[0, 10 if fault == "baseline-load-gate" else 11] = np.float32(1e-3)
            if publication is not None:
                # A failed area gate never requests a native publication check.
                publication[0] = 0
                publication[0, 0] = -1
            expected = "constitutive closure gates"
        else:
            expected = "baseline/outer area-load source binding"
    elif fault == "first-load-source":
        # Change the first candidate alone: final endpoint and all native KKT
        # summaries remain original. Its recorded area delta now has no source.
        trace[0, 14] = np.float32(trace[0, 14] + 10.0)
        expected = "baseline/outer area-load source binding"
    else:
        raw["trace_meta"][0, 2] = 1
        # Keep monotone convergence metadata so the physical exit gate decides.
        raw["trace_meta"][: last + 1, 2] = 1
        expected = "constitutive closure gates"
    solver["raw_diagnostics"] = MappingProxyType(raw)
    _reject(case, _solver_receipt(case.receipt, solver), expected, monkeypatch)


@pytest.mark.parametrize("fault", ("plane-source-pose", "full-free-entry-q"))
def test_source_geometry_and_complete_free_coordinates_are_bound(accepted_peak, monkeypatch, fault):
    case = accepted_peak
    receipt = case.receipt
    if fault == "plane-source-pose":
        plan = case.owner._plan
        plane_geoms = np.flatnonzero(plan.geom_ids[:, 1] == 0)
        assert len(plane_geoms)
        body = int(plan.geom_ids[plane_geoms[0], 0])
        positions = receipt.dynamics.body_position.copy()
        positions[body, 2] = np.float32(positions[body, 2] + 0.125)
        altered = replace(receipt, dynamics=replace(receipt.dynamics, body_position=positions))
        expected = "resident mixed contact geometry"
    else:
        q = receipt.input_q.copy()
        assert q.shape == (case.model.nq,) and case.model.nq == case.model.nv + 1
        free = case.model.joint("object-free").id
        qa = int(case.model.jnt_qposadr[free])
        q[qa + 2] = np.float32(q[qa + 2] + 0.125)
        # Mutate both receipt and dynamics copies coherently, so the duplicate
        # entry-q equality cannot be the reason this forged history is rejected.
        altered = replace(receipt, input_q=q, dynamics=replace(receipt.dynamics, input_q=q.copy()))
        expected = "resident native endpoint integration"
    _reject(case, altered, expected, monkeypatch)
