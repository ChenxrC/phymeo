"""Adversarial typed observation audits, not equality to producer expectations."""

from dataclasses import replace

import numpy as np
import pytest
from test_composite_cloth import settled_workspace

from phymeo.constraints.cloth_material import CompositeMaterialAuditSettings
from phymeo.core.capabilities import SolverFailure
from phymeo.validation.composite_material_path import CompositeMaterialPathLedger


def audit(ws, data, candidate=None):
    return ws.ledger.require(
        ws.candidate if candidate is None else candidate,
        ws.inputs,
        np.zeros_like(ws.candidate.position),
        data,
        sweeps=len(ws.frames),
    )


def changed_bending_row(data, column, change):
    path = data["material_path"]
    sweep = path.sweeps[0]
    block = sweep.blocks[1]
    rows = block.rows.copy()
    rows[0, column] += change
    altered = replace(block, rows=rows)
    altered_sweep = replace(sweep, blocks=(sweep.blocks[0], altered))
    return dict(data, material_path=replace(path, sweeps=(altered_sweep, *path.sweeps[1:])))


@pytest.mark.parametrize("column", [0, 12, 24, 36, 37, 38, 39, 40, 41, 42, 43, 44])
def test_each_bending_observation_family_is_independently_checked(column):
    ws = settled_workspace()
    try:
        data = ws.diagnostics()
        with pytest.raises(SolverFailure):
            audit(ws, changed_bending_row(data, column, 1 / 16))
    finally:
        ws.close()


def test_missing_frames_and_wrong_material_or_input_binding_rejected():
    ws = settled_workspace()
    try:
        data = ws.diagnostics()
        path = data["material_path"]
        for altered in (
            replace(path, material_fingerprint="different-material"),
            replace(path, input_fingerprint="different-inputs"),
            replace(path, sweeps=()),
        ):
            with pytest.raises(SolverFailure):
                audit(ws, dict(data, material_path=altered))
        with pytest.raises(ValueError, match="order"):
            replace(path.sweeps[0], blocks=path.sweeps[0].blocks[::-1])
    finally:
        ws.close()


def test_conversion_launch_and_final_hidden_state_cannot_be_forged():
    ws = settled_workspace()
    try:
        data = ws.diagnostics()
        path, sweep = data["material_path"], data["material_path"].sweeps[0]
        block = sweep.blocks[1]
        for altered in (
            replace(block, launches=1, completed_launches=1),
            replace(block, compute_start=block.compute_start + 1 / 16),
            replace(block, compute_rest=block.compute_rest + 1 / 16),
        ):
            frame = replace(sweep, blocks=(sweep.blocks[0], altered))
            changed = replace(path, sweeps=(frame, *path.sweeps[1:]))
            with pytest.raises(SolverFailure):
                audit(ws, dict(data, material_path=changed))
        final = ws.candidate.copy()
        final.position[0, 2] += 1 / 16
        with pytest.raises(SolverFailure, match="final"):
            audit(ws, data, final)
        final = ws.candidate.copy()
        final.velocity[0, 2] += 1 / 16
        with pytest.raises(SolverFailure):
            audit(ws, data, final)
    finally:
        ws.close()


def test_altered_model_parameters_and_proof_limits_do_not_rebind_existing_receipt():
    ws = settled_workspace()
    try:
        data = ws.diagnostics()
        other = CompositeMaterialPathLedger(
            ws.model,
            ws.material,
            ws.initial,
            ws.inputs,
            ws.h,
            ws.settings,
            audit_settings=CompositeMaterialAuditSettings(angle_terms=1),
        )
        with pytest.raises(SolverFailure, match="binding"):
            other.require(
                ws.candidate,
                ws.inputs,
                np.zeros_like(ws.candidate.position),
                data,
                sweeps=len(ws.frames),
            )
    finally:
        ws.close()


def test_ordered_work_includes_numerical_discrepancy_without_erasing_num005():
    ws = settled_workspace("B", h=1 / 128)
    try:
        data = ws.diagnostics()
        work = data["material_audit"]["ordered_material_work"]
        assert len(work) == 2 * len(ws.frames)
        assert any(block["rows"] for block in work)
        for block in work:
            for row in block["rows"]:
                assert row["arithmetic_work_discrepancy"] == pytest.approx(
                    row["kinetic_change"] - row["material_impulse_work"], abs=1e-15
                )
                assert len(row["discrete_angular_contribution"]) == 3
        assert "angular_defect" in data["physical_audit"]
    finally:
        ws.close()


def test_prediction_reconstruction_and_alpha_compute_boundaries_are_bound():
    ws = settled_workspace()
    try:
        data = ws.diagnostics()
        path = data["material_path"]
        boundary = path.compute_boundaries[0]
        changed = replace(boundary, computed=boundary.computed + 1 / 16)
        for altered in (
            replace(path, compute_boundaries=(changed, *path.compute_boundaries[1:])),
            replace(path, compute_boundaries=path.compute_boundaries[1:]),
            replace(path, alpha_launches=(0, 2)),
        ):
            with pytest.raises(SolverFailure):
                audit(ws, dict(data, material_path=altered))
    finally:
        ws.close()
