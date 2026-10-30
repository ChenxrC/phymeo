"""Unexecuted complete analytic F1 geometry windows, not runtime trajectories."""

import dataclasses
import hashlib
import json
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest
from test_conforming_shell import f1_positions, f1_reference
from test_shell_embedding import disconnected_reference

from phymeo.constraints.conforming_shell import ShellProofLimits, ShellUnresolved
from phymeo.geometry.shell_embedding import certify_shell_embedding
from phymeo.geometry.shell_sweep import certify_shell_sweep
from phymeo.validation.conforming_shell import verify_shell_sweep


def evidence_value(value):
    if isinstance(value, F):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": value.shape, "values": value.tolist()}
    if dataclasses.is_dataclass(value):
        return {"type": type(value).__name__, **{f.name: evidence_value(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, (tuple, list)):
        return [evidence_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): evidence_value(v) for k, v in value.items()}
    return value


def save_evidence(folder, name, value):
    raw = json.dumps(evidence_value(value), sort_keys=True, separators=(",", ":")).encode()
    target = folder / (name + ".json")
    target.write_bytes(raw)
    (folder / (name + ".sha256")).write_text(hashlib.sha256(raw).hexdigest() + "  " + target.name + "\n")
    return raw


@pytest.mark.parametrize("variant", ["identity", "rotation", "translation", "permutation"])
@pytest.mark.parametrize("h", [F(1, 64), F(1, 128), F(1, 256)])
def test_f1_whole_window_every_substep_two_independent_rebuilds(variant, h, tmp_path):
    independent = []
    for rebuild in range(2):
        ref, rotation, shift = f1_reference(variant)
        original = f1_positions(ref, 1 / 8, rotation, shift)
        final = f1_positions(ref, 1 / 4, rotation, shift)
        whole_path = np.stack((original, final))
        whole = certify_shell_sweep(ref, whole_path, h=1 / 8)
        assert verify_shell_sweep(whole, ref, whole_path, h=1 / 8)
        assert max(cell.k_max for cell in whole.cells) < F(33, 64)
        assert len(whole.cells) > 0
        saved = [save_evidence(tmp_path, f"rebuild-{rebuild}-whole", {"reference": ref, "path": whole_path, "h": F(1, 8), "certificate": whole})]
        for step in range(int(F(1, 8) / h)):
            q0 = F(1, 8) + step*h
            q1 = q0+h
            path = np.stack((f1_positions(ref, float(q0), rotation, shift), f1_positions(ref, float(q1), rotation, shift)))
            cert = certify_shell_sweep(ref, path, h=float(h))
            assert verify_shell_sweep(cert, ref, path, h=float(h))
            assert all(cell.k_max < F(33, 64) for cell in cert.cells)
            assert all(len(cell.tetrahedra) == 6 and len(cell.pair_incidence) == 15 for cell in cert.cells)
            saved.append(save_evidence(tmp_path, f"rebuild-{rebuild}-step-{step:03}", {"path": path, "h": h, "certificate": cert}))
        independent.append(saved)
    assert independent[0] == independent[1]


def test_n3_legal_endpoint_components_cross_interior():
    ref = disconnected_reference()
    first = ref.reference_positions.copy()
    last = first.copy()
    last[:3, 2], last[3:, 2] = -1, 1
    assert certify_shell_embedding(ref, first).status == "strict_geometry_certified"
    assert certify_shell_embedding(ref, last).status == "strict_geometry_certified"
    with pytest.raises(ShellUnresolved):
        certify_shell_sweep(ref, np.stack((first, last)), h=1.0)


def test_pure_large_common_translation_is_factored_before_interval_bounds():
    ref, _, _ = f1_reference()
    first = ref.reference_positions.copy()
    last = first + np.array([2**30, -(2**29), 2**28], dtype=np.float64)
    cert = certify_shell_sweep(ref, np.stack((first, last)), h=1.0)
    assert all(cell.k_max < F(1, 2**40) for cell in cert.cells)
    assert verify_shell_sweep(cert, ref, np.stack((first, last)), h=1.0)


def test_interior_time_face_or_director_collapse_not_endpoint_sampling():
    ref, _, _ = f1_reference()
    first = ref.reference_positions.copy()
    last = first.copy()
    # A half-turn swaps both planar directions continuously through a collapsed center.
    last[:, :2] *= -1
    with pytest.raises(ShellUnresolved):
        certify_shell_sweep(ref, np.stack((first, last)), h=1.0)


def test_failed_parent_cells_and_depth_are_charged_not_dropped():
    ref = disconnected_reference()
    last = ref.reference_positions.copy()
    last[:3, 2], last[3:, 2] = -1, 1
    limits = replace(ShellProofLimits(), max_time_cells=1, max_time_depth=1)
    with pytest.raises(ShellUnresolved, match="cells|depth|embedding"):
        certify_shell_sweep(ref, np.stack((ref.reference_positions, last)), h=1.0, limits=limits)