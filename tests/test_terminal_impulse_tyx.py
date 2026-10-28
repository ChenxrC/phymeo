"""Actual backend execution of prescribed terminal phases, with exact oracles."""

from fractions import Fraction as F

import numpy as np
import pytest
from test_terminal_impulse import requests, run_case

import phymeo as ph

pytestmark = pytest.mark.tyx


def exact_reference(chain):
    """Derive from independent scalar impulse/XPBD equations in rational SI."""
    h, mass, eta = F(1, 4), F(2), F(1, 3)
    p = F(3, 7) if chain else F(1, 3)
    x0 = [F(1, 8), F(9, 8)] if chain else [F(1, 8)]
    vm = [F(-1) + p] + ([F(-1)] if chain else [])
    x = [a + h * b for a, b in zip(x0, vm, strict=True)]
    if chain:
        multiplier = -(x[1] - x[0] - 1) / (2 + F(1, 16) / h**2)
        x[0] -= multiplier
        x[1] += multiplier
        vm[0] -= multiplier / h
        vm[1] += multiplier / h
    q, rv = -h * p / mass, -p / mass
    vp, rp = [vm[0] + eta] + vm[1:], rv - eta / mass
    assert vp[0] == rp == (F(-8, 21) if chain else F(-1, 3))
    return x, q, vm, rv, vp, rp


@pytest.mark.parametrize("chain", [False, True])
def test_actual_terminal_analytic_and_original_material_ledger_only_accepts_pre(tyx_mode, chain):
    scene, initial, _, base, settings, context, material, trial = run_case(tyx_mode, chain)
    evidence = trial.result.report.substeps[0].diagnostics["terminal_impulse"]
    pre, post = evidence["pre_impulse_state"], trial.result.state
    x, q, vm, rv, vp, rp = exact_reference(chain)
    for actual, expected in (
        (pre.domains["nodes"].position[:, 1], x),
        (pre.domains["robot"].q, [q]),
        (pre.domains["nodes"].velocity[:, 1], vm),
        (pre.domains["robot"].v, [rv]),
        (post.domains["nodes"].velocity[:, 1], vp),
        (post.domains["robot"].v, [rp]),
    ):
        np.testing.assert_allclose(actual, [float(v) for v in expected], rtol=0, atol=2e-6)
    assert evidence["ledger"]["compute_precision"] == "float32"
    assert evidence["ledger"]["max_residual"] <= settings.tolerance
    assert trial.result.report.max_residual >= evidence["ledger"]["max_residual"]
    for name in ("robot", "nodes"):
        before = evidence["pre_impulse_diagnostics"]["participants"][name]
        after = trial.result.report.substeps[0].diagnostics["participants"][name]
        assert after["execution"]["compute_dtype"] == "float32"
        assert after["execution"]["mode"] == tyx_mode
        assert (
            after["execution"]["completed_launches"] > before["execution"]["completed_launches"] > 0
        )
        assert evidence["post_execution_delta"][name]["completed_launches"] > 0
    particle_model = scene.domains[1].model
    original = ph.StepInputs.zeros(particle_model.particle_count)
    book = ph.MaterialPathLedger(
        particle_model, material, initial.domains["nodes"], original, 0.25, settings, context
    )
    expected = np.full(initial.domains["nodes"].position.shape, F(0), dtype=object)
    expected[0, 1] = F(3, 7) if chain else F(1, 3)
    diagnostics = evidence["pre_impulse_diagnostics"]["participants"]["nodes"]
    book.require(
        pre.domains["nodes"],
        trial.effective_inputs.domains["nodes"],
        expected,
        diagnostics,
        sweeps=evidence["pre_position_iterations"],
    )
    with pytest.raises(ph.SolverFailure, match="velocity"):
        book.require(
            post.domains["nodes"],
            trial.effective_inputs.domains["nodes"],
            expected,
            diagnostics,
            sweeps=evidence["pre_position_iterations"],
        )
    ledger = ph.TerminalImpulseLedger(
        scene, pre, requests(), settings, context, registry=base.registry
    )
    for domain, velocity in (("robot", "v"), ("nodes", "velocity")):
        corrupt = base.registry.clone_state(scene, post, context)
        values = getattr(corrupt.domains[domain], velocity)
        index = 1 if domain == "nodes" else 0
        values.flat[index] = np.nextafter(values.flat[index], np.inf)
        with pytest.raises(ph.SolverFailure, match="stored FP32"):
            ledger.require(corrupt)
