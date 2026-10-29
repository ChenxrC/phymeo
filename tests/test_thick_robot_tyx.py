"""Required actual pinned TYX paths; no backend fallback or host-state replacement."""

from dataclasses import replace

import numpy as np
import pytest
from test_terminal_impulse import ObservedProgram, ObservedWorkspace
from test_thick_robot_contact import build, diag, jaw_window, lifecycle, review_side_case, step
from test_thick_robot_transaction import FaultProgram

import phymeo as ph
from phymeo.core.scene import fingerprint

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("mu", [1.0, 0.25])
def test_actual_two_jaw_support_and_slip(tyx_mode, mu):
    r = step(build(tyx_mode, scenario="A", mu=mu))
    d = diag(r)
    assert r.report.backend == "tyx:" + tyx_mode
    np.testing.assert_allclose(d["held_impulses"], 1 / 6, atol=2e-5, rtol=0)
    np.testing.assert_allclose(
        r.state.domains["nodes"].velocity[:, 2], 0.0 if mu == 1.0 else -1 / 6, atol=2e-5, rtol=0
    )
    assert d["terminal_ledger"]["ledger"]["compute_precision"] == "float32"


@pytest.mark.parametrize("material", [False, True])
def test_actual_three_corner_impact_and_material(tyx_mode, material):
    r = step(build(tyx_mode, material=material))
    d = diag(r)
    np.testing.assert_allclose(d["held_impulses"], 0.2, atol=2e-5, rtol=0)
    np.testing.assert_allclose(r.state.domains["nodes"].velocity[:, 1], -0.6, atol=2e-5, rtol=0)
    order = np.argsort(np.argmax(d["anchor_material"][:, :3], axis=1))
    np.testing.assert_allclose(
        d["terminal_impulses"][order],
        [0.2] * 3 if material else [13 / 80, 1 / 5, 19 / 80],
        atol=2e-5,
        rtol=0,
    )
    if material:
        np.testing.assert_allclose(
            d["held_state"].domains["nodes"].velocity[:, 0],
            [1 / 48, 0.0, -1 / 48],
            atol=2e-5,
            rtol=0,
        )
    else:
        np.testing.assert_allclose(
            r.state.domains["nodes"].velocity[:, 0], [131 / 160, 4 / 5, 25 / 32], atol=2e-5, rtol=0
        )


def test_actual_partial_overlap_lifecycle(tyx_mode):
    lifecycle(tyx_mode, 0.25)


def test_actual_thickness_window_and_rebuild(tyx_mode):
    for mu in (1.0, 0.25):
        for h in (0.125, 0.0625):
            for rebuild in range(2):
                jaw_window(tyx_mode, h, mu, rebuild)
    # One fresh C rebuild in addition to the complete lifecycle test.
    lifecycle(tyx_mode, 0.25, 1)


def test_actual_failure_and_all_trials_count(tyx_mode, monkeypatch):
    case = build(tyx_mode, base_transform=ObservedProgram)
    r = step(case)
    d = diag(r)
    owner = case[3].base
    assert len(owner.workspaces) == d["held_replays"] + d["terminal_replays"]
    assert all(
        w.counts["predict"] == w.counts["terminal"] == w.counts["close"] == 1
        for w in owner.workspaces
    )
    assert r.report.substeps[0].diagnostics["execution"]["completed_launches"] > 0
    for state in (d["held_state"], r.state):
        for name, fields in (("robot", ("q", "v")), ("nodes", ("position", "velocity"))):
            for field in fields:
                v = getattr(state.domains[name], field)
                assert v.dtype == np.float64
                np.testing.assert_array_equal(v, v.astype(np.float32).astype(np.float64))
    assert d["terminal_ledger"]["applications"] == 1

    # A real backend result altered by one host ULP must fail exact FP32 storage.
    class Program(ObservedProgram):
        def create_workspace(self, *args):
            ws = Workspace(self, self.base.create_workspace(*args), len(self.workspaces) + 1)
            self.workspaces.append(ws)
            return ws

    class Workspace(ObservedWorkspace):
        def result(self):
            s = self.native.result()
            v = s.domains["nodes"].velocity.copy()
            if np.any(v):
                index = tuple(np.argwhere(v != 0)[0])
                v[index] = np.nextafter(v[index], np.inf)
                return ph.SceneState(
                    {**s.domains, "nodes": ph.ParticleState(s.domains["nodes"].position, v)},
                    s.history,
                )
            return s

    bad = step(build(tyx_mode, base_transform=Program), accepted=False)
    assert "FP32" in bad.report.reason
    scene, state, inputs, p, settings, context = build(
        tyx_mode, base_transform=lambda b: FaultProgram(b, "terminal", 1)
    )
    before = fingerprint((state, inputs))
    with ph.SceneStepper(scene, p, replace(settings, substeps=2), context, p.registry) as runtime:
        failed = runtime.step(state, inputs, 0.5)
        assert not failed.report.accepted and failed.report.completed_substeps == 1
        assert "terminal" in failed.report.reason and fingerprint(failed.state) == fingerprint(
            state
        )
        p.base.failure = None
        retried = runtime.step(state, inputs, 0.5)
        assert retried.report.accepted, retried.report.reason
    assert fingerprint((state, inputs)) == before


@pytest.mark.parametrize("feature", ["edge", "point"])
def test_review_actual_side_feature(tyx_mode, feature):
    case = review_side_case(tyx_mode, feature)
    r = step(case)
    d = diag(r)
    expected = case[1].domains["nodes"].velocity.copy()
    expected[:2] = 0.0
    assert r.report.backend == "tyx:" + tyx_mode
    np.testing.assert_allclose(r.state.domains["nodes"].velocity, expected, atol=2e-5, rtol=0)
    np.testing.assert_allclose(
        r.state.domains["nodes"].position,
        case[1].domains["nodes"].position + 0.25 * expected,
        atol=2e-5,
        rtol=0,
    )
    for name, fields in (("robot", ("q", "v")), ("nodes", ("position", "velocity"))):
        for field in fields:
            value = getattr(r.state.domains[name], field)
            np.testing.assert_array_equal(value, value.astype(np.float32).astype(np.float64))
    np.testing.assert_allclose(d["terminal_world"], 0.0, atol=2e-5, rtol=0)
    assert d["held_replays"] > 1 and all(c["status"] == "certified" for c in d["ccd"])
    assert d["terminal_ledger"]["ledger"]["compute_precision"] == "float32"
