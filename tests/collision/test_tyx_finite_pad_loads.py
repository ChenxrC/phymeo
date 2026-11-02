"""Actual finite pad friction law with device state evolution and history."""

import json
from contextlib import ExitStack
from dataclasses import asdict

import numpy as np
import pytest
from test_finite_pad_loads import pad_state

import phymeo as ph

pytestmark = pytest.mark.tyx


def test_mass_slides_off_finite_pad(tyx_mode, tmp_path):
    with ExitStack() as stack:
        layer = stack.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path / "cache"))
        system, s = pad_state((0.375, 0.0, 3 / 32), (1.0, 0.0, 0.0), operator=layer)

        def f32(a):
            return np.asarray(a, dtype=np.float32)

        state = ph.CoupledState(
            ph.ParticleState(f32(s.particles.position), f32(s.particles.velocity)),
            ph.RigidBodyState(
                f32(s.rigid.position),
                f32(s.rigid.orientation),
                f32(s.rigid.velocity),
                f32(s.rigid.angular_velocity),
            ),
        )
        models = (
            ph.ParticleModel(f32([1]), f32([0.0, 0.0, -4.0])),
            ph.RigidBodyModel(f32([0]), f32([[0, 0, 0]]), f32([0, 0, 0])),
        )
        ps = stack.enter_context(ph.TyxSemiImplicitSolver(tyx_mode, cache_root=tmp_path / "cache"))
        rs = stack.enter_context(
            ph.TyxRigidSemiImplicitSolver(tyx_mode, cache_root=tmp_path / "cache")
        )
        ws = stack.enter_context(
            ph.CollisionWorkspace(system.model, tyx_mode, cache_root=tmp_path / "cache")
        )
        frames = []
        for k in range(48):
            frame = system.step(
                state, ph.CoupledInputs.zeros(1, 1, dtype=np.float32), models, (ps, rs), 1 / 128, ws
            )
            frames.append(frame)
            state = frame.state
        first = frames[0]
        np.testing.assert_allclose(
            first.loads.inputs.particles.force[0], [-2.0, 0.0, 4.0], atol=1e-6
        )
        assert first.loads.contacts[0].power_w == pytest.approx(-2.0)
        assert state.particles.position[0, 0] > 0.625
        assert state.particles.velocity[0, 2] < 0.0
        assert not frames[-1].loads.contacts
        assert any(f.loads.contacts and f.loads.contacts[0].normal[0] > 0.1 for f in frames)
        assert all(
            f.path.backend == tyx_mode and f.particle_report.backend == "tyx:" + tyx_mode
            for f in frames
        )
        stats = layer.stats()
        assert stats["sites"] > 0 and stats["completed_launches"] > 0
        (tmp_path / "slide.json").write_text(
            json.dumps(
                dict(steps=[asdict(f) for f in frames], stats=stats),
                default=lambda a: a.tolist(),
                allow_nan=False,
            )
        )
