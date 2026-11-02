"""Analytic controls for finite capsule shaft and cap loading an elastic triangle."""

from contextlib import ExitStack

import numpy as np

import phymeo as ph


def create(*, reverse=False, cap=False, friction=0.0, mode="host", stack=None, cache=None):
    dtype = np.float64 if mode == "host" else np.float32
    face = ph.CollisionShape(ph.CollisionKey(0, 0 if reverse else 1, 0), "triangle", (11, 12, 13))
    shaft = ph.CollisionShape(
        ph.CollisionKey(0, 1 if reverse else 0, 0), "capsule", (90, 70), radius=0.125
    )
    model = ph.CollisionModel((face, shaft))
    vertices = ((-0.125, 0, 0), (0.375, 0, 0.5)) if cap else ((-0.25, 0, 0), (0.25, 0, 0))
    x = np.array([[-1.0, -1, 0], [1.0, -1, 0], [0, 2.0, 0]], dtype=dtype)
    state = ph.CoupledState(
        ph.ParticleState(x, np.zeros_like(x)),
        ph.RigidBodyState(
            np.array([[0.0, 0, 0.25]], dtype=dtype),
            np.array([[0.0, 0, 0.0, 1.0]], dtype=dtype),
            np.zeros((1, 3), dtype=dtype),
            np.zeros((1, 3), dtype=dtype),
        ),
    )
    operator = (
        None if mode == "host" else stack.enter_context(ph.TyxNormalLayer(mode, cache_root=cache))
    )
    system = ph.CollisionLoadSystem(
        model,
        {face.key: ph.ParticleAttachment((0, 1, 2)), shaft.key: ph.RigidAttachment(0, vertices)},
        {(face.key, shaft.key): ph.NormalLayer(0.25, 16.0, 0.0, friction, 0.1)},
        layer_operator=operator,
    )
    pm = ph.ParticleModel(np.ones(3, dtype=dtype), np.zeros(3, dtype=dtype))
    rm = ph.RigidBodyModel(
        np.array([2.0], dtype=dtype), np.ones((1, 3), dtype=dtype), np.zeros(3, dtype=dtype)
    )
    return system, state, (pm, rm)


def actual_step(mode, cache=None):
    with ExitStack() as stack:
        system, state, models = create(mode=mode, stack=stack, cache=cache)
        work = stack.enter_context(
            ph.CollisionWorkspace(system.model, mode, cache_root=cache, broad_phase="bvh")
        )
        if mode == "host":
            solvers = ph.XPBDSolver(), ph.RigidSemiImplicitSolver()
        else:
            solvers = (
                stack.enter_context(ph.TyxXPBDSolver(mode=mode, cache_root=cache)),
                stack.enter_context(ph.TyxRigidSemiImplicitSolver(mode=mode, cache_root=cache)),
            )
        h = 1 / 64
        frame = system.step(
            state,
            ph.CoupledInputs.zeros(3, 1, dtype=state.particles.position.dtype),
            models,
            solvers,
            h,
            work,
        )
        return frame, work.stats()
