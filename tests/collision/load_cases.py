"""Independent mass, gap and initial-velocity controls for the three load families."""

from contextlib import ExitStack
from itertools import product

import numpy as np

import phymeo as ph
from phymeo.solvers.collision_loads import (
    CollisionLoadSystem,
    NormalLayer,
    ParticleAttachment,
    RigidAttachment,
)


def scene(family, dtype=np.float64):
    cube = tuple(product((-0.125, 0.125), repeat=3))
    lower = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0, 0.5, 0)]
    if family == "RR":
        kinds = ("hull",) * 3
        points, pv, inv = [], [], []
        bodies = [(0, 0, 0), (0, 0, 19 / 64), (0, 0, 77 / 128)]
        velocities, im = [(0, 0, 0)] * 3, [0, 1, 1]
        gravity = (0, 0, -1)
        bindings = [RigidAttachment(i, cube) for i in range(3)]
    elif family == "RD":
        kinds = ("sphere", "triangle")
        points, pv, inv = lower, [(0, 0, 0)] * 3, [1.0] * 3
        bodies, velocities, im = [(0, 0, 0.25)], [(0, 0, -0.5)], [2]
        gravity = (0, 0, 0)
        bindings = [RigidAttachment(0, ((0, 0, 0),)), ParticleAttachment((0, 1, 2))]
    else:
        kinds = ("triangle", "triangle")
        points = lower + [(0, 0, 0.125), (0.3, 0, 0.25), (0, 0.3, 0.25)]
        pv, inv = [(0, 0, 0.25)] * 3 + [(0, 0, -0.25)] * 3, [1.0] * 6
        bodies, velocities, im = [], [], []
        gravity = (0, 0, 0)
        bindings = [ParticleAttachment((0, 1, 2)), ParticleAttachment((3, 4, 5))]
    shapes = [
        ph.CollisionShape(
            ph.CollisionKey(0, 10 + i, 0),
            kind,
            tuple(range(100 * i, 100 * i + {"hull": 8, "sphere": 1, "triangle": 3}[kind])),
            radius=0.125 if kind == "sphere" else 0.0,
        )
        for i, kind in enumerate(kinds)
    ]

    def vec(values, n):
        return np.array(values, dtype=dtype).reshape(-1, n)

    p = ph.ParticleState(vec(points, 3), vec(pv, 3))
    r = ph.RigidBodyState(
        vec(bodies, 3),
        vec([(0, 0, 0, 1)] * len(im), 4),
        vec(velocities, 3),
        vec([(0, 0, 0)] * len(im), 3),
    )
    state = ph.CoupledState(p, r)
    models = (
        ph.ParticleModel(np.array(inv, dtype=dtype), np.array(gravity, dtype=dtype)),
        ph.RigidBodyModel(
            np.array(im, dtype=dtype), vec([(1, 1, 1)] * len(im), 3), np.array(gravity, dtype=dtype)
        ),
    )
    model = ph.CollisionModel(shapes)
    layer = NormalLayer(1 / 16, 128.0, 0.25)
    laws = {(a.key, b.key): layer for a, b in zip(shapes, shapes[1:])}
    sys = CollisionLoadSystem(
        model, {s.key: v for s, v in zip(shapes, bindings, strict=True)}, laws
    )
    edges = [(i + j, i + (j + 1) % 3) for i in range(0, len(inv), 3) for j in range(3)]
    constraints = None
    if edges:
        idx = np.array(edges, dtype=np.int64)
        rest = np.linalg.norm(p.position[idx[:, 1]] - p.position[idx[:, 0]], axis=1)
        constraints = ph.DistanceConstraints(idx, rest, np.full(len(edges), 0.01, dtype=dtype))
    return sys, state, models, constraints


def window(family, mode="host", cache=None):
    system, state, models, constraints = scene(family, np.float32 if mode != "host" else np.float64)
    # Two independently moving panels can present a second edge after the first release.
    steps = 128 if family == "DD" else 42
    history = []
    with ExitStack() as stack:
        work = stack.enter_context(ph.CollisionWorkspace(system.model, mode, cache_root=cache))
        if mode == "host":
            ps, rs = (
                ph.XPBDSolver(constraints=constraints, tolerance=5e-5),
                ph.RigidSemiImplicitSolver(),
            )
            layer = None
        else:
            layer = stack.enter_context(ph.TyxNormalLayer(mode, cache_root=cache))
            system = ph.CollisionLoadSystem(
                system.model, system.bindings, system.layers, layer_operator=layer
            )
            ps = (
                stack.enter_context(
                    ph.TyxXPBDSolver(constraints, mode=mode, tolerance=5e-5, cache_root=cache)
                )
                if family != "RR"
                else None
            )
            rs = (
                stack.enter_context(ph.TyxRigidSemiImplicitSolver(mode=mode, cache_root=cache))
                if family != "DD"
                else None
            )
        for k in range(steps):
            loads = ph.CoupledInputs.zeros(
                state.particles.particle_count,
                state.rigid.body_count,
                dtype=state.particles.position.dtype,
            )
            if family == "RR" and k >= 6:
                loads.rigid.force[2, 2] = 4.0
            frame = system.step(state, loads, models, (ps, rs), 1 / 64, work)
            state = frame.state
            history.append(frame)
        stats = work.stats()
        if layer is not None:
            stats["normal_layer"] = layer.stats()
    return history, stats
