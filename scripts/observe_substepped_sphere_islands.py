"""Observe matching 1/8 s finite-sphere windows in a native Newton or TYX process."""

import argparse
import json
from pathlib import Path

import numpy as np

from observe_sphere_islands import H, STEPS, Ball, World, newton


def cases():
    return (
        World(11, (Ball(1, 1, 0.25, (0, 0, 0)), Ball(2, 1, 0.25, (0.5, 0, 0), (-1, 2, 0))),
              (0, 0, 0), None, 0),
        World(12, (Ball(1, 1, 0.25, (-1, 0, 0), (16, 0, 0)),
                   Ball(2, 1, 0.25, (1, 0, 0), (-16, 0, 0))), (0, 0, 0), None, 1),
    )


def phy(worlds, device, cache):
    import phymeo as ph
    from phymeo.backends.tyx_sphere_island import SphereIslandWorld, TyxSphereIslandBatch

    specs = [(w.key, SphereIslandWorld(tuple((b.key, ph.FreeSphere(b.m, b.r, b.x, b.v, w.g))
                                             for b in w.balls), restitution=w.e)) for w in worlds]
    order = [(w.key, b.key) for w in worlds for b in w.balls]
    with TyxSphereIslandBatch(specs, "cuda" if device == "cuda:0" else "interpret",
                              cache_root=cache) as batch:
        source = batch.snapshot()
        result = batch.step_substepped(H * STEPS, substeps=STEPS)
        assert all(r.accepted for r in result.values()), result
        advances = {k: [s for s in r.stages if s.kind == "advance"] for k, r in result.items()}
        states = [source] + [{k: stages[i].report.state for k, stages in advances.items()}
                             for i in range(STEPS)]
        series = np.array([[(*frame[k].bodies[b].position, *frame[k].bodies[b].velocity)
                            for k, b in order] for frame in states])
        rows = {k: [dict(kind=s.kind, start_s=s.start_s, end_s=s.end_s,
                         contacts=[vars(c) for c in s.report.contacts],
                         momentum_residual_n_s=s.report.momentum_residual_n_s,
                         contact_kinetic_change_j=s.report.contact_kinetic_change_j)
                    for s in r.stages] for k, r in result.items()}
        return series, rows, dict(device=device, module=ph.__file__, stats=batch.stats())


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--engine", choices=("newton", "phymeo"), required=True)
    cli.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--cache", required=True, type=Path)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("provide a new output path")
    w = cases()
    state, contacts, metadata = (newton if args.engine == "newton" else phy)(w, args.device, args.cache)
    assert state.shape == (STEPS + 1, 4, 6) and np.isfinite(state).all()
    values = {}
    for j, case in enumerate(w):
        v = state[:, j * 2:j * 2 + 2, 3:].astype(float)
        x = state[:, j * 2:j * 2 + 2, :3].astype(float)
        momentum = v.sum(axis=1)  # Both original masses are 1 kg.
        energy = np.sum(v * v, axis=(1, 2)) / 2
        distance = np.linalg.norm(x[:, 1] - x[:, 0], axis=1) - 0.5
        values[case.key] = dict(
            window_delta_p_second_x_n_s=float(v[-1, 1, 0] - v[0, 1, 0]),
            total_kinetic_change_j=float(energy[-1] - energy[0]),
            max_momentum_residual_n_s=float(np.max(np.linalg.norm(momentum - momentum[0], axis=1))),
            min_sampled_gap_m=float(distance.min()), final_gap_m=float(distance[-1]),
            final_state=state[-1, 2 * j:2 * j + 2].tolist(),
        )
    report = dict(engine=args.engine, device=args.device, h_s=H, steps=STEPS, state=state.tolist(),
                  raw_contacts=contacts, observations=values, metadata=metadata)
    with args.output.open("x", encoding="utf8") as f:
        json.dump(report, f, allow_nan=False, indent=2)
    print(json.dumps(values, allow_nan=False))


if __name__ == "__main__":
    main()
