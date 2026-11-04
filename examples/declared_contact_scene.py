"""Run either declared contact scene through the same geometry, material and step entry points."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.solvers.collision_loads import NormalLayer


def box_stack():
    """Two co-located stacks; only the top box in world 42 receives the external pull."""
    b = ContactSceneBuilder(gravity=(0, 0, -1))
    for world in (7, 42):
        for body, z, mass in ((10, 0, None), (20, 19 / 64, 1), (30, 77 / 128, 1)):
            b.add_box(body, [1 / 8] * 3, world=world, mass=mass, position=(0, 0, z))
        b.pair(10, 20, NormalLayer(1 / 16, 128.0, 0.25), world=world)
        b.pair(20, 30, NormalLayer(1 / 16, 128.0, 0.25), world=world)
        b.pair(10, 30, None, world=world)
    return b, 1 / 64, 14, [(2, (42, 30), (0, 0, 4))]


def tool_sheet():
    """An off-centre tetrahedron above a two-face sheet with four independently moving nodes."""
    b = ContactSceneBuilder(gravity=(0, 0, 0))
    b.add_convex(
        7,
        (
            (1 / 16, -1 / 8, -1 / 8),
            (-1 / 8, -1 / 16, 1 / 8),
            (1 / 8, -1 / 16, 1 / 8),
            (0, 1 / 8, 1 / 8),
        ),
        mass=2.0,
        inertia_diagonal=(1 / 16,) * 3,
        position=(0, 0, 5 / 32),
        world=3,
    )
    b.add_sheet(
        13,
        [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]],
        [(0, 1, 2), (0, 2, 3)],
        node_masses=(1, 2, 3, 4),
        edge_compliance=1 / 256,
        world=3,
    )
    b.pair(7, 13, NormalLayer(1 / 16, 128.0), world=3)
    b.pair(13, 13, None, world=3)
    return b, 1 / 512, 4, []


def physical(state):
    return dict(
        body_position_m=state.rigid.position.tolist(),
        orientation_xyzw=state.rigid.orientation.tolist(),
        body_velocity_m_s=state.rigid.velocity.tolist(),
        omega_rad_s=state.rigid.angular_velocity.tolist(),
        node_position_m=state.particles.position.tolist(),
        node_velocity_m_s=state.particles.velocity.tolist(),
    )


def contact(loads):
    return dict(
        pairs=[asdict(row) for row in loads.contacts],
        body_force_n=loads.inputs.rigid.force.tolist(),
        body_torque_nm=loads.inputs.rigid.torque.tolist(),
        node_force_n=loads.inputs.particles.force.tolist(),
    )


def shape_key(key):
    return [key.world, key.body, key.shape]


def geometry(result):
    return dict(
        status=result.status,
        backend=result.backend,
        required_candidates=result.required_candidates,
        pairs=[
            dict(
                first=shape_key(p.first),
                second=shape_key(p.second),
                status=p.status,
                reason=p.reason,
            )
            for p in result.pairs
        ],
    )


def run(case, backend, cache_root):
    scene, dt, steps, drives = {"stack": box_stack, "tool-sheet": tool_sheet}[case]()
    with scene.build(backend=backend, cache_root=cache_root) as runtime:
        observed = runtime.sample()
        history = []
        output = dict(
            schema="phymeo-declared-contact-scene-v1",
            case=case,
            backend=backend,
            dt_s=dt,
            rigid_rows=[[*key, row] for key, row in runtime.rigid_rows.items()],
            particle_rows=[[*key, list(rows)] for key, rows in runtime.particle_rows.items()],
            initial=physical(runtime.state),
            initial_geometry=geometry(observed.geometry),
            initial_loads=contact(observed.loads),
            steps=history,
        )
        for i in range(steps):
            external = runtime.inputs()
            for start, address, force in drives:
                if i >= start:
                    external.rigid.force[runtime.rigid_rows[address]] += force
            frame = runtime.step(dt, external)
            history.append(
                dict(
                    entry_s=i * dt,
                    end_s=(i + 1) * dt,
                    entry_loads=contact(frame.loads),
                    applied_rigid_force_n=frame.external.rigid.force.tolist(),
                    state=physical(frame.state),
                    swept_geometry=geometry(frame.path),
                    rigid_backend=None
                    if frame.rigid_report is None
                    else frame.rigid_report.backend,
                    particle_backend=None
                    if frame.particle_report is None
                    else frame.particle_report.backend,
                )
            )
        output["stats"] = runtime.stats()
        return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=("stack", "tool-sheet"), required=True)
    p.add_argument("--backend", choices=("host", "interpret", "cuda"), default="host")
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    out = run(a.case, a.backend, a.cache_root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf8") as stream:
        json.dump(out, stream, allow_nan=False, indent=2)
    print(f"{a.case} {a.backend} {len(out['steps'])} accepted physical substeps -> {a.output}")


if __name__ == "__main__":
    main()
