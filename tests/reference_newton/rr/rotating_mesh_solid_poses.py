"""Newton 1.7 native mesh pairs at independently prescribed rotating poses."""

import argparse
import json
import math
import runpy
from importlib.metadata import version
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def poses(name):
    if name == "u-wall":
        u = runpy.run_path(str(HERE / "baseline.py"))["_u_geometry"]
        assets = (u([1, 1, 1], False), u([0.25, 0.25, 0.5], True))
        fractions = (0, 0.125, 0.25, 0.5, 0.75, 0.875, 1)
    else:
        cube = runpy.run_path(str(HERE / "mesh_round_poses.py"))["geometry"]("box")
        assets = (cube, (cube[0] / 16, cube[1]))
        fractions = (0, 0.0625, 0.125, 0.15625, 0.1875, 0.25, 0.5, 1)
    states = []
    for t in fractions:
        phase = t * math.pi / 2
        c, s = math.cos(phase), math.sin(phase)
        if name == "u-wall":
            outer = np.array([0.125 * t, 0.0625 * t, 0])
            inner = outer + np.array([0.375, 0, 0])
            reach = 0.375 + 0.5 * (c + s)
            relation = "separated" if reach < 1 else "boundary_overlap"
            wall_gap = 1 - reach
        else:
            outer = np.zeros(3)
            inner = np.array([0.75 + s, 1 - c, 0])
            extent = (c + s) / 16
            relation = (
                "first_contains_second"
                if max(inner[:2]) + extent < 1
                else "separated"
                if max(inner[:2]) - extent > 1
                else "boundary_overlap"
            )
            wall_gap = None
        states.append((t, phase, outer, inner, relation, wall_gap))
    return assets, states


def observe(name, device):
    import newton
    import warp as wp

    assets, states = poses(name)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = builder.ShapeConfig(gap=0, margin=0, has_particle_collision=False)
    shapes = []
    for v, indices in assets:
        body = builder.add_body()
        shapes.append(builder.add_shape_mesh(body, mesh=newton.Mesh(v, indices), cfg=cfg))
    model = builder.finalize(device=device)
    ids = model.shape_body.numpy()[shapes]
    assert len(set(map(int, ids))) == 2 and np.all(model.body_mass.numpy()[ids] > 0)
    assert [newton.GeoType(int(k)).name for k in model.shape_type.numpy()[shapes]] == [
        "MESH",
        "MESH",
    ]
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", reduce_contacts=False, rigid_contact_max=1024
    )
    state = model.state()
    contacts = pipeline.contacts()
    distances = wp.empty(contacts.rigid_contact_max, dtype=float, device=device)
    points = [wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=device) for _ in shapes]
    reader = runpy.run_path(str(HERE / "baseline.py"))["_newton_witnesses"]
    rows = []
    for t, phase, outer, inner, relation, wall_gap in states:
        rotations = ([0, 0, 0, 1], [0, 0, math.sin(phase / 2), math.cos(phase / 2)])
        requested = np.array(
            [list(p) + list(q) for p, q in zip((outer, inner), rotations, strict=True)]
        )
        native = state.body_q.numpy()
        native[ids] = requested
        state.body_q.assign(native)
        pipeline.collide(state, contacts)
        newton.eval_rigid_contact_kinematics(
            model,
            state,
            contacts,
            out_distance=distances,
            out_point0_world=points[0],
            out_point1_world=points[1],
        )
        wp.synchronize_device(device)
        count = int(contacts.rigid_contact_count.numpy()[0])
        assert 0 <= count < contacts.rigid_contact_max
        actual = state.body_q.numpy()[ids]
        np.testing.assert_allclose(actual, requested, rtol=0, atol=2e-7)
        samples = (
            reader(contacts, count, distances, points, shapes, True)["contacts"] if count else []
        )
        offsets = [
            a.numpy()[:count]
            for a in (contacts.rigid_contact_offset0, contacts.rigid_contact_offset1)
        ]
        margins = [
            a.numpy()[:count]
            for a in (contacts.rigid_contact_margin0, contacts.rigid_contact_margin1)
        ]
        for i, w in enumerate(samples):
            swapped = w["observed_pair_ids"][0] != shapes[0]
            w["contact_margins_m"] = []
            for side, label in enumerate(("a", "b")):
                native_side = 1 - side if swapped else side
                local = offsets[native_side][i]
                q = actual[side, 3:]
                world = local + 2 * np.cross(q[:3], np.cross(q[:3], local) + q[3] * local)
                anchor = np.asarray(w[label + "_support_world_m"])
                w[label + "_anchor_world_m"] = anchor.tolist()
                w[label + "_offset_world_m"] = world.tolist()
                w[label + "_support_world_m"] = (anchor + world).tolist()
                w["contact_margins_m"].append(float(margins[native_side][i]))
            gap = np.subtract(w["b_support_world_m"], w["a_support_world_m"]) @ w["normal_a_to_b"]
            np.testing.assert_allclose(gap, w["gap_m"], rtol=0, atol=2e-6)
        row = dict(
            t=t,
            phase_rad=phase,
            body_q_xyzw=actual.tolist(),
            independent_solid_relation=relation,
            native_contact_count=count,
            native_contacts=samples,
        )
        if wall_gap is not None:
            row["independent_inner_wall_gap_m"] = wall_gap
        if samples:
            row["minimum_native_gap_m"] = min(w["gap_m"] for w in samples)
        rows.append(row)
    return rows


def main():
    import sys

    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert version("newton") == "1.7.0"
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    dev = wp.get_device("cpu")
    with wp.ScopedDevice(dev):
        data = {name: observe(name, dev) for name in ("u-wall", "contained-exit")}
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(
            dict(newton=version("newton"), device=str(dev), poses=data),
            out,
            allow_nan=False,
            indent=2,
        )
    for name, rows in data.items():
        print(
            name,
            [(r["t"], r["independent_solid_relation"], r["native_contact_count"]) for r in rows],
        )


if __name__ == "__main__":
    main()
