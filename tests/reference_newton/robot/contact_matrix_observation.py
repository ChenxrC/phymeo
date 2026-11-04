"""Compare complete shape-pair observations on the same Newton XPBD physical contact stream."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import newton
import numpy as np
import warp as wp
from contact_wrench import window

import phymeo as ph
from phymeo.geometry.collision_witness import ConvexContactPoint
from phymeo.solvers.collision_loads import LoadSite, PairLoad, SurfaceLoads


def observe(model, state, contacts, obs, sensor, shapes):
    count = int(contacts.rigid_contact_count.numpy()[0])
    body_of = model.shape_body.numpy()
    pose = state.body_q.numpy()
    keys = {
        s: ph.CollisionKey(0, int(body_of[s]) if body_of[s] >= 0 else len(pose) + s, s)
        for s in shapes
    }
    indices = [
        v.numpy()[:count] for v in (contacts.rigid_contact_shape0, contacts.rigid_contact_shape1)
    ]
    local = [
        p.numpy()[:count] + off.numpy()[:count]
        for p, off in (
            (contacts.rigid_contact_point0, contacts.rigid_contact_offset0),
            (contacts.rigid_contact_point1, contacts.rigid_contact_offset1),
        )
    ]
    normal = contacts.rigid_contact_normal.numpy()[:count]
    force0 = obs.contact_f.numpy()[:count, :3]
    pairs = []
    for i in range(count):
        first, second = (int(ids[i]) for ids in indices)
        world = []
        for side, s in enumerate((first, second)):
            p = local[side][i]
            if body_of[s] >= 0:
                tf = pose[body_of[s]]
                p = wp.transform_point(
                    wp.transform(wp.vec3(*tf[:3]), wp.quat(*tf[3:])), wp.vec3(*p)
                )
            world.append(tuple(map(float, p)))
        site = LoadSite(
            ConvexContactPoint(*world, (), ()),
            1.0,
            0.0,
            float(abs(force0[i] @ normal[i])),
            *world,
            tuple(-force0[i]),
            0.0,
        )
        pairs.append(
            PairLoad(
                keys[first], keys[second], tuple(normal[i]), 0.0, site.force_n, 0.0, 0.0, (site,)
            )
        )
    record = SurfaceLoads(ph.CoupledInputs.zeros(0, 0), tuple(pairs))
    axes = [keys[s] for s in shapes]
    matrix = ph.measure_contact_matrix(record, axes, axes)
    error = {}
    for field, actual in (
        ("total_force", matrix.total_force_n),
        ("total_force_friction", matrix.total_friction_n),
        ("force_matrix", matrix.force_n),
        ("force_matrix_friction", matrix.friction_n),
        ("position_matrix", matrix.position_m),
    ):
        table = getattr(sensor, field).numpy()
        want = []
        for s in shapes:
            row = list(sensor.sensing_indices).index(s)
            want.append(
                table[row]
                if field.startswith("total")
                else [table[row, list(sensor.counterpart_indices[row]).index(k)] for k in shapes]
            )
        expected = np.asarray(want)
        error[field] = float(np.max(np.abs(expected - actual)))
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-6)
    return dict(
        contacts=count,
        errors=error,
        weight_n=matrix.weight_n.tolist(),
        pair_force_n=matrix.force_n.tolist(),
        native_total_n=sensor.total_force.numpy().tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if version("newton") != "1.7.0" or args.out.exists():
        raise ValueError("require Newton 1.7.0 and a fresh output")
    wp.init()
    samples = []
    with wp.ScopedDevice(args.device):
        for sliding in (False, True):
            rows = []
            window(
                newton,
                wp,
                args.device,
                sliding,
                observe=lambda *values: rows.append(observe(*values)),
            )
            samples.append(rows)
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(dict(device=args.device, samples=samples), stream, allow_nan=False)
    print(
        "same-contact-stream Newton/Phymeo matrices matched:",
        sum(map(len, samples)),
        "physical steps",
    )


if __name__ == "__main__":
    main()
