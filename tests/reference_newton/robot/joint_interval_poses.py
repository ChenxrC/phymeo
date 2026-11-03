"""Native Newton FK and sampled capsule/sphere contacts on an imported articulated tree."""

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path

import newton
import numpy as np
import warp as wp


@wp.kernel
def capsule_axis(
    bodies: wp.array(dtype=wp.transform),
    shapes: wp.array(dtype=wp.transform),
    scales: wp.array(dtype=wp.vec3),
    body: int,
    shape: int,
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    sign = 2.0 * float(i) - 1.0
    pose = wp.transform_multiply(bodies[body], shapes[shape])
    out[i] = wp.transform_point(pose, wp.vec3(0.0, 0.0, sign * scales[shape][1]))


def index(labels, name):
    ids = [i for i, v in enumerate(labels) if v == name or v.endswith("/" + name)]
    assert len(ids) == 1, (name, labels)
    return ids[0]


def study(device):
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
    b.rigid_gap = 0.0
    b.add_mjcf(str(Path(__file__).with_name("joint_interval_arm.xml")), floating=False)
    model = b.finalize(device=device)
    state = model.state()
    rotor = index(model.body_label, "rotor")
    base = index(model.body_label, "base")
    owner = model.shape_body.numpy()
    kind = model.shape_type.numpy()
    ptr = int(np.flatnonzero((owner == rotor) & (kind == int(newton.GeoType.CAPSULE)))[0])
    balls = np.flatnonzero((owner == base) & (kind == int(newton.GeoType.SPHERE)))
    local = model.shape_transform.numpy()
    north = int(balls[np.argmax(local[balls, 1])])
    south = int(balls[np.argmin(local[balls, 1])])
    times = model.joint_q_start.numpy()
    names = ("advance", "diagonal", "spin", "extend", "bend", "independent_spin")
    coordinate = {name: int(times[index(model.joint_label, name)]) for name in names}
    defaults = state.joint_q.numpy().copy()
    out = wp.empty(2, dtype=wp.vec3, device=device)
    contact = newton.CollisionPipeline(
        model, broad_phase="nxn", reduce_contacts=False, rigid_contact_max=256
    )
    stream = contact.contacts()
    ds = wp.empty(stream.rigid_contact_max, dtype=float, device=device)

    def sample(q):
        values = defaults.copy()
        for name, value in q.items():
            values[coordinate[name]] = value
        state.joint_q.assign(values)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        wp.launch(
            capsule_axis,
            dim=2,
            inputs=[state.body_q, model.shape_transform, model.shape_scale, rotor, ptr],
            outputs=[out],
            device=device,
        )
        assert str(out.device) == str(device) and str(state.body_q.device) == str(device)
        point = out.numpy()
        spin = q.get("spin", 0.0)
        pivot = np.array(
            [q.get("advance", 0.0), 0.6 * q.get("diagonal", 0.0), 0.8 * q.get("diagonal", 0.0)]
        )
        direction = np.array([np.cos(spin), np.sin(spin), 0.0])
        expected = pivot + np.array([0.5, 1.0])[:, None] * direction
        error = min(np.max(np.abs(point - expected)), np.max(np.abs(point[::-1] - expected)))
        assert error < 4e-6, (q, point, expected, error)
        return point, float(error), pivot

    threshold = math.acos(1 / 6) / math.pi
    arcs = []
    mixed = []
    for sign, target in ((1, north), (-1, south)):
        rows = []
        for t in (0.0, threshold - 0.002, threshold + 0.002, 0.48, 1.0):
            q = sign * np.pi * t
            point, error, _ = sample(dict(spin=q))
            contact.collide(state, stream)
            newton.eval_rigid_contact_kinematics(model, state, stream, out_distance=ds)
            n = int(stream.rigid_contact_count.numpy()[0])
            assert 0 <= n < stream.rigid_contact_max and str(
                stream.rigid_contact_count.device
            ) == str(device)
            a, c = stream.rigid_contact_shape0.numpy()[:n], stream.rigid_contact_shape1.numpy()[:n]
            mask = ((a == ptr) & (c == target)) | ((a == target) & (c == ptr))
            values = ds.numpy()[:n][mask]
            center = np.array([0.0, sign * 0.75, 0.0])
            axis = point[1] - point[0]
            closest = point[0] + axis * np.clip((center - point[0]) @ axis / (axis @ axis), 0, 1)
            measured = float(np.linalg.norm(closest - center) - 0.125)
            expected_contact = t in (threshold + 0.002, 0.48)
            assert (measured < 0) == expected_contact and bool(len(values)) == expected_contact, (
                q,
                t,
                values,
                measured,
            )
            if len(values):
                np.testing.assert_allclose(float(min(values)), measured, atol=2e-5, rtol=0)
            rows.append(
                dict(
                    t=t,
                    q=q,
                    ends_m=point.tolist(),
                    FK_max_error_m=error,
                    surface_gap_m=measured,
                    contact_gap_m=values.tolist(),
                )
            )
        arcs.append(
            dict(
                sign=sign,
                root_sphere_id=target,
                contact_bracket_from_samples=[rows[1]["t"], rows[2]["t"]],
                rows=rows,
            )
        )
    independent = index(model.body_label, "independent")
    for t in (0.0, 0.23, 0.5, 0.82, 1.0):
        q = dict(
            advance=-0.125 + 0.75 * t,
            diagonal=0.25 - 0.5 * t,
            spin=-0.25 + 1.375 * t,
            independent_spin=0.3 - 1.175 * t,
            extend=0.375,
            bend=0.2,
        )
        point, error, pivot = sample(q)
        bodyq = state.body_q.numpy()[independent]
        measured = np.array(wp.quat_to_matrix(wp.quat(*bodyq[3:]))).reshape(3, 3)
        c, s = np.cos(q["independent_spin"]), np.sin(q["independent_spin"])
        np.testing.assert_allclose(measured, [[c, 0, s], [0, 1, 0], [-s, 0, c]], atol=3e-6, rtol=0)
        np.testing.assert_allclose(bodyq[:3], [4, 0, 0], atol=3e-6, rtol=0)
        mixed.append(
            dict(
                t=t,
                q=q,
                pivot_m=pivot.tolist(),
                capsule_ends_m=point.tolist(),
                FK_max_error_m=error,
                independent_body_xyzw=bodyq[3:].tolist(),
            )
        )
    return dict(
        device=str(device),
        pointer_id=ptr,
        body_id=rotor,
        analytic_first_t=threshold,
        authored_from_to_parameter=math.sqrt(35) / 4 - 1,
        sampled_arcs=arcs,
        moving_pivot=mixed,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = p.parse_args()
    assert version("newton") == "1.7.0" and version("warp-lang") == "1.17.0"
    with wp.ScopedDevice(args.device):
        result = study(args.device)
    print(json.dumps(result, allow_nan=False))
