"""Independent planar 2R geometry, per-pose engine measurements, and future CCD acceptance."""

import argparse
import json
from importlib.metadata import version
from math import asin, cos, dist, pi, sin, sqrt
from pathlib import Path

import numpy as np

ASSET = Path(__file__).with_name("two_hinge_ellipse.xml")
RADIUS = 1 / 16
FIRST = asin((sqrt(322) - 2) / 16) / pi
DELTA = 1 / 1024
TIMES = (0.0, FIRST - DELTA, FIRST + DELTA, 0.5, 1 - FIRST - DELTA, 1 - FIRST + DELTA, 1.0)


def q_at(t, sign):
    return np.array([sign * pi * t, -2 * sign * pi * t])


def center_at(t, sign):
    # Link one has world angle +theta and link two has world angle -theta.
    return [1.5 * cos(pi * t), sign * 0.5 * sin(pi * t), 0.0]


def targets(sign):
    return ("north", "south") if sign == 1 else ("south", "north")


def observation(t, sign, centers, gaps, statuses, tolerance):
    """Check actual world centers and surface gaps against the independent 2R mechanism."""
    want = {"tool": center_at(t, sign), "north": [0, 0.5, 0], "south": [0, -0.5, 0]}
    for name, point in want.items():
        np.testing.assert_allclose(centers[name], point, atol=tolerance, rtol=0)
    target, control = targets(sign)
    for name in (target, control):
        expected = dist(want["tool"], want[name]) - 2 * RADIUS
        measured = dist(centers["tool"], centers[name]) - 2 * RADIUS
        assert abs(expected - measured) <= 2 * tolerance, (t, name, expected, measured)
        assert statuses[name] == (expected < 0), (t, sign, name, statuses, expected)
        # At exact center coincidence the native normal/anchor selection may be degenerate.
        if gaps[name] and not (name == target and t == 0.5):
            assert abs(min(gaps[name]) - expected) <= 4 * tolerance, (t, name, gaps, expected)
    return dict(
        t=t,
        q_rad=q_at(t, sign).tolist(),
        centers_m={k: list(map(float, v)) for k, v in centers.items()},
        center_error_m=dist(centers["tool"], want["tool"]),
        distance_gaps_m={
            k: dist(centers["tool"], centers[k]) - 2 * RADIUS for k in (target, control)
        },
        collision_gaps_m=gaps,
        occupied=statuses,
    )


def study_phy():
    import phymeo as ph

    asset = ph.load_mjcf(ASSET)
    assert asset.joint_names == ("shoulder", "elbow")
    keys = {c.name: c.shape.key for c in asset.collisions}
    assert all(c.shape.kind == "sphere" and c.shape.radius == RADIUS for c in asset.collisions)
    model = asset.collision_model
    forward = {}
    with ph.CollisionWorkspace(model, "host") as work:
        policy = ph.CollisionPolicy(margin=4)

        def pairs(paths):
            out = ph.detect(model, paths, policy, work)
            assert out.status == "ok"
            return {
                name: next(
                    p for p in out.pairs if {p.first, p.second} == {keys["tool"], keys[name]}
                )
                for name in ("north", "south")
            }

        for sign in (1, -1):
            rows = []
            for t in TIMES:
                paths = asset.collision_motion(q_at(t, sign))
                read = pairs(paths)
                centers = {name: paths[key].start[0] for name, key in keys.items()}
                gaps = {k: [v.witness.gap_m] for k, v in read.items()}
                statuses = {k: v.status == "overlap" for k, v in read.items()}
                rows.append(observation(t, sign, centers, gaps, statuses, 1e-12))
            start = asset.collision_motion(q_at(0, sign))
            end = asset.collision_motion(q_at(1, sign))
            chord = {k: ph.CollisionMotion(path.start, end[k].start) for k, path in start.items()}
            target, control = targets(sign)
            linear = pairs(chord)[target]
            assert linear.status == "separated" and linear.sweep.first_t is None
            try:
                complete = asset.collision_motion(q_at(0, sign), q_at(1, sign))
            except ValueError as exc:
                # Record today's named capability gap. A future implementation must pass the oracle.
                if not all(word in str(exc) for word in ("shoulder", "elbow", "one moving hinge")):
                    raise
                continuous = dict(status="unavailable", reason=str(exc))
            else:
                full = pairs(complete)
                hit, clear = full[target], full[control]
                assert hit.status == "hit" and clear.status == "separated"
                lo, hi = hit.sweep.first_t
                assert lo <= FIRST <= hi and hi - lo <= 1e-7
                assert clear.sweep.first_t is None
                continuous = dict(status="certified", interval=[lo, hi])
            forward[str(sign)] = dict(rows=rows, chord_status=linear.status, continuous=continuous)
    return dict(engine="phymeo", device="host", first_t=FIRST, signs=forward)


def named(labels, suffix):
    return next(i for i, label in enumerate(labels) if label.rsplit("/", 1)[-1] == suffix)


def study_newton(device):
    import newton
    import warp as wp

    assert version("newton") == "1.7.0" and version("warp-lang") == "1.17.0"
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0, 0, 0))
    builder.rigid_gap = 0.0
    builder.add_mjcf(str(ASSET), floating=False)
    model = builder.finalize(device=device)
    state = model.state()
    owner = model.shape_body.numpy()
    local = model.shape_transform.numpy()
    sphere = model.shape_type.numpy() == int(newton.GeoType.SPHERE)
    base = int(named(model.body_label, "base"))
    forearm = int(named(model.body_label, "forearm"))
    fixed = np.flatnonzero((owner == base) & sphere)
    (tool,) = np.flatnonzero((owner == forearm) & sphere)
    assert len(fixed) == 2
    ids = {
        "north": int(fixed[np.argmax(local[fixed, 1])]),
        "south": int(fixed[np.argmin(local[fixed, 1])]),
        "tool": int(tool),
    }
    assert np.all(model.shape_scale.numpy()[list(ids.values()), 0] == RADIUS)
    coords = model.joint_q_start.numpy()
    dof = [int(coords[named(model.joint_label, name)]) for name in ("shoulder", "elbow")]
    defaults = state.joint_q.numpy()
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", reduce_contacts=False, rigid_contact_max=32
    )
    stream = pipeline.contacts()
    ds = wp.empty(stream.rigid_contact_max, dtype=float, device=device)
    assert all(str(a.device) == device for a in (state.body_q, ds, stream.rigid_contact_count))
    forward = {}
    for sign in (1, -1):
        rows = []
        for t in TIMES:
            q = defaults.copy()
            q[dof] = q_at(t, sign)
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            pipeline.collide(state, stream)
            newton.eval_rigid_contact_kinematics(model, state, stream, out_distance=ds)
            body = state.body_q.numpy()
            centers = {}
            for name, shape in ids.items():
                transform = body[owner[shape]]
                pose = wp.transform(wp.vec3(*transform[:3]), wp.quat(*transform[3:]))
                center = wp.transform_point(pose, wp.vec3(*local[shape, :3]))
                centers[name] = list(center)
            n = int(stream.rigid_contact_count.numpy()[0])
            assert 0 <= n < stream.rigid_contact_max
            a, b = stream.rigid_contact_shape0.numpy()[:n], stream.rigid_contact_shape1.numpy()[:n]
            values = ds.numpy()[:n]
            gaps = {}
            for name in ("north", "south"):
                k = ids[name]
                mask = ((a == tool) & (b == k)) | ((a == k) & (b == tool))
                gaps[name] = values[mask].astype(float).tolist()
            occupied = {k: bool(v) for k, v in gaps.items()}
            rows.append(observation(t, sign, centers, gaps, occupied, 3e-6))
        forward[str(sign)] = dict(rows=rows, contact_bracket_from_samples=[TIMES[1], TIMES[2]])
    return dict(engine="newton-1.7.0", device=device, shape_ids=ids, first_t=FIRST, signs=forward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("host", "cpu", "cuda:0"), required=True)
    device = parser.parse_args().device
    result = study_phy() if device == "host" else study_newton(device)
    print(json.dumps(result, allow_nan=False))
