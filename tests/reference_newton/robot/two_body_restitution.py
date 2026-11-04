"""Two free unequal spheres, from the same touching state, with independent collision/impulse paths."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import newton
import numpy as np
import warp as wp

import phymeo as ph


def comparison(device, coefficient, normal):
    h, radius = 1 / 128, 1 / 8
    masses = np.array([1.0, 2.0])
    axis = np.asarray(normal, dtype=float)
    common = np.array([0.125, -0.25, 0.0])
    centers = np.array([-radius * axis, radius * axis])
    velocity = np.array([2 * axis + common, -axis + common])
    inertias = 0.4 * masses * radius**2
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0, restitution=coefficient, gap=0.01)
    signatures = []
    states = {}
    attachments, shapes, poses = {}, [], {}
    for i in range(2):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*centers[i]), wp.quat_identity()),
            mass=masses[i],
            inertia=wp.mat33(*np.diag([inertias[i]] * 3).ravel()),
            lock_inertia=True,
        )
        signatures.append(builder.add_shape_sphere(body, radius=radius, cfg=cfg))
        key = ph.CollisionKey(0, i, 0)
        shapes.append(ph.CollisionShape(key, "sphere", (0,), radius=radius))
        poses[key] = ph.CollisionMotion(centers[i : i + 1])
        name = str(i)
        rigid = ph.RigidBodyModel(
            np.array([1 / masses[i]]), np.full((1, 3), inertias[i]), np.zeros(3)
        )
        movable = ph.FloatingRobotModel(rigid, np.zeros(3))
        states[name] = (
            movable,
            ph.FloatingRobotState(
                np.r_[centers[i], 0.0, 0.0, 0.0, 1.0], np.r_[velocity[i], np.zeros(3)]
            ),
            ph.FloatingRobotInputs.zeros(0),
        )
        attachments[name] = (np.array([0]), (-centers[i])[None, :])
    geom = ph.CollisionModel(shapes)
    with ph.CollisionWorkspace(geom) as work:
        (pair,) = ph.detect(geom, poses, ph.CollisionPolicy(), work).pairs
    assert pair.status == "touching"
    rows = [(pair, {s.key: (str(i), 0) for i, s in enumerate(shapes)})]
    step = ph.step_floating_contacts(
        states, attachments, rows, [0.0], [0.0], h, restitution=coefficient
    )
    assert step.report.accepted, step.report.reason
    phy = np.stack([step.states[str(i)].v[:3] for i in range(2)])
    model = builder.finalize(device)
    s, out = model.state(), model.state()
    s.body_qd.assign(np.column_stack((velocity, np.zeros((2, 3)))).astype(np.float32))
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(s, contacts)
    solver = newton.solvers.SolverXPBD(
        model, iterations=10, angular_damping=0.0, enable_restitution=True
    )
    solver.step(s, out, model.control(), contacts, h)
    count = int(contacts.rigid_contact_count.numpy()[0])
    assert count >= 1
    pairs = zip(
        contacts.rigid_contact_shape0.numpy()[:count],
        contacts.rigid_contact_shape1.numpy()[:count],
        strict=True,
    )
    assert {tuple(sorted(map(int, row))) for row in pairs} == {tuple(sorted(signatures))}
    actual = out.body_qd.numpy()[:, :3].astype(float)
    oracle = np.stack([common - 2 * coefficient * axis, common + coefficient * axis])
    records = {}
    for name, final, tol in (("phymeo", phy, 1e-9), ("newton", actual, 3e-4)):
        impulse = masses[:, None] * (final - velocity)
        change = float(np.sum(masses[:, None] * (final**2 - velocity**2)) / 2)
        records[name] = dict(
            final_velocity=final.tolist(), impulses_ns=impulse.tolist(), energy_change_j=change
        )
        np.testing.assert_allclose(final, oracle, rtol=0, atol=tol)
        np.testing.assert_allclose(impulse.sum(0), 0, rtol=0, atol=tol)
        np.testing.assert_allclose(change, -3 * (1 - coefficient**2), rtol=0, atol=5 * tol)
    return dict(
        restitution=coefficient,
        axis=axis.tolist(),
        dt_s=h,
        masses_kg=masses.tolist(),
        initial_velocity=velocity.tolist(),
        newton_contacts=count,
        **records,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or version("newton") != "1.7.0":
        raise ValueError("expected a new report path and Newton 1.7.0")
    wp.init()
    with wp.ScopedDevice(args.device):
        data = [
            comparison(args.device, c, n) for c in (0.0, 0.5, 1.0) for n in ((1, 0, 0), (0, 0, -1))
        ]
    with args.out.open("x", encoding="utf8") as stream:
        json.dump({"device": args.device, "cases": data}, stream, allow_nan=False)
    print(f"verified {len(data)} free two-body impacts on {args.device}")
