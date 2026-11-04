"""Compare independent free box/sphere motion under the same finite normal force law."""

import argparse
import json
from pathlib import Path

import numpy as np

HALF = np.array([0.5, 0.375, 0.125])
RADIUS = 1 / 8
H = 1 / 256
COUNT = 48


def rotated(q, vectors):
    xyz = q[:3]
    return vectors + 2 * np.cross(xyz, np.cross(xyz, vectors) + q[3] * vectors)


def native(center, device):
    import newton as nt
    import warp as wp

    builder = nt.ModelBuilder(gravity=(0, 0, 0))
    config = builder.ShapeConfig(density=0, gap=0, margin=0)
    x, y, z = HALF
    inertia = 2 / 3 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
    a = builder.add_body(mass=2.0, inertia=np.diag(inertia), lock_inertia=True, label="box")
    b = builder.add_body(
        xform=wp.transform(center, wp.quat_identity()),
        mass=1.0,
        inertia=np.eye(3) * (2 / 5) * RADIUS**2,
        lock_inertia=True,
        label="ball",
    )
    box = builder.add_shape_box(a, hx=x, hy=y, hz=z, cfg=config)
    ball = builder.add_shape_sphere(b, radius=RADIUS, cfg=config)
    model = builder.finalize(device=device)
    np.testing.assert_array_equal(
        model.shape_type.numpy()[[box, ball]], [nt.GeoType.BOX, nt.GeoType.SPHERE]
    )
    np.testing.assert_allclose(model.body_inertia.numpy()[a], np.diag(inertia), atol=1e-7)
    dims = model.shape_scale.numpy().astype(float)
    solver = nt.solvers.SolverSemiImplicit(model, angular_damping=0.0, enable_tri_contact=False)
    state, after, control = model.state(), model.state(), model.control()
    history = []
    for _ in range(COUNT):
        poses = state.body_q.numpy().astype(float)
        q = poses[a, 3:] / np.linalg.norm(poses[a, 3:])
        delta = poses[b, :3] - poses[a, :3]
        local = rotated(np.r_[-q[:3], q[3]], delta)
        lever = rotated(q, np.clip(local, -dims[box], dims[box]))
        separation = delta - lever
        length = np.linalg.norm(separation)
        gap = length - dims[ball, 0]
        assert gap > 0
        force = max(0.0, 128 * (1 / 16 - gap))
        f = force * separation / length
        w = np.zeros((2, 6), np.float32)
        w[a] = np.r_[-f, np.cross(lever, -f)]
        w[b, :3] = f
        state.clear_forces()
        state.body_f.assign(w)
        solver.step(state, after, control, None, H)
        state, after = after, state
        history.append([force, state.body_q.numpy().tolist(), state.body_qd.numpy().tolist()])
    return history


def engine(center):
    import phymeo as ph

    builder = ph.ContactSceneBuilder(gravity=(0, 0, 0))
    builder.add_box(2, HALF, mass=2)
    builder.add_sphere(9, RADIUS, mass=1, position=center)
    builder.pair(2, 9, ph.NormalLayer(1 / 16, 128))
    history = []
    with builder.build() as scene:
        for _ in range(COUNT):
            f = scene.step(H)
            rb = f.state.rigid
            force = sum(c.force_n for c in f.loads.contacts)
            history.append(
                [
                    force,
                    np.column_stack([rb.position, rb.orientation]).tolist(),
                    np.column_stack([rb.velocity, rb.angular_velocity]).tolist(),
                ]
            )
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("output already exists")
    result = {"device": args.device, "dt_s": H, "cases": {}}
    for name, center in (
        ("face", (1 / 16, 0, 9 / 32)),
        ("edge", (0.5 + 3 / 32, 0.375 + 1 / 8, 1 / 32)),
    ):
        ours = engine(center)
        theirs = native(center, args.device)
        fa, fb = (np.array([row[0] for row in trace]) for trace in (ours, theirs))
        poses = [np.array([row[1] for row in tr]) for tr in (ours, theirs)]
        twists = [np.array([row[2] for row in tr]) for tr in (ours, theirs)]
        diff = np.abs(twists[0] - twists[1])
        result["cases"][name] = dict(
            center=center,
            phymeo=ours,
            newton=theirs,
            loaded=[int(np.count_nonzero(fa)), int(np.count_nonzero(fb))],
            first_forces=[float(fa[0]), float(fb[0])],
            last_forces=[float(fa[-1]), float(fb[-1])],
            max_position_m=float(np.max(np.abs(poses[0][:, :, :3] - poses[1][:, :, :3]))),
            max_linear_m_s=float(diff[:, :, :3].max()),
            max_angular_rad_s=float(diff[:, :, 3:].max()),
            max_force_n=float(np.max(np.abs(fa - fb))),
        )
    with args.out.open("x", encoding="utf8") as f:
        json.dump(result, f, allow_nan=False)
    for case in result["cases"].values():
        print({k: v for k, v in case.items() if k not in ("phymeo", "newton")})
        np.testing.assert_allclose(case["first_forces"], 4, atol=3e-5)
        assert 4 < min(case["loaded"]) <= max(case["loaded"]) < COUNT
        assert case["last_forces"] == [0, 0]
        assert case["max_position_m"] < 1e-4 and case["max_linear_m_s"] < 1e-3
        assert case["max_angular_rad_s"] < 1e-2 and case["max_force_n"] < 2e-3


if __name__ == "__main__":
    main()
