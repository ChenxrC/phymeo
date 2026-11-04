"""Read actual multi-body momenta and support cuts in Newton XPBD and TYX. No timing ranking."""

import argparse
import json
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

H = 1 / 128
STEPS = 16


@dataclass(frozen=True)
class Ball:
    key: int
    m: float
    r: float
    x: tuple
    v: tuple = (0, 0, 0)


@dataclass(frozen=True)
class World:
    key: int
    balls: tuple
    g: tuple
    floor: float | None
    e: float


def cases():
    return (
        World(
            17,
            (Ball(81, 3, 0.3125, (0.375, 0.5, 0)), Ball(4, 1, 0.3125, (0, 0, 0), (1.2, 1.6, 0))),
            (0, 0, 0),
            None,
            0.5,
        ),
        World(
            23,
            (
                Ball(8, 1, 0.25, (0, 0, 0.25)),
                Ball(5, 2, 0.25, (0, 0, 0.75)),
                Ball(2, 1, 0.25, (0, 0, 1.25)),
            ),
            (0, 0, -8),
            0,
            0,
        ),
        World(
            4,
            (Ball(81, 2, 0.25, (0, 0, 3), (1, 0, 0)), Ball(4, 1, 0.25, (7, 0, 3), (0, 1, 0))),
            (0, 0, -8),
            0,
            0,
        ),
    )


def phymeo(worlds, device, cache):
    import phymeo as ph
    from phymeo.backends.tyx_sphere_island import SphereIslandWorld, TyxSphereIslandBatch

    assert "newton" not in sys.modules
    model = []
    for w in worlds:
        balls = tuple((b.key, ph.FreeSphere(b.m, b.r, b.x, b.v, w.g)) for b in w.balls)
        model.append((w.key, SphereIslandWorld(balls, w.floor, w.e)))
    order = [(w.key, b.key) for w in worlds for b in w.balls]
    history = []
    contacts = []
    with TyxSphereIslandBatch(
        model, "cuda" if device == "cuda:0" else "interpret", cache_root=cache
    ) as batch:

        def take():
            s = batch.snapshot()
            return [(*s[w].bodies[b].position, *s[w].bodies[b].velocity) for w, b in order]

        history.append(take())
        for _ in range(STEPS):
            rows = batch.step(H)
            assert all(r.accepted for r in rows.values()), rows
            history.append(take())
            contacts.append(
                {
                    w: dict(
                        position_residual_m=r.position_residual_m,
                        velocity_residual_m_s=r.velocity_residual_m_s,
                        momentum_residual_n_s=r.momentum_residual_n_s,
                        kinetic_change_contact_j=r.contact_kinetic_change_j,
                        rows=[
                            dict(
                                first=c.first,
                                second=c.second,
                                normal=c.normal,
                                gap_m=c.gap_m,
                                normal_impulse_n_s=c.normal_impulse_n_s,
                                point_first=c.point_first,
                                point_second=c.point_second,
                            )
                            for c in r.contacts
                        ],
                    )
                    for w, r in rows.items()
                }
            )
        counts = batch.stats()
    return (
        np.array(history, np.float32),
        contacts,
        dict(
            phymeo=version("phymeo"),
            tyx=version("tyx-compute"),
            module=ph.__file__,
            device=device,
            solver_sweeps=96,
            counts=counts,
        ),
    )


def newton(worlds, device, cache):
    import newton as nt
    import warp as wp

    assert (
        "phymeo" not in sys.modules
        and version("newton") == "1.7.0"
        and version("warp-lang") == "1.17.0"
    )
    wp.config.kernel_cache_dir = str(cache.resolve())
    wp.init()
    dev = wp.get_device(device)
    assert str(dev) == device and (device != "cuda:0" or dev.is_cuda)
    wp.set_device(dev)
    builder = nt.ModelBuilder(gravity=(0, 0, -8), up_axis=nt.Axis.Z)
    owners = {}
    body_index = {}
    for w in worlds:
        builder.begin_world(label=f"world_{w.key}", gravity=w.g)
        cfg = builder.ShapeConfig(
            density=0, mu=0, mu_torsional=0, mu_rolling=0, restitution=w.e, margin=0, gap=1 / 64
        )
        if w.floor is not None:
            shape = builder.add_ground_plane(height=w.floor, cfg=cfg, label=f"plane_{w.key}")
            owners[shape] = (w.key, None)
        for b in w.balls:
            body = builder.add_link(
                xform=wp.transform(b.x, wp.quat_identity()),
                com=(0, 0, 0),
                mass=b.m,
                inertia=np.eye(3) * (2 * b.m * b.r**2 / 5),
                lock_inertia=True,
                label=f"b_{w.key}_{b.key}",
            )
            builder.add_articulation([builder.add_joint_free(body)])
            shape = builder.add_shape_sphere(body, radius=b.r, cfg=cfg, label=f"s_{w.key}_{b.key}")
            owners[shape] = (w.key, b.key)
            body_index[w.key, b.key] = body
        builder.end_world()
    model = builder.finalize(device=dev)
    keys = [(w.key, b.key) for w in worlds for b in w.balls]
    assert model.body_count == model.joint_count == len(keys) and model.world_count == len(worlds)
    shape_world = model.shape_world.numpy()
    for shape, (wid, bid) in owners.items():
        assert worlds[int(shape_world[shape])].key == wid
        assert int(model.shape_body.numpy()[shape]) == (-1 if bid is None else body_index[wid, bid])
    read_order = [body_index[k] for k in keys]
    masses = model.body_mass.numpy()
    twists = np.zeros((len(keys), 6), np.float32)
    for w in worlds:
        for b in w.balls:
            i = body_index[w.key, b.key]
            assert masses[i] == b.m
            twists[i, :3] = b.v
    state, next_state = model.state(), model.state()
    state.joint_qd.assign(twists.ravel())
    nt.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = nt.solvers.SolverXPBD(
        model,
        iterations=64,
        rigid_contact_relaxation=0.8,
        rigid_contact_con_weighting=False,
        angular_damping=0,
        enable_restitution=True,
    )
    pipe = nt.CollisionPipeline(model, broad_phase="sap", rigid_contact_max=64, soft_contact_max=0)
    contact = pipe.contacts()
    obs = solver.observables(
        kinds=[nt.solvers.SolverBase.ObservableKind.CONTACT_F], contacts=contact
    )

    def take():
        q = state.body_q.numpy()[read_order]
        v = state.body_qd.numpy()[read_order]
        assert np.isfinite(q).all() and np.isfinite(v).all()
        return np.column_stack((q[:, :3], v[:, :3]))

    history = [take()]
    contact_log = []
    for _ in range(STEPS):
        state.clear_forces()
        pipe.collide(state, contact)
        solver.step(state, next_state, None, contact, H, observables=obs)
        state, next_state = next_state, state
        wp.synchronize_device(dev)
        history.append(take())
        count = int(contact.rigid_contact_count.numpy()[0])
        assert 0 <= count <= contact.rigid_contact_max
        shapes_a = contact.rigid_contact_shape0.numpy()[:count]
        shapes_b = contact.rigid_contact_shape1.numpy()[:count]
        f = obs.contact_f.numpy()[:count, :3]
        n = contact.rigid_contact_normal.numpy()[:count]
        group = {w.key: {"rows": []} for w in worlds}
        for a, b, direction, force in zip(shapes_a, shapes_b, n, f, strict=True):
            wi, loca = owners[int(a)]
            wj, locb = owners[int(b)]
            assert wi == wj
            group[wi]["rows"].append(
                dict(
                    shape0=int(a),
                    shape1=int(b),
                    first=loca,
                    second=locb,
                    native_normal=direction.tolist(),
                    native_force_on_first_n=force.tolist(),
                )
            )
        contact_log.append(group)
    return (
        np.stack(history),
        contact_log,
        dict(
            newton=version("newton"),
            warp=version("warp-lang"),
            module=nt.__file__,
            device=device,
            device_name=dev.name,
            iterations=64,
            rigid_contact_relaxation=0.8,
            weighting=False,
        ),
    )


def measures(worlds, state):
    lookup = {(w.key, b.key): i for i, (w, b) in enumerate((w, b) for w in worlds for b in w.balls)}
    direction = np.array([0.6, 0.8, 0])
    a = state[:, lookup[17, 4]].astype(float)
    b = state[:, lookup[17, 81]].astype(float)
    j_a = -float((a[1, 3:] - a[0, 3:]) @ direction)
    j_b = 3 * float((b[1, 3:] - b[0, 3:]) @ direction)
    kinetic = 0.5 * np.sum(a[:, 3:] ** 2, axis=1) + 1.5 * np.sum(b[:, 3:] ** 2, axis=1)
    p = a[:, 3:] + 3 * b[:, 3:]
    impact = dict(
        impulse_from_small_n_s=j_a,
        impulse_from_large_n_s=j_b,
        observed_restitution=float((b[1, 3:] - a[1, 3:]) @ direction / 2),
        kinetic_loss_j=float(kinetic[0] - kinetic[1]),
        max_linear_momentum_error_n_s=float(np.max(np.linalg.norm(p - p[0], axis=1))),
    )
    top, mid, bottom = (state[:, lookup[23, i]].astype(float) for i in (2, 5, 8))
    deficits = np.stack(
        [m * (np.diff(s[:, 5]) + 8 * H) for s, m in ((top, 1), (mid, 2), (bottom, 1))]
    )
    above = np.cumsum(deficits, axis=0) / H
    gaps = np.column_stack(
        (
            bottom[:, 2] - 0.25,
            np.linalg.norm(mid[:, :3] - bottom[:, :3], axis=1) - 0.5,
            np.linalg.norm(top[:, :3] - mid[:, :3], axis=1) - 0.5,
        )
    )
    forces = above[:, -8:].mean(axis=1)
    support = dict(
        mean_cut_load_top_middle_ground_n=forces.tolist(),
        all_cut_loads_n=above.tolist(),
        max_penetration_m=float(max(0, -gaps.min())),
        max_last8_speed_m_s=float(
            np.max(abs(np.stack((top[-8:, 3:], mid[-8:, 3:], bottom[-8:, 3:]))))
        ),
    )
    free_error = 0.0
    for ball in worlds[2].balls:
        init = np.array(ball.x)
        v0 = np.array(ball.v)
        ticks = np.arange(len(state))[:, None]
        q = init + ticks * H * v0 + np.array(worlds[2].g) * H * H * ticks * (ticks + 1) / 2
        free_error = max(free_error, float(np.max(abs(state[:, lookup[4, ball.key], :3] - q))))
    gates = dict(
        impact=abs(j_a - 2.25) < 0.02
        and abs(j_b - 2.25) < 0.02
        and abs(impact["observed_restitution"] - 0.5) < 0.02,
        energy=abs(impact["kinetic_loss_j"] - 1.125) < 0.02,
        momentum=impact["max_linear_momentum_error_n_s"] < 0.001,
        support=bool(
            np.all(abs(forces - np.array([8, 24, 32])) <= 0.01 * np.array([8, 24, 32]))
            and support["max_penetration_m"] <= 2e-5
        ),
        free_flight=free_error < 2e-6,
    )
    return dict(impact=impact, support=support, free_position_max_error_m=free_error, gates=gates)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--engine", choices=("phymeo", "newton"), required=True)
    cli.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    cli.add_argument("--cache-root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    a = cli.parse_args()
    if a.out.exists():
        cli.error("fresh output required")
    a.cache_root.mkdir(parents=True, exist_ok=True)
    worlds = cases()
    state, contacts, meta = (phymeo if a.engine == "phymeo" else newton)(
        worlds, a.device, a.cache_root
    )
    physical = measures(worlds, state)
    result = dict(
        engine=a.engine,
        device=a.device,
        meta=meta,
        keys=[(w.key, b.key) for w in worlds for b in w.balls],
        dt_s=H,
        steps=STEPS,
        states_xyz_v_m_s=state.tolist(),
        native_contacts=contacts,
        observed=physical,
    )
    with a.out.open("x", encoding="utf8") as f:
        json.dump(result, f, allow_nan=False, indent=2)
    print(
        physical["gates"],
        physical["impact"],
        physical["support"]["mean_cut_load_top_middle_ground_n"],
    )


if __name__ == "__main__":
    main()
