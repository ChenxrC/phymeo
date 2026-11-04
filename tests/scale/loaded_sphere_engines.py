"""Isolated public TYX and native Newton rigid-sphere/plane adapters for the same observer."""

import sys
from importlib.metadata import version

import numpy as np
from sphere_plane_cases import MASS, PLANE, RADIUS, H


class Phymeo:
    def __init__(self, worlds, device, cache, *, force_audit):
        import tyx as tx

        import phymeo as ph

        assert "newton" not in sys.modules
        self.tx, self.worlds = tx, worlds
        self.keys = tuple(w.key for w in worlds)
        models = [
            (
                w.key,
                ph.SpherePlaneWorld(
                    MASS,
                    RADIUS,
                    (0, 0, w.z),
                    (w.vx, 0, w.vz),
                    (0, 0, w.gravity),
                    PLANE,
                    w.restitution,
                ),
            )
            for w in worlds
        ]
        self.work = ph.TyxWorldBatch(
            models, "cuda" if device == "cuda:0" else "interpret", cache_root=cache
        )
        self.audit = force_audit
        self.meta = dict(
            phymeo=version("phymeo"),
            tyx=version("tyx-compute"),
            source=ph.__file__,
            device=device,
            method="TyxWorldBatch staged free step and sphere-plane projection",
        )
        self.steps = 0

    def read(self):
        obs = self.work.kinematics()
        assert (
            obs.keys == self.keys
            and np.all(obs.ticks == self.steps)
            and obs.widths == (3,) * len(self.keys)
        )
        return obs.values

    def advance(self):
        status = self.work.step(H)
        assert tuple(status) == self.keys and all(status.values())
        self.steps += 1
        return self.read()

    def details(self):
        return {
            k: dict(normal_impulse_n_s=s.normal_impulse_n_s)
            for k, s in self.work.snapshot().items()
        }

    def counts(self):
        native = self.tx.runtime_stats()
        v = self.work.stats()
        return dict(
            solver_calls=self.steps,
            device_submissions=v["submitted_kernels"],
            explicit_sync=v["synchronizations"],
            accepted_world_steps=v["accepted_world_steps"],
            rejected_world_steps=v["rejected_world_steps"],
            state_read_b=v["kinematics_downloaded_bytes"],
            status_read_b=v["status_downloaded_bytes"],
            native_cuda_kernels=native.cuda_kernel_launches,
            native_d2h_b=native.cuda_d2h_bytes,
            native_h2d_b=native.cuda_h2d_bytes,
        )

    def close(self):
        self.work.close()


class Newton:
    def __init__(self, worlds, device, cache, *, force_audit):
        import newton as nt
        import warp as wp

        assert (
            "phymeo" not in sys.modules
            and version("newton") == "1.7.0"
            and version("warp-lang") == "1.17.0"
        )
        wp.config.kernel_cache_dir = str(cache.resolve())
        wp.init()
        self.dev = wp.get_device(device)
        assert str(self.dev) == device and (device != "cuda:0" or self.dev.is_cuda)
        wp.set_device(self.dev)
        self.wp, self.nt = wp, nt
        self.request = {w.key: w for w in worlds}
        b = nt.ModelBuilder(gravity=(0, 0, -8), up_axis=nt.Axis.Z)
        self.shape_owner = {}
        self.geometry = {}
        for w in worlds:
            b.begin_world(label=f"world_{w.key}", gravity=(0, 0, w.gravity))
            cfg = b.ShapeConfig(
                density=0,
                mu=0,
                mu_torsional=0,
                mu_rolling=0,
                restitution=w.restitution,
                margin=0,
                gap=1 / 64,
            )
            ground = b.add_ground_plane(height=PLANE, cfg=cfg, label=f"plane_{w.key}")
            body = b.add_link(
                xform=wp.transform((0.0, 0.0, w.z), wp.quat_identity()),
                mass=MASS,
                com=(0, 0, 0),
                inertia=np.eye(3) * (2 * MASS * RADIUS**2 / 5),
                lock_inertia=True,
                label=f"ball_{w.key}",
            )
            free = b.add_joint_free(body)
            b.add_articulation([free])
            ball = b.add_shape_sphere(body, radius=RADIUS, cfg=cfg, label=f"sphere_{w.key}")
            self.shape_owner[ball] = (w.key, "sphere")
            self.shape_owner[ground] = (w.key, "plane")
            self.geometry[w.key] = (ground, body, ball)
            b.end_world()
        m = self.model = b.finalize(device=self.dev)
        assert m.world_count == m.body_count == m.joint_count == len(
            worlds
        ) and m.shape_count == 2 * len(worlds)
        body_world = m.body_world.numpy()
        shape_world = m.shape_world.numpy()
        types = m.shape_type.numpy()
        radii = m.shape_scale.numpy()
        grav = m.gravity.numpy()
        self.keys = tuple(worlds[int(i)].key for i in body_world)
        for w in worlds:
            ground, body, ball = self.geometry[w.key]
            assert shape_world[ground] == shape_world[ball] == body_world[body]
            assert types[ground] == int(nt.GeoType.PLANE) and types[ball] == int(nt.GeoType.SPHERE)
            assert radii[ball, 0] == RADIUS and grav[body_world[body], 2] == w.gravity
        np.testing.assert_array_equal(m.body_mass.numpy(), np.full(len(worlds), MASS))
        np.testing.assert_array_equal(m.body_com.numpy(), np.zeros((len(worlds), 3)))
        np.testing.assert_array_equal(
            m.joint_type.numpy(), np.full(len(worlds), int(nt.JointType.FREE))
        )
        self.state, self.next = m.state(), m.state()
        twist = np.zeros((len(worlds), 6), np.float32)
        for index, key in enumerate(self.keys):
            w = self.request[key]
            twist[index, [0, 2]] = (w.vx, w.vz)
        self.state.joint_qd.assign(twist.ravel())
        nt.eval_fk(m, self.state.joint_q, self.state.joint_qd, self.state)
        self.pipe = nt.CollisionPipeline(
            m, broad_phase="sap", rigid_contact_max=4 * len(worlds), soft_contact_max=0
        )
        self.contacts = self.pipe.contacts()
        self.solver = nt.solvers.SolverXPBD(
            m,
            iterations=4,
            rigid_contact_relaxation=1.0,
            rigid_contact_con_weighting=False,
            angular_damping=0,
            enable_restitution=any(w.restitution > 0 for w in worlds),
        )
        self.obs = (
            self.solver.observables(
                kinds=[nt.solvers.SolverBase.ObservableKind.CONTACT_F], contacts=self.contacts
            )
            if force_audit
            else None
        )
        self.steps, self.syncs, self.read_bytes, self.reads = 0, 0, 0, 0
        wp.synchronize_device(self.dev)
        self.meta = dict(
            newton=version("newton"),
            warp=version("warp-lang"),
            device=device,
            device_name=self.dev.name,
            source=nt.__file__,
            method="CollisionPipeline(sap) + SolverXPBD",
            iterations=4,
            contact_relaxation=1,
            contact_weighting=False,
            margin_m=0,
            speculative_shape_gap_m=1 / 64,
            shape_source=self.geometry,
            native_contact_force_observable=bool(force_audit),
        )

    def read(self):
        q = self.state.body_q.numpy()
        v = self.state.body_qd.numpy()
        self.reads += 2
        if self.dev.is_cuda:
            self.read_bytes += q.nbytes + v.nbytes
        result = np.empty((len(self.keys), 6), np.float32)
        result[:, :3], result[:, 3:] = q[:, :3], v[:, :3]
        return result

    def advance(self):
        self.state.clear_forces()
        self.pipe.collide(self.state, self.contacts)
        self.solver.step(self.state, self.next, None, self.contacts, H, observables=self.obs)
        self.state, self.next = self.next, self.state
        self.wp.synchronize_device(self.dev)
        self.syncs += 1
        self.steps += 1
        return self.read()

    def details(self):
        contacts = self.contacts
        count = int(contacts.rigid_contact_count.numpy()[0])
        assert 0 <= count <= contacts.rigid_contact_max
        a = contacts.rigid_contact_shape0.numpy()[:count]
        b = contacts.rigid_contact_shape1.numpy()[:count]
        force = self.obs.contact_f.numpy()[:count, :3].astype(float)
        normals = contacts.rigid_contact_normal.numpy()[:count]
        rows = {k: [] for k in self.keys}
        for (sa, sb), f, n in zip(zip(a, b, strict=True), force, normals, strict=True):
            wa, role = self.shape_owner[int(sa)]
            wb, other = self.shape_owner[int(sb)]
            assert wa == wb and {role, other} == {"plane", "sphere"}
            applied = f if role == "sphere" else -f
            rows[wa].append(
                dict(
                    shape0=int(sa),
                    shape1=int(sb),
                    normal=n.tolist(),
                    force_on_sphere_n=applied.tolist(),
                )
            )
        return rows

    def counts(self):
        return dict(
            solver_calls=self.steps,
            collision_calls=self.steps,
            explicit_sync=self.syncs,
            public_state_reads=self.reads,
            state_read_b=self.read_bytes,
        )

    def close(self):
        # Check physical dimensions outside the shared timed application output.
        v = self.state.body_qd.numpy()
        q = self.state.body_q.numpy()
        np.testing.assert_allclose(v[:, 3:], 0, rtol=0, atol=2e-5)
        np.testing.assert_allclose(q[:, 3:], np.tile([0, 0, 0, 1], (len(q), 1)), rtol=0, atol=2e-5)
