"""Engine-isolated adapters and common checked XYZ/velocity output for the G2 free ball."""

import sys
from collections import Counter
from importlib.metadata import version

import numpy as np

DT = 1 / 128
STEPS = 32
MASS = 2.0
RADIUS = 1 / 8
GRAVITY = (0.0, 0.0, -8.0)
INERTIA = 2 * MASS * RADIUS**2 / 5


def setup_rows(count, asymmetric=False):
    keys = tuple(17 + 3 * i for i in reversed(range(count)))
    z = np.full(count, 2.0, dtype=np.float64)
    vz = np.zeros(count, dtype=np.float64)
    if asymmetric:
        # A qualification-only control makes wrong world-to-body maps observable.
        z += np.arange(count) / 8
        vz = (np.arange(count) + 1) / 16
        vz[1::2] *= -1
    return keys, z, vz


class ApplicationOutput:
    """Apply the same external key order, finite checks and semi-implicit oracle to either engine."""

    def __init__(self, raw_keys, keys, z, vz):
        if len(raw_keys) != len(keys) or len(set(keys)) != len(keys) or set(raw_keys) != set(keys):
            raise ValueError("each registered world must map to exactly one observed body")
        self.keys = tuple(sorted(keys))
        self.order = np.array([raw_keys.index(k) for k in self.keys], np.intp)
        initial = {k: (p, v) for k, p, v in zip(keys, z, vz, strict=True)}
        self.z = np.array([initial[k][0] for k in self.keys])
        self.vz = np.array([initial[k][1] for k in self.keys])
        self.max_position_error = 0.0
        self.max_velocity_error = 0.0

    def read(self, tick, pos, vel):
        n = len(self.keys)
        if (
            pos.shape != (n, 3)
            or vel.shape != (n, 3)
            or pos.dtype != np.float32
            or vel.dtype != np.float32
        ):
            raise ValueError("application requires native FP32 XYZ positions and velocities")
        result = np.empty((n, 6), np.float32)
        result[:, :3] = pos[self.order]
        result[:, 3:] = vel[self.order]
        if not np.isfinite(result).all():
            raise AssertionError("non-finite physical observation")
        want_z = self.z + self.vz * (tick * DT) + GRAVITY[2] * DT**2 * tick * (tick + 1) / 2
        want_vz = self.vz + GRAVITY[2] * DT * tick
        pe = max(float(np.max(np.abs(result[:, 2] - want_z))), float(np.max(np.abs(result[:, :2]))))
        ve = max(
            float(np.max(np.abs(result[:, 5] - want_vz))), float(np.max(np.abs(result[:, 3:5])))
        )
        self.max_position_error = max(self.max_position_error, pe)
        self.max_velocity_error = max(self.max_velocity_error, ve)
        # All fixture values and expected coordinates up through the sampled run are dyadic FP32.
        if pe != 0.0 or ve != 0.0:
            raise AssertionError(
                f"physical trajectory mismatch at step {tick}: position {pe} m, velocity {ve} m/s"
            )
        return result


class PhymeoFall:
    def __init__(self, modules, keys, z, vz, device, cache):
        ph, self.tx = modules
        self.keys = tuple(keys)
        self.device = device
        self._specs = [
            (
                k,
                ph.FreeSphere(
                    mass=MASS,
                    radius=RADIUS,
                    position=(0, 0, x),
                    velocity=(0, 0, v),
                    gravity=GRAVITY,
                ),
            )
            for k, x, v in zip(keys, z, vz, strict=True)
        ]
        self.work = ph.TyxWorldBatch(
            self._specs, "cuda" if device == "cuda:0" else "interpret", cache_root=cache
        )
        self.calls = 0

    def binding(self):
        constants = (
            self.work._props.to_numpy() if self.device == "cuda:0" else self.work._props.copy()
        )
        np.testing.assert_array_equal(constants[:, 0], np.full(len(self.keys), 1 / MASS))
        np.testing.assert_array_equal(constants[:, 1:4], np.tile(GRAVITY, (len(self.keys), 1)))
        np.testing.assert_array_equal(constants[:, 7], np.full(len(self.keys), RADIUS))
        np.testing.assert_array_equal(constants[:, 8], np.full(len(self.keys), MASS))
        return dict(
            implementation="TyxWorldBatch/free sphere",
            device=self.device,
            keys_by_body=self.keys,
            physical_bodies=len(self.keys),
            sphere_radius_m=float(constants[0, 7]),
            mass_kg=float(constants[0, 8]),
            gravity_m_s2=constants[0, 1:4].tolist(),
            position_velocity_read="owned public WorldCheckpoint position and velocity",
            status="one required int32 acceptance per world",
        )

    def read(self, tick):
        snapshot = self.work.snapshot()
        if tuple(snapshot) != self.keys or any(
            s.tick != tick or s.world != k for k, s in snapshot.items()
        ):
            raise AssertionError("wrong accepted state tick or world")
        return np.asarray([s.position for s in snapshot.values()], np.float32), np.asarray(
            [s.velocity for s in snapshot.values()], np.float32
        )

    def advance(self, tick):
        status = self.work.step(DT)
        if len(status) != len(self.keys) or not all(status.values()):
            raise AssertionError("Phymeo rejected a physical world step")
        self.calls += 1
        return self.read(tick)

    def counters(self):
        s = self.work.stats()
        result = {
            k: s[k]
            for k in (
                "submitted_kernels",
                "synchronizations",
                "accepted_world_steps",
                "rejected_world_steps",
                "device_allocations",
                "topology_uploaded_bytes",
                "control_uploaded_bytes",
                "status_downloaded_bytes",
                "snapshot_downloaded_bytes",
            )
        }
        if self.device == "cuda:0":
            n = self.tx.runtime_stats()
            result.update(
                {
                    k: getattr(n, k)
                    for k in (
                        "cuda_kernel_launches",
                        "cuda_h2d_bytes",
                        "cuda_d2h_bytes",
                        "cuda_d2d_bytes",
                    )
                }
            )
        return dict(result, solver_calls=self.calls)

    def events(self, output, tick):
        before = self.counters()
        output.read(tick, *self.advance(tick))
        after = self.counters()
        return dict(
            type="TYX runtime counter delta on one extra audited step",
            delta={k: after[k] - v for k, v in before.items()},
        )

    def close(self):
        self.work.close()


class CountedFall(PhymeoFall):
    def counters(self):
        return dict(
            super().counters(),
            kinematics_downloaded_bytes=self.work.stats()["kinematics_downloaded_bytes"],
        )


class KinematicFall(CountedFall):
    def binding(self):
        result = super().binding()
        result["position_velocity_read"] = (
            "TyxWorldBatch.kinematics: keyed committed ticks and FP32 XYZ/COM velocity"
        )
        return result

    def read(self, tick):
        observed = self.work.kinematics()
        if (
            observed.keys != self.keys
            or observed.widths != (3,) * len(self.keys)
            or not np.all(observed.ticks == tick)
        ):
            raise AssertionError("typed observation did not report the expected committed worlds")
        return observed.values[:, :3], observed.values[:, 3:]


class NewtonFall:
    def __init__(self, modules, keys, z, vz, device, cache):
        nt, self.wp = modules
        wp = self.wp
        self.nt = nt
        self.device = wp.get_device(device)
        self._registry = tuple(keys)
        prototype = nt.ModelBuilder(gravity=GRAVITY)
        body = prototype.add_body(
            xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()),
            mass=MASS,
            com=(0.0, 0.0, 0.0),
            inertia=np.eye(3) * INERTIA,
            lock_inertia=True,
            label="ball",
        )
        cfg = prototype.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
            gap=0.0,
            margin=0.0,
        )
        prototype.add_shape_sphere(body, radius=RADIUS, cfg=cfg, label="sphere")
        builder = nt.ModelBuilder(gravity=GRAVITY)
        builder.replicate(prototype, len(keys), spacing=(0.0, 0.0, 0.0))
        self.model = builder.finalize(device=self.device)
        self.current, self.next = self.model.state(), self.model.state()
        self.control = self.model.control(clone_variables=False)
        self.solver = nt.solvers.SolverSemiImplicit(
            self.model, angular_damping=0.0, enable_tri_contact=False
        )
        self.calls = self.syncs = self.reads = self.read_bytes = 0
        self.last_q = self.last_v = None
        self.inspect_unused = False
        self.keys = tuple(keys[int(i)] for i in self.model.body_world.numpy())
        if np.any(z != 2.0) or np.any(vz != 0.0):
            poses = self.current.body_q.numpy().copy()
            twists = self.current.body_qd.numpy().copy()
            initial = {k: (x, v) for k, x, v in zip(keys, z, vz, strict=True)}
            for i, key in enumerate(self.keys):
                poses[i, 2], twists[i, 2] = initial[key]
            self.current.body_q.assign(poses)
            self.current.body_qd.assign(twists)
        wp.synchronize_device(self.device)

    def binding(self):
        m = self.model
        n = len(self.keys)
        body_world = m.body_world.numpy()
        shape_world = m.shape_world.numpy()
        source = m.shape_body.numpy()
        assert m.world_count == m.body_count == m.shape_count == n and m.joint_count == n
        np.testing.assert_array_equal(np.sort(body_world), np.arange(n))
        np.testing.assert_array_equal(np.sort(source), np.arange(n))
        np.testing.assert_array_equal(shape_world, body_world[source])
        np.testing.assert_array_equal(m.joint_type.numpy(), np.full(n, int(self.nt.JointType.FREE)))
        np.testing.assert_array_equal(m.body_mass.numpy(), np.full(n, MASS))
        np.testing.assert_array_equal(m.body_inv_mass.numpy(), np.full(n, 1 / MASS))
        np.testing.assert_array_equal(m.body_com.numpy(), np.zeros((n, 3)))
        np.testing.assert_allclose(
            m.body_inertia.numpy(), np.tile(np.eye(3) * INERTIA, (n, 1, 1)), rtol=2e-6, atol=1e-8
        )
        np.testing.assert_array_equal(m.gravity.numpy()[body_world], np.tile(GRAVITY, (n, 1)))
        np.testing.assert_array_equal(m.shape_type.numpy(), np.full(n, int(self.nt.GeoType.SPHERE)))
        np.testing.assert_array_equal(m.shape_scale.numpy()[:, 0], np.full(n, RADIUS))
        assert np.all(m.body_flags.numpy() & int(self.nt.BodyFlags.DYNAMIC))
        shape_flags = m.shape_flags.numpy()
        assert not np.any(
            shape_flags
            & (int(self.nt.ShapeFlags.COLLIDE_SHAPES) | int(self.nt.ShapeFlags.COLLIDE_PARTICLES))
        )
        assert (
            self.current.body_q is not self.next.body_q
            and self.current.body_qd is not self.next.body_qd
        )
        assert (
            str(self.current.body_q.device) == str(self.current.body_qd.device) == str(self.device)
        )
        return dict(
            implementation="newton.solvers.SolverSemiImplicit",
            device=str(self.device),
            keys_by_body=self.keys,
            body_world=body_world.tolist(),
            shape_body=source.tolist(),
            shape_world=shape_world.tolist(),
            physical_bodies=n,
            sphere_radius_m=RADIUS,
            mass_kg=MASS,
            gravity_m_s2=m.gravity.numpy()[body_world[0]].tolist(),
            scalar_inertia_kg_m2=INERTIA,
            free_joints=n,
            locked_mass=True,
            collision_enabled=False,
            shape_flags=shape_flags.tolist(),
            position_velocity_read="body_q(world pose, 7 FP32) and body_qd(world COM twist, 6 FP32)",
            status="synchronous return plus common application verification",
        )

    def read(self, tick):
        q = self.current.body_q.numpy()
        v = self.current.body_qd.numpy()
        self.reads += 2
        if self.device.is_cuda:
            self.read_bytes += q.nbytes + v.nbytes
        self.last_q, self.last_v = q, v
        if self.inspect_unused:
            np.testing.assert_array_equal(
                q[:, 3:], np.tile((0.0, 0.0, 0.0, 1.0), (len(self.keys), 1))
            )
            np.testing.assert_array_equal(v[:, 3:], np.zeros((len(self.keys), 3)))
        return q[:, :3], v[:, :3]

    def advance(self, tick):
        self.solver.step(self.current, self.next, self.control, None, DT)
        self.current, self.next = self.next, self.current
        self.wp.synchronize_device(self.device)
        self.syncs += 1
        self.calls += 1
        return self.read(tick)

    def counters(self):
        return dict(
            solver_calls=self.calls,
            explicit_device_synchronizations=self.syncs,
            public_array_reads=self.reads,
            explicit_state_d2h_bytes=self.read_bytes,
        )

    def events(self, output, tick):
        wp = self.wp
        if not self.device.is_cuda:
            output.read(tick, *self.advance(tick))
            return dict(type="CPU qualification")
        wp.timing_begin()
        try:
            output.read(tick, *self.advance(tick))
        finally:
            rec = wp.timing_end()
        names = Counter(e.name for e in rec)
        kernel_names = [e.name for e in rec if e.filter == wp.TIMING_KERNEL]
        assert any("integrate_bodies" in str(n) for n in kernel_names), kernel_names
        return dict(
            type="Warp activity timing on one extra audited step, outside wall samples",
            names=dict(names),
            kernel_names=kernel_names,
            activity_elapsed_ms=[(e.name, float(e.elapsed)) for e in rec],
        )

    def close(self):
        # The unused rotational components and persistent input remain physical controls.
        if self.last_q is None:
            return
        n = len(self.keys)
        np.testing.assert_array_equal(self.last_q[:, 3:], np.tile((0.0, 0.0, 0.0, 1.0), (n, 1)))
        np.testing.assert_array_equal(self.last_v[:, 3:], np.zeros((n, 3)))
        np.testing.assert_array_equal(self.current.body_f.numpy(), np.zeros((n, 6)))


def load_engine(name, device, cache):
    if name == "phymeo":
        import tyx as tx

        import phymeo as ph

        assert "newton" not in sys.modules
        return (
            (ph, tx),
            PhymeoFall,
            dict(phymeo=version("phymeo"), tyx=version("tyx-compute"), source=ph.__file__),
        )
    import newton
    import warp as wp

    assert "phymeo" not in sys.modules
    if version("newton") != "1.7.0" or version("warp-lang") != "1.17.0":
        raise RuntimeError("this qualification requires Newton 1.7.0 and Warp 1.17.0")
    wp.config.kernel_cache_dir = str(cache.resolve())
    wp.init()
    d = wp.get_device(device)
    if device == "cuda:0" and not d.is_cuda:
        raise RuntimeError("CUDA device required")
    return (
        (newton, wp),
        NewtonFall,
        dict(
            newton=version("newton"),
            warp=version("warp-lang"),
            source=newton.__file__,
            device_name=d.name,
        ),
    )
