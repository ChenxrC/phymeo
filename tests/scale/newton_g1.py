"""Newton 1.7 G1 reference. Import only inside the independent Newton process."""

import numpy as np
from cases import BASE_X_M, HOLDOUT_X_M, LOCAL_ORDER, RADIUS_M, check_worlds, contact, permuted


class NewtonGroup:
    def __init__(self, ids, device, cache):
        import newton
        import warp as wp

        self.ids = tuple(ids)
        self.device = device
        self.counters = dict(
            collide_calls=0,
            kinematics_calls=0,
            explicit_synchronizations=0,
            host_read_calls=0,
            host_read_bytes=0,
            pose_assign_calls=0,
            pose_assign_bytes=0,
        )
        prototype = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        cfg = prototype.ShapeConfig(gap=0.0, margin=0.0, has_particle_collision=False)
        local_of_shape = {}
        for local in LOCAL_ORDER:
            pose = wp.transform((BASE_X_M[local], 0.0, 0.0), wp.quat_identity())
            body = prototype.add_body(xform=pose, label=f"ball_{local}")
            shape = prototype.add_shape_sphere(
                body, radius=RADIUS_M, cfg=cfg, label=f"sphere_{local}"
            )
            local_of_shape[shape] = local
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        world_of_slot = permuted(self.ids)
        builder.replicate(prototype, len(ids), spacing=(0.0, 0.0, 0.0))
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            rigid_contact_max=2 * len(ids),
            max_triangle_pairs=32,
            reduce_contacts=False,
        )
        self.contacts = self.pipeline.contacts()
        cap = self.contacts.rigid_contact_max
        self.gaps = wp.empty(cap, dtype=float, device=device)
        self.points = [wp.empty(cap, dtype=wp.vec3, device=device) for _ in range(2)]
        assert str(self.contacts.device) == device
        shape_world = self.read(self.model.shape_world)
        body_world = self.read(self.model.body_world)
        shape_body = self.read(self.model.shape_body)
        starts = self.read(self.model.shape_world_start)
        self.base_q = self.read(self.state.body_q).copy()
        self.key = {}
        self.body = {}
        assert self.model.world_count == len(ids) and len(shape_world) == 4 * len(ids)
        for i, slot in enumerate(shape_world):
            assert slot >= 0 and body_world[shape_body[i]] == slot
            local = local_of_shape[i - starts[slot]]
            key = (world_of_slot[slot], local)
            self.key[i] = key
            self.body[key] = int(shape_body[i])
            np.testing.assert_array_equal(self.base_q[shape_body[i], :3], (BASE_X_M[local], 0, 0))
        assert len(set(self.body.values())) == 4 * len(ids)
        self.shape_world = shape_world.tolist()
        self.moved = None

    def read(self, array):
        host = array.numpy()
        self.counters["host_read_calls"] += 1
        self.counters["host_read_bytes"] += host.nbytes
        return host

    def select(self, moved):
        # Mutate a real State, then read back all body positions. Pose uploads are outside query timing.
        own = moved if moved in self.ids else None
        if own == self.moved:
            return
        q = self.base_q.copy()
        if own is not None:
            q[self.body[own, 1], 0] = HOLDOUT_X_M
        self.state.body_q.assign(q)
        self.counters["pose_assign_calls"] += 1
        self.counters["pose_assign_bytes"] += q.nbytes
        np.testing.assert_array_equal(self.read(self.state.body_q), q)
        self.moved = own

    def query(self):
        import newton
        import warp as wp

        c = self.contacts
        self.pipeline.collide(self.state, c)
        newton.eval_rigid_contact_kinematics(
            self.model,
            self.state,
            c,
            out_distance=self.gaps,
            out_point0_world=self.points[0],
            out_point1_world=self.points[1],
        )
        wp.synchronize_device(self.device)
        self.counters["collide_calls"] += 1
        self.counters["kinematics_calls"] += 1
        self.counters["explicit_synchronizations"] += 1
        n = int(self.read(c.rigid_contact_count)[0])
        k = int(self.read(self.pipeline.broad_phase_pair_count)[0])
        assert 0 <= n <= c.rigid_contact_max and 0 <= k <= self.pipeline.shape_pairs_max
        candidates = self.read(self.pipeline.broad_phase_shape_pairs[:k]) if k else []
        names = ("shape0", "shape1", "normal", "margin0", "margin1")
        if n:
            fields = [self.read(getattr(c, "rigid_contact_" + name)[:n]) for name in names]
            fields.extend(self.read(x[:n]) for x in (self.gaps, *self.points))
        else:
            fields = [[] for _ in range(8)]
        return candidates, fields

    def inspect(self, output, moved):
        candidates, fields = output
        candidate_keys = set()
        for a, b in candidates:
            (w, i), (v, j) = self.key[int(a)], self.key[int(b)]
            assert w == v and i != j and (w, min(i, j), max(i, j)) not in candidate_keys
            candidate_keys.add((w, min(i, j), max(i, j)))
        hits = []
        for a, b, n, m0, m1, gap, p0, p1 in zip(*fields, strict=True):
            (w, i), (v, j) = self.key[int(a)], self.key[int(b)]
            assert w == v and (w, min(i, j), max(i, j)) in candidate_keys
            # The API returns support geometry and separate effective margins. Reconstruct both sphere surfaces.
            hits.append(contact(w, i, j, gap, n, p0 + m0 * n, p1 - m1 * n))
        result = dict(
            published_pair_rows=None,
            same_world_rows=None,
            cross_world_filtered_rows=None,
            candidate_pairs=len(candidates),
            contacts=sorted(hits, key=lambda row: row["world"]),
        )
        check_worlds(result, self.ids, moved)
        return result

    def stats(self):
        return dict(self.counters)

    def binding(self):
        return {
            "columns": ["world_slot", "shape_index", "body_index", "world", "local_body"],
            "rows": [[self.shape_world[i], i, self.body[key], *key] for i, key in self.key.items()],
        }

    def storage(self):
        return dict(
            broad_candidate_capacity=int(self.pipeline.shape_pairs_max),
            rigid_contact_capacity=int(self.contacts.rigid_contact_max),
            model_explicit_pair_count=int(self.model.shape_contact_pair_count),
        )

    def close(self):
        pass
