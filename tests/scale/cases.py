"""G1 co-located four-sphere worlds, analytic witnesses and the existing Phymeo query."""

from itertools import combinations
from math import dist, isclose

RADIUS_M = 1 / 8
BASE_X_M = (0.0, 3 / 16, 2.0, 4.0)
HOLDOUT_X_M = 5 / 16
SCALES = (1, 2, 8, 32, 64)
LOCAL_ORDER = (2, 0, 3, 1)
LENGTH_TOL_M = 1e-6
NORMAL_TOL = 1e-5


def world_ids(count):
    if count not in SCALES:
        raise ValueError(f"select one of the frozen world counts {SCALES}")
    return tuple(17 + 19 * i for i in range(count))


def permuted(ids):
    return (*ids[1::2][::-1], *ids[::2])


def contact(world, a, b, gap, normal, pa, pb):
    """Canonical local-body order, retaining the two independent boundary points."""
    if a > b:
        a, b, pa, pb, normal = b, a, pb, pa, [-v for v in normal]
    return dict(
        world=int(world),
        local_pair=[int(a), int(b)],
        gap_m=float(gap),
        normal=[float(x) for x in normal],
        a_point_m=[float(x) for x in pa],
        b_point_m=[float(x) for x in pb],
    )


def check_witness(row, moved=False):
    x = HOLDOUT_X_M if moved else BASE_X_M[1]
    assert row["local_pair"] == [0, 1], row
    assert isclose(row["gap_m"], x - 2 * RADIUS_M, rel_tol=0, abs_tol=LENGTH_TOL_M), row
    assert dist(row["normal"], [1, 0, 0]) <= NORMAL_TOL, row
    assert dist(row["a_point_m"], [RADIUS_M, 0, 0]) <= LENGTH_TOL_M, row
    assert dist(row["b_point_m"], [x - RADIUS_M, 0, 0]) <= LENGTH_TOL_M, row


def check_worlds(snapshot, ids, moved):
    rows = snapshot["contacts"]
    assert sorted(row["world"] for row in rows) == sorted(set(ids) - {moved}), rows
    for row in rows:
        check_witness(row)
    assert snapshot["candidate_pairs"] == len(rows), snapshot["candidate_pairs"]


def g1_model(ids, shuffle=True):
    import phymeo as ph

    worlds = permuted(ids) if shuffle else ids
    keys, shapes = {}, []
    for world in worlds:
        for local in LOCAL_ORDER if shuffle else range(len(BASE_X_M)):
            key = ph.CollisionKey(world, local, 0)
            keys[world, local] = key
            shapes.append(ph.CollisionShape(key, "sphere", (0,), RADIUS_M))
    model = ph.CollisionModel(shapes)
    base = {key: ph.CollisionMotion([[BASE_X_M[local], 0, 0]]) for (_, local), key in keys.items()}
    return model, keys, base


def inspect_phymeo(model, ids, out, moved, retain_positive_gap=False):
    assert out.status == "ok"
    assert len(out.pairs) == len(model.shapes) * (len(model.shapes) - 1) // 2
    extent = RADIUS_M + (RADIUS_M / 2 if retain_positive_gap else 0)
    assert len(out.bounds) == len(model.shapes)
    for bound in out.bounds:
        w, local = bound.key.world, bound.key.body
        x = HOLDOUT_X_M if (w, local) == (moved, 1) else BASE_X_M[local]
        assert bound.lower == (x - extent, -extent, -extent) and bound.upper == (
            x + extent,
            extent,
            extent,
        )
    same, active, separated = set(), [], []
    cross = 0
    for pair in out.pairs:
        a, b = pair.first, pair.second
        assert a.shape == b.shape == 0 and pair.family == "RR"
        if a.world != b.world:
            assert (
                not pair.candidate
                and pair.status == "filtered"
                and pair.reason == "different_world"
            )
            assert pair.witness is None
            cross += 1
            continue
        slot = (a.world, a.body, b.body)
        assert slot not in same
        same.add(slot)
        if not pair.candidate:
            assert pair.status == "separated" and pair.reason == "aabb" and pair.witness is None
            continue
        w = pair.witness
        assert w is not None and w.first_weights == w.second_weights == ((0, 1.0),)
        row = contact(a.world, a.body, b.body, w.gap_m, w.normal, w.first_point, w.second_point)
        if pair.status == "overlap":
            active.append(row)
        else:
            assert retain_positive_gap and a.world == moved and pair.status == "separated"
            check_witness(row, moved=True)
            separated.append(row)
    assert same == {(w, a, b) for w in ids for a, b in combinations(range(4), 2)}
    assert out.required_candidates == len(active) + len(separated)
    snap = dict(
        published_pair_rows=len(out.pairs),
        same_world_rows=len(same),
        cross_world_filtered_rows=cross,
        candidate_pairs=out.required_candidates,
        contacts=sorted(active, key=lambda row: row["world"]),
        positive_gap_rows=separated,
    )
    if not retain_positive_gap:
        check_worlds(snap, ids, moved)
    return snap


def combine_observations(parts):
    counts = (
        "published_pair_rows",
        "same_world_rows",
        "cross_world_filtered_rows",
        "candidate_pairs",
    )
    snapshot = {k: None if parts[0][k] is None else sum(p[k] for p in parts) for k in counts}
    snapshot["contacts"] = sorted(
        (row for p in parts for row in p["contacts"]), key=lambda row: row["world"]
    )
    return snapshot


class PhymeoGroup:
    """One complete public model and a retained workspace for its world IDs."""

    def __init__(self, ids, device, cache, *, shuffle=True):
        import phymeo as ph

        self.ids = tuple(ids)
        self.model, self.keys, self.base = g1_model(self.ids, shuffle)
        self.paths = self.base
        self.policy = ph.CollisionPolicy()
        self.work = ph.CollisionWorkspace(
            self.model, device, cache_root=cache, broad_phase="all_pairs"
        )
        if device == "cuda":
            assert self.work._engine._runtime.device == 0

    def select(self, moved):
        import phymeo as ph

        self.paths = self.base
        if moved in self.ids:
            key = self.keys[moved, 1]
            self.paths = {**self.base, key: ph.CollisionMotion([[HOLDOUT_X_M, 0, 0]])}

    def query(self):
        import phymeo as ph

        result = ph.detect(self.model, self.paths, self.policy, self.work)
        assert result.status == "ok"
        return result

    def inspect(self, out, moved, *, retain_positive_gap=False):
        assert out.status == "ok" and self.work.last_result is out
        return inspect_phymeo(self.model, self.ids, out, moved, retain_positive_gap)

    def near_control(self, moved):
        import phymeo as ph

        # Zero production margin culls the held-out pair. Keep it once for a signed-distance readout.
        out = ph.detect(self.model, self.paths, ph.CollisionPolicy(margin=RADIUS_M), self.work)
        rows = self.inspect(out, moved, retain_positive_gap=True)["positive_gap_rows"]
        assert len(rows) == int(moved in self.ids)
        return rows

    def stats(self):
        return self.work.stats()

    def binding(self):
        return {
            "columns": ["world", "body", "shape", "material_id"],
            "rows": [
                [s.key.world, s.key.body, s.key.shape, s.vertex_ids[0]] for s in self.model.shapes
            ],
        }

    def storage(self):
        # These are the actual existing host pair-table dimensions, not total process/device memory.
        pairs = self.work._pairs
        return dict(host_pair_table_slots=len(pairs), host_pair_table_bytes=pairs.nbytes)

    def close(self):
        self.work.close()


class PartitionedGroup:
    """Same six public rows per world as separate queries, from one shared batch."""

    def __init__(self, ids, device, cache):
        import phymeo as ph

        self.ids = tuple(ids)
        fixtures = {w: g1_model((w,)) for w in permuted(self.ids)}
        self.models = {w: item[0] for w, item in fixtures.items()}
        self.base = {w: item[2] for w, item in fixtures.items()}
        self.paths = self.base
        self.policies = dict.fromkeys(self.ids, ph.CollisionPolicy())
        self.work = ph.WorldCollisionWorkspace(self.models.items(), device, cache_root=cache)

    def select(self, moved):
        import phymeo as ph

        self.paths = self.base
        if moved is not None:
            key = ph.CollisionKey(moved, 1, 0)
            self.paths = {
                **self.base,
                moved: {**self.base[moved], key: ph.CollisionMotion([[HOLDOUT_X_M, 0, 0]])},
            }

    def query(self):
        out = self.work.detect(self.paths, self.policies)
        assert len(out) == len(self.ids) and all(row.status == "ok" for row in out.values())
        return out

    def inspect(self, out, moved):
        parts = []
        for w, record in out.items():
            assert self.work.last_result(w) is record
            parts.append(inspect_phymeo(self.models[w], (w,), record, moved))
        return combine_observations(parts)

    def near_control(self, moved):
        import phymeo as ph

        policies = {**self.policies, moved: ph.CollisionPolicy(margin=RADIUS_M)}
        out = self.work.detect(self.paths, policies)
        snap = inspect_phymeo(self.models[moved], (moved,), out[moved], moved, True)
        assert len(snap["positive_gap_rows"]) == 1
        return snap["positive_gap_rows"]

    def stats(self):
        return self.work.stats()

    def storage(self):
        stats = self.stats()
        return {
            k: stats[k]
            for k in (
                "host_pair_table_slots",
                "host_pair_table_bytes",
                "device_pair_table_slots",
                "device_pair_table_bytes",
                "device_flag_slots",
                "device_flag_bytes",
            )
        }

    def binding(self):
        return dict(
            columns=["world", "body", "shape", "material_id"],
            rows=[
                [s.key.world, s.key.body, s.key.shape, s.vertex_ids[0]]
                for model in self.models.values()
                for s in model.shapes
            ],
        )

    def close(self):
        self.work.close()
