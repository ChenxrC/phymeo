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


class PhymeoGroup:
    """One complete public model and a retained workspace for its world IDs."""

    def __init__(self, ids, device, cache, *, shuffle=True):
        import phymeo as ph

        self.ids = tuple(ids)
        worlds = permuted(self.ids) if shuffle else self.ids
        keys, shapes = {}, []
        for world in worlds:
            for local in LOCAL_ORDER if shuffle else range(len(BASE_X_M)):
                key = ph.CollisionKey(world, local, 0)
                keys[world, local] = key
                shapes.append(ph.CollisionShape(key, "sphere", (0,), RADIUS_M))
        self.model = ph.CollisionModel(shapes)
        self.keys = keys
        self.base = {
            key: ph.CollisionMotion([[BASE_X_M[local], 0, 0]]) for (_, local), key in keys.items()
        }
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

        return ph.detect(self.model, self.paths, self.policy, self.work)

    def inspect(self, out, moved, *, retain_positive_gap=False):
        assert out.status == "ok" and self.work.last_result is out
        assert len(out.pairs) == len(self.model.shapes) * (len(self.model.shapes) - 1) // 2
        extent = RADIUS_M + (RADIUS_M / 2 if retain_positive_gap else 0)
        assert len(out.bounds) == len(self.model.shapes)
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
        assert same == {(w, a, b) for w in self.ids for a, b in combinations(range(4), 2)}
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
            check_worlds(snap, self.ids, moved)
        return snap

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
