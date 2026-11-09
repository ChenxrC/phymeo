"""Installed-wheel consumer for the four K0 first controls.

Run with python -I -B from outside the checkout. The script inserts only the
installed site and refuses a Phymeo loaded from the source tree.
"""

import argparse
import gzip
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np


CASES = (
    ("K0-Q01-SS-sep", "SS", 0.02),
    ("K0-Q01-SS-overlap", "SS", -0.01),
    ("K0-Q01-CH-sep", "CH", 0.02),
    ("K0-Q01-CH-overlap", "CH", -0.01),
)


def forest():
    import phymeo as ph

    n = 1
    positions = np.zeros((n, 3))
    model = ph.RobotModel(
        np.full(n, -1, dtype=np.int32),
        ("prismatic",) * n,
        positions,
        np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1)),
        np.tile([0.0, 0.0, 1.0], (n, 1)),
        np.ones(n),
        np.zeros((n, 3)),
        np.full((n, 3), 1 / 16),
        np.zeros(3),
    )
    program = ph.DrivenRobotProgram(
        model, kp=np.zeros(n), kd=np.zeros(n), effort_limit=np.full(n, 8.0), delay_steps=0
    )
    state = program.initial_state(ph.RobotState(np.zeros(n), np.zeros(n)))
    return ph.FixedRobotTree(program, state)


def geometry(family, gap):
    if family == "SS":
        first = np.array([[0.0, 0.0, 0.0]])
        second = np.array([[0.0, 0.0, 0.10 + gap]])
        kinds = ("sphere", "sphere")
        radii = (0.05, 0.05)
    else:
        ring = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0.0), (0.3, 0.1), (-0.5, 0.1))
        hull = np.array([(x, y, z) for x, y in ring for z in (-0.1, 0.1)])
        first = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        second = hull + np.array([0.0, 0.0, 0.15 + gap])
        kinds = ("capsule", "hull")
        radii = (0.05, 0.0)
    return first, second, kinds, radii


def one_query(mode, cache_root, identity, family, gap):
    import phymeo as ph

    first, second, kinds, radii = geometry(family, gap)
    world = 17
    keys = (ph.CollisionKey(world, 1, 0), ph.CollisionKey(world, 2, 0))
    with ph.TyxWorldBatch([(world, forest())], mode, cache_root=cache_root) as batch:
        before = batch.snapshot()
        shapes, bindings = [], []
        points = (first, second)
        for side, key in enumerate(keys):
            ids = tuple(range(100 + 100 * side, 100 + 100 * side + len(points[side])))
            shapes.append(ph.CollisionShape(key, kinds[side], ids, radius=radii[side], revision=7 + side))
            if side == 0:
                bindings.append(ph.ResidentTreeBinding(key, world, 0, points[side], topology=before[world].topology))
            else:
                bindings.append(ph.ResidentStaticBinding(key, ph.CollisionMotion(points[side])))
        model = ph.CollisionModel(tuple(shapes))
        with batch.prepare_collision(model, tuple(bindings)) as broad:
            with batch.prepare_convex_query(broad, pairs=(keys,)) as plan:
                result = batch.convex_query(plan)
                row = result.rows[0]
                return {
                    "identity": identity,
                    "status": row.status,
                    "relation": row.relation,
                    "reason": row.reason,
                    "gap_m": row.gap_m,
                    "declared_gap_m": gap,
                    "normal": list(row.normal),
                    "queries": 1,
                    "live_bytes_after_close": None,
                    "pair_source": result.pair_source,
                }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel-site", type=Path, required=True)
    parser.add_argument("--expected-package-root", type=Path, required=True)
    parser.add_argument("--expected-wheel", type=Path, required=True)
    parser.add_argument("--forbidden-checkout", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    site = args.wheel_site.resolve()
    package = args.expected_package_root.resolve()
    checkout = args.forbidden_checkout.resolve()
    wheel = args.expected_wheel.resolve()
    sys.path.insert(0, str(site))
    import phymeo as ph

    loaded = Path(ph.__file__).resolve()
    if package not in loaded.parents or checkout in loaded.parents:
        raise SystemExit(f"phymeo loaded from {loaded}, expected under {package}")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    rows = []
    accepted = 0
    rejected = 0
    try:
        for identity, family, gap in CASES:
            row = one_query(args.mode, args.cache_root, identity, family, gap)
            error = abs(row["gap_m"] - gap)
            row["absolute_gap_error_m"] = error
            row["accepted"] = row["status"] == "RESOLVED" and error <= 1e-4
            accepted += int(row["accepted"])
            rejected += int(not row["accepted"])
            rows.append(row)
    except Exception:
        rejected += 1
        rows.append({"error": traceback.format_exc()})
    report = {
        "schema": "k0-wheel-consumer-v1",
        "mode": args.mode,
        "phymeo_file": str(loaded),
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "accepted_queries": accepted,
        "rejected_queries": rejected,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as stream:
        json.dump(report, stream)
    print(args.output)
    print(args.mode, accepted, "accepted", rejected, "rejected")
    if rejected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
