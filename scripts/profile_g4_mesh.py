"""Locate work inside one already-verified warm G4 W1 mesh query, outside cost samples."""

import argparse
import cProfile
import json
import pstats
import sys
from pathlib import Path
from time import perf_counter_ns

import phymeo as ph

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "scale"))
from g4_geometry_cases import WORLD_IDS, copies, digest, frozen  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.expected.read_text(encoding="utf8"))
    records = baseline["workloads"]["mesh112"]
    expected = next(r for r in records if r["worlds"] == 1)["frames"]["0"]["canonical_sha256"]
    source, frames = frozen("mesh112")
    world = WORLD_IDS[0]
    models, paths = copies(source, frames, (world,))
    rules = {world: ph.CollisionPolicy()}
    with ph.WorldCollisionWorkspace(
        models, baseline["backend"], cache_root=args.cache_root
    ) as workspace:
        first = workspace.detect(paths[0], rules)
        assert digest(first[world], world) == expected
        profile = cProfile.Profile()
        start = perf_counter_ns()
        result = profile.runcall(workspace.detect, paths[0], rules)
        wall = perf_counter_ns() - start
        assert result == first and digest(result[world], world) == expected
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, function), (
        primitive,
        total,
        self_s,
        cumulative_s,
        _,
    ) in stats.stats.items():
        path = filename.replace("\\", "/")
        if "/src/phymeo/" in path:
            path = path.split("/src/phymeo/", 1)[1]
            rows.append(
                dict(
                    file=path,
                    line=line,
                    function=function,
                    primitive_calls=primitive,
                    calls=total,
                    self_s=self_s,
                    cumulative_s=cumulative_s,
                )
            )
    report = dict(
        schema="phymeo-g4-geometry-profile/1",
        case="mesh112",
        worlds=1,
        frame=0,
        backend=baseline["backend"],
        canonical_sha256=expected,
        profiled_wall_ns=wall,
        process_profile_s=stats.total_tt,
        cumulative_package_functions=sorted(rows, key=lambda row: -row["cumulative_s"])[:25],
        self_package_functions=sorted(rows, key=lambda row: -row["self_s"])[:20],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"mesh112 W1 frame0 checksum {expected}; profiled wall {wall * 1e-9:.3f} s")


if __name__ == "__main__":
    main()
