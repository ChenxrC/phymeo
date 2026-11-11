"""Six predeclared COL1 failure inputs; collection-free, bounded diagnostic runner.

Uses existing independent fixture declarations, never product geometry as oracle.
The runner records geometry failure measures; it does not turn them into passes.
"""
import argparse
from contextlib import closing
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys

IDENTITIES = (
    "Y-ordinary", "Y-affine", "Y-floating", "Y-floating_affine",
    "sep-HN", "rotated-reversed-HN",
)


def helpers():
    collision = Path(__file__).resolve().parents[2] / "collision"
    if str(collision) not in sys.path:
        sys.path.insert(0, str(collision))
    return (
        importlib.import_module("test_device_convex_query"),
        importlib.import_module("test_device_convex_support_family"),
        importlib.import_module("test_device_convex_family_sources"),
    )


def same_state_geometry(normal, traced, encode):
    """Exact stored equality. Counters are retained, but are not geometry fields."""
    from dataclasses import fields
    import numpy as np

    assert normal.owner is traced.owner and normal.registration is traced.registration
    for item in fields(normal):
        if item.name in ("stats", "attempt_tag", "raw", "owner", "registration"):
            continue
        assert encode(getattr(normal, item.name)) == encode(getattr(traced, item.name)), item.name
    assert tuple(normal.raw) == tuple(traced.raw)
    for name in normal.raw:
        first, second = np.array(normal.raw[name]), np.array(traced.raw[name])
        if name == "meta":
            # Only the documented per-attempt identity differs.
            first[:, 12] = 0
            second[:, 12] = 0
        assert first.dtype == second.dtype and first.shape == second.shape
        assert first.tobytes(order="C") == second.tobytes(order="C"), "raw." + name


def run(identity, mode, output, cache_root):
    if identity not in IDENTITIES or mode not in ("interpret", "cuda"):
        raise ValueError("not one of the six predeclared diagnostic identities")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    k0, family, sources = helpers()
    import numpy as np
    import phymeo as ph

    product = Path(__file__).resolve().parents[3] / "src/phymeo/backends"
    source_paths = (Path(__file__), Path(k0.__file__), Path(family.__file__),
                    Path(sources.__file__), Path(family.reference().__file__),
                    product / "_device_convex_query.py",
                    product / "_device_convex_query_trace.py",
                    product / "_tyx_convex_query_kernels.py")
    record = dict(schema="COL1-GJK-failure-diagnostic-v1", identity=identity, mode=mode,
                  source=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                          for p in source_paths], trace=None, completed=False, failure=None)
    try:
        if identity.startswith("Y-"):
            layout = identity[2:]
            trees = tuple(sources.declaration(layout, i) for i in range(4))
            with ph.TyxWorldBatch(
                list(zip(sources.WORLDS, trees, strict=True))[::-1],
                mode, cache_root=cache_root,
            ) as batch:
                shapes, bindings, declarations, pairs = [], [], {}, []
                for index, (world, tree) in enumerate(zip(sources.WORLDS, trees, strict=True)):
                    shape, binding, declared = sources.install(batch, tree, layout, "Y", world, index)
                    shapes.extend(shape)
                    bindings.extend(binding)
                    declarations[world] = declared
                    pairs.append(tuple(item.key for item in shape))
                record.update(initial=batch.snapshot(), declarations=declarations,
                              selected_world=17, physical_step_h=sources.H)
                # This is exactly the original first real continuing step. No reset.
                with batch.prepare_collision(ph.CollisionModel(tuple(shapes)), tuple(bindings)) as broad:
                    with closing(batch.prepare_convex_query(broad, pairs=tuple(pairs))) as plan:
                        stepped = batch.step(sources.H)
                        record.update(physical_step_result=stepped, accepted=batch.snapshot())
                        if not all(stepped.values()):
                            raise RuntimeError("original accepted physical step failed")
                        selected_pair = next(pair for pair in pairs if pair[0].world == 17)
                        normal = plan.query()
                        record["normal_query"] = normal
                        trace = plan.query_with_diagnostics(pairs=(selected_pair,))
                        record["trace"] = trace
                        oracles = {}
                        for row in trace.query.rows:
                            world = row.pair[0].world
                            data = sources.current_case(record["accepted"][world], layout,
                                                        declarations[world])
                            truth = family.reference().oracle(data)
                            oracles[world] = dict(
                                input=data, truth=truth, status=row.status,
                                normal_error=float(np.linalg.norm(np.asarray(row.normal)-truth["normal"])),
                                normal_gate=0.002,
                                gap_error=abs(row.gap_m-truth["gap_m"]),
                                gap_gate=1e-4*truth["scale_m"],
                            )
                        record["independent_oracles"] = oracles
                        same_state_geometry(normal, trace.query, k0.encode)
                        record["same_state_geometry_equal"] = True
                        plan.require_current_diagnostics(trace)
        else:
            data = family.reference().case("HN", "sep", cycle=identity=="rotated-reversed-HN")
            record["input"] = data
            with family.prepared(data, mode, cache_root) as (batch, _, plan):
                record["accepted"] = batch.snapshot()
                pair = tuple(ph.CollisionKey(*key) for key in data["keys"])
                normal = plan.query()
                record["normal_query"] = normal
                trace = plan.query_with_diagnostics(pairs=(pair,))
                record["trace"] = trace
                record["independent_oracle"] = family.reference().oracle(data)
                same_state_geometry(normal, trace.query, k0.encode)
                record["same_state_geometry_equal"] = True
                plan.require_current_diagnostics(trace)
        record["completed"] = True
    except BaseException as error:
        record["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        with gzip.open(output, "wt", encoding="utf-8") as stream:
            json.dump(k0.encode(record), stream, allow_nan=False, separators=(",", ":"))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=IDENTITIES, required=True)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()
    run(args.case, args.mode, args.output, args.cache_root)
