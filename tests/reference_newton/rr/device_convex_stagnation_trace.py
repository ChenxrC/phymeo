"""Nine fixed stop-policy regressions: diagnostic capture, never a repair test.

Authoring only until the parent's serial execution. Exit 2 means that capture
completed but an original geometry oracle failed; it is not a pytest pass.
"""

import argparse
import gzip
import hashlib
import importlib
import importlib.util
import json
import sys
from contextlib import closing, contextmanager
from pathlib import Path

IDENTITIES = (
    "K0-Q01-CB-sep",
    "K0-Q01-CB-overlap",
    "K0-Q01-BH-sep",
    "K0-Q03-CB-reverse",
    "K0-Q03-CB-cycle",
    "K0-Q03-CB-tilt",
    "K0-Q03-CB-permute",
    "K0-Q03-CB-cycle_reverse_ids",
    "cb_cycle_reverse_ids_segment",
)
KERNEL_SHA256 = {
    "accepted-baseline": "c86c4565aa8d2556e062f32c71d4d84b558a320abd171756b0ca3318296ccd26",
    "candidate-experiment": "4d75b7c68f86090bae4fbdd248beeccc4b9e290716124002e29def5545de5492",
}


def helpers():
    collision = Path(__file__).resolve().parents[2] / "collision"
    if str(collision) not in sys.path:
        sys.path.insert(0, str(collision))
    k0 = importlib.import_module("test_device_convex_query")
    qualification = importlib.import_module("test_device_convex_patch_qualification")
    path = Path(__file__).with_name("device_convex_query_trace.py")
    spec = importlib.util.spec_from_file_location("col_existing_trace_runner", path)
    trace_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace_runner)
    return k0, qualification, trace_runner


def declaration(identity, k0, qualification):
    if identity not in IDENTITIES:
        raise ValueError("not one of the nine predeclared regressions")
    ref = k0.reference()
    if identity.startswith("K0-Q01-"):
        family, phase = identity.removeprefix("K0-Q01-").split("-")
        return (
            ref.family_case(family, ref.GAPS[phase]),
            None,
            "tests/collision/test_device_convex_query.py::test_k0_family_gap_witness"
            f"[{identity}]",
        )
    if identity.startswith("K0-Q03-"):
        family, variant = identity.removeprefix("K0-Q03-").split("-", 1)
        return (
            ref.transformed(ref.family_case(family, -0.01), variant),
            None,
            "tests/collision/test_device_convex_query.py::test_k0_frame_and_identity"
            f"[{identity}]",
        )
    expected = qualification.reference().cb_overlap()
    case = ref.transformed(
        ref.family_case(expected["family"], expected["gap_m"]), "cycle_reverse_ids"
    )
    return (
        case,
        qualification.reference().cycle_reverse_ids(expected),
        "tests/collision/test_device_convex_patch_qualification.py::"
        f"test_affected_k1_identity[{identity}]",
    )


@contextmanager
def prepared(case, expected_patch, mode, cache_root, k0, qualification):
    # K1 must retain its actual eight-site patch allocation and reversed request.
    if expected_patch is not None:
        with qualification.prepared(
            case, mode, cache_root, site_capacity=8, requested_reverse=True
        ) as values:
            yield values
        return
    import phymeo as ph

    world = case["keys"][0][0]
    tree = k0.ordinary_forest(
        anchor=case["positions"][0], orientation=k0.case_quaternion(case)
    )
    with ph.TyxWorldBatch([(world, tree)], mode, cache_root=cache_root) as batch:
        before = batch.snapshot()
        model, bindings = k0.registrations(case, before[world].topology)
        keys = tuple(ph.CollisionKey(*key) for key in case["keys"])
        request = tuple(reversed(keys)) if case["requested_reverse"] else keys
        with batch.prepare_collision(model, bindings) as broad:
            # Omit site_capacity exactly as the original Q01/Q03 tests do.
            with closing(batch.prepare_convex_query(broad, pairs=(request,))) as plan:
                yield batch, broad, plan, before


def source_manifest(paths, repository):
    result = []
    for path in paths:
        path = path.resolve()
        content = path.read_bytes()
        result.append(
            dict(
                path=str(path),
                relative_path=path.relative_to(repository).as_posix(),
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    return result


def original_geometry(result, case, expected_patch, k0, qualification):
    """Catch only the original geometry assertion, not transport/source failures."""
    row = result.rows[0]
    try:
        if expected_patch is None:
            checks = k0.check_case_row(row, case, result.source)
        else:
            import numpy as np

            qualification._affected_patch_geometry(row, expected_patch)
            assert abs(np.linalg.norm(row.normal) - 1) <= 2e-4
            assert np.linalg.norm(np.asarray(row.normal) - expected_patch["normal"]) <= 2e-3
            checks = {"original_complete_patch_oracle": True}
    except AssertionError as error:
        return {
            "status": "FAILED",
            "row_status": row.status,
            "row_reason": row.reason,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    return {
        "status": "PASSED",
        "row_status": row.status,
        "row_reason": row.reason,
        "checks": checks,
    }


def run(identity, mode, role, output, cache_root):
    if identity not in IDENTITIES or mode not in ("interpret", "cuda"):
        raise ValueError("invalid fixed identity or backend")
    if role not in KERNEL_SHA256:
        raise ValueError("invalid comparison role")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    k0, qualification, trace_runner = helpers()
    import numpy as np
    import phymeo as ph

    repository = Path(__file__).resolve().parents[3]
    package = (repository / "src" / "phymeo").resolve()
    if Path(ph.__file__).resolve().parent != package:
        raise RuntimeError("phymeo did not load from this selected checkout")
    product = package / "backends"
    kernel = product / "_tyx_convex_query_kernels.py"
    if hashlib.sha256(kernel.read_bytes()).hexdigest() != KERNEL_SHA256[role]:
        raise RuntimeError("selected product kernel does not match comparison role")
    paths = (
        Path(__file__), Path(trace_runner.__file__), Path(k0.__file__),
        Path(qualification.__file__), Path(k0.reference().__file__),
        Path(qualification.reference().__file__), kernel,
        product / "_device_convex_query.py",
        product / "_device_convex_query_trace.py",
        product / "_tyx_convex_patch_kernels.py",
        product / "_resident_convex_source.py",
    )
    case, expected_patch, nodeid = declaration(identity, k0, qualification)
    record = dict(
        schema="COL-stop-stagnation-capture-v1",
        identity=identity,
        identity_ordinal=IDENTITIES.index(identity),
        original_nodeid=nodeid,
        mode=mode,
        comparison_role=role,
        package_origin=str(Path(ph.__file__).resolve()),
        input=case,
        expected_patch=expected_patch,
        input_binding_sha256=hashlib.sha256(
            k0.canonical({"case": case, "expected_patch": expected_patch}).encode()
        ).hexdigest(),
        source_before=source_manifest(paths, repository),
        capture_status="INCOMPLETE",
        geometry_status="NOT_CHECKED",
        normal_query=None,
        trace=None,
        failure=None,
    )
    try:
        record["independent_literal_oracle"] = k0.reference().literal_parallel_precondition(case)
        with prepared(case, expected_patch, mode, cache_root, k0, qualification) as (
            batch, broad, plan, before
        ):
            record["before"] = before
            keys = tuple(ph.CollisionKey(*key) for key in case["keys"])
            model, bindings = k0.registrations(case, before[keys[0].world].topology)
            record.update(model=model, bindings=bindings, stats_before=dict(plan.stats))
            normal = plan.query()
            record["normal_query"] = normal
            # Store geometry verdict before tracing, even if trace transport then fails.
            record["normal_geometry"] = original_geometry(
                normal, case, expected_patch, k0, qualification
            )
            trace = plan.query_with_diagnostics(pairs=(keys,))
            record["trace"] = trace
            record["trace_geometry"] = original_geometry(
                trace.query, case, expected_patch, k0, qualification
            )
            trace_runner.same_state_geometry(normal, trace.query, k0.encode)
            record["same_state_geometry_equal"] = True
            plan.require_current_diagnostics(trace)
            k0.check_raw(trace.query)
            k0.check_source(trace.query, model, bindings, before, case=case)
            requested = (
                tuple(reversed(keys))
                if expected_patch is not None or case["requested_reverse"]
                else keys
            )
            assert trace.query.requested_pairs == (requested,)
            assert trace.query.request_reversed == (requested != tuple(sorted(keys)),)
            assert len(trace.query.rows) == 1
            assert bool(trace.query.patch["enabled"]) == (expected_patch is not None)
            if expected_patch is not None:
                assert trace.query.request_reversed == (False,)
                assert tuple(trace.query.shape_keys) == tuple(sorted(keys))
                assert tuple(map(int, trace.query.source["vertex_ids"])) == (
                    tuple(case["ids"][1]) + tuple(case["ids"][0])
                )
                anchor = np.asarray(case["positions"][0])
                uploaded = np.vstack((case["points"][1] - anchor, case["local"][0]))
                assert np.allclose(
                    trace.query.source["uploaded_local_points_m"], uploaded, atol=1e-6
                )
            record["after"] = batch.snapshot()
            assert k0.canonical(record["after"]) == k0.canonical(before)
            record["stats_after"] = dict(plan.stats)
            record["broad_stats"] = broad.stats()
            statuses = (record["normal_geometry"]["status"], record["trace_geometry"]["status"])
            record["geometry_status"] = "PASSED" if statuses == ("PASSED", "PASSED") else "FAILED"
        record["closed_query_stats"] = dict(plan.stats)
        record["closed_batch_stats"] = batch.stats()
        record["source_after"] = source_manifest(paths, repository)
        assert record["source_before"] == record["source_after"], "source changed during capture"
        record["capture_status"] = "COMPLETE"
    except BaseException as error:
        record["capture_status"] = "FAILED"
        record["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        # Nonfinite scratch remains original array bytes plus existing nonfinite tags.
        # Exclusive creation prevents replacing the original failure or earlier run.
        with gzip.open(output, "xt", encoding="utf-8") as stream:
            json.dump(k0.encode(record), stream, allow_nan=False, separators=(",", ":"))
    summary = {
        key: record[key]
        for key in ("identity", "comparison_role", "capture_status", "geometry_status")
    }
    summary["output"] = str(output)
    summary["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps(summary, separators=(",", ":")))
    return 0 if record["geometry_status"] == "PASSED" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=IDENTITIES, required=True)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--role", choices=tuple(KERNEL_SHA256), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()
    sys.exit(run(args.case, args.mode, args.role, args.output, args.cache_root))
