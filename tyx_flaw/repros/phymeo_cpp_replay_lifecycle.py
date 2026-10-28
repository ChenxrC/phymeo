"""Instrument the public Phymeo replay workload for TYX-003; not formal timing."""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from windows_cpp_lifecycle import loaded_tyx_libraries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=303)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.calls < 1 or args.output.exists():
        parser.error("positive calls and a new output path are required")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    from benchmark_load_replay import make_replay

    from phymeo.experiments.rigid_cloth import provenance

    record = dict(
        scope="instrumented Phymeo replay reproduction, not formal timing; public tyx-only controls separate",
        source=provenance(),
        requested_calls=args.calls,
        successful_calls=0,
        before_count=len(loaded_tyx_libraries()),
        checkpoints=[],
        status="running",
    )
    try:
        with make_replay("cpu_cpp") as replay:
            for index in range(args.calls):
                trial = replay.evaluate(
                    {
                        "robot": {"effort": np.array([0.04])},
                        "node": {"force": np.array([[0.1, 0, 0]])},
                    }
                )
                assert trial.result.report.accepted
                np.testing.assert_allclose(trial.result.state.domains["robot"].v, [0.32], atol=2e-5)
                np.testing.assert_allclose(
                    trial.result.state.domains["node"].velocity, [[0.1, 0, 0]], atol=2e-5
                )
                record["successful_calls"] += 1
                if (index + 1) % 16 == 0:
                    row = dict(call=index + 1, loaded_count=len(loaded_tyx_libraries()))
                    record["checkpoints"].append(row)
                    print(json.dumps(row), flush=True)
        record["status"] = "all_calls_correct"
    except Exception as error:
        record.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    finally:
        record["after_close_count"] = len(loaded_tyx_libraries())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            json.dumps({k: record[k] for k in ("status", "successful_calls", "after_close_count")}),
            flush=True,
        )
    return 0 if record["status"] == "all_calls_correct" else 1


if __name__ == "__main__":
    raise SystemExit(main())
