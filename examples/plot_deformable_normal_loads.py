"""Plot the two closed D4-N failed windows without running any physics.

Usage (absolute paths):
    python plot_deformable_normal_loads.py INTERPRET_WINDOW CUDA_WINDOW --output FRESH_DIR

Each input contains summary.json and normal-load.jsonl.gz. CSV rows are saved
observation samples, not recomputed mechanics. Arrays remain encoded. Matplotlib
is imported only when rendering; no Phymeo, runner, or oracle is imported.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

SAMPLE_FIELDS = (
    "step",
    "time",
    "support_force",
    "gap",
    "oriented_side",
    "rigid_normal_speed",
    "center_normal_speed",
    "outside",
    "positive",
)
WINDOWS = ((0.75, 1.0), (2.75, 3.0))
# Original N1 declaration and protocol; not new acceptance thresholds.
ORIGINAL_GATES = {
    "support": [12.8, 19.2],
    "RMS": 0.1,
    "peak": 128.0,
    "minimum_gap": -0.0625,
    "event_duration": 0.0078125,
}
LAYER_WIDTH = 0.0625
LOAD = 16.0
UNLOAD = (1.0, 1.125)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mapping(value):
    """Decode only one string-keyed Mapping layer; never array bytes."""
    require(
        isinstance(value, dict) and set(value) == {"mapping_entries"}, "expected encoded Mapping"
    )
    result = {}
    for pair in value["mapping_entries"]:
        require(isinstance(pair, list) and len(pair) == 2, "invalid Mapping entry")
        key, item = pair
        require(type(key) is str and key not in result, "invalid or duplicate Mapping key")
        result[key] = item
    return result


def decode_mappings(value):
    """Only used for the final summary, which contains no physical arrays."""
    if isinstance(value, dict):
        if set(value) == {"mapping_entries"}:
            return {key: decode_mappings(item) for key, item in mapping(value).items()}
        return {key: decode_mappings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_mappings(item) for item in value]
    return value


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_window(directory):
    directory = Path(directory)
    require(directory.is_absolute(), "input window must be absolute")
    directory = directory.resolve(strict=True)
    summary_path, trace_path = directory / "summary.json", directory / "normal-load.jsonl.gz"
    summary_raw = summary_path.read_bytes()
    summary = json.loads(summary_raw)
    require(summary["schema"] == "d4-normal-window-v1", "unknown window schema")
    require(summary["status"] == "FAIL", "this comparison must retain the original FAIL")
    require(summary["case"] == "n1_sphere", "this tool is scoped to the original N1 windows")
    require(summary["mode"] in ("interpret", "cuda"), "unknown backend")
    require(summary["qualification"] == "DIAGNOSTIC_ONLY", "unexpected qualification")
    steps, h, duration = summary["planned_steps"], summary["h"], summary["T"]
    require(type(steps) is int and steps > 0, "invalid planned steps")
    require(finite(h) and h > 0 and finite(duration), "invalid step or duration")
    require(steps * h == duration == 3.0, "incomplete or changed planned physical window")
    progress = summary["progress"]
    for key in ("steps_attempted", "steps_returned", "steps_checked", "worldsteps"):
        require(type(progress[key]) is int and progress[key] == steps, "incomplete " + key)
    require(
        progress["construct_returned"] == progress["close_returned"] == 1,
        "window is not successfully constructed and closed",
    )
    require(progress["control_submissions"] == 3, "changed control schedule")
    require(summary["observations"]["steps"] == steps, "summary observation count")
    windows = summary["observations"]["windows"]
    require(len(windows) == len(WINDOWS), "wrong support windows")
    for row, (begin, end) in zip(windows, WINDOWS, strict=True):
        require((row["begin"], row["end"]) == (begin, end), "changed support interval")
        require(row["count"] == int((end - begin) / h), "incomplete support interval")
        for key in ("mean_force", "rigid_velocity_rms", "center_velocity_rms"):
            require(finite(row[key]), "nonfinite saved window statistic")

    digest, logical_bytes, records = hashlib.sha256(), 0, 0
    samples, controls = [], []
    declared, gates, embedded = None, None, None
    close_returns, terminal_geometries = 0, 0
    last_event = None
    # Complete streaming read verifies gzip CRC and the producer's logical hash.
    # Large array hex strings are parsed as strings and discarded with each row.
    with gzip.open(trace_path, "rb") as stream:
        for raw in stream:
            require(raw.endswith(b"\n"), "unterminated JSONL record")
            digest.update(raw)
            logical_bytes += len(raw)
            record = json.loads(raw)
            require(
                type(record["ordinal"]) is int and record["ordinal"] == records,
                "noncontiguous journal ordinal",
            )
            records += 1
            event = record["event"]
            require(type(event) is str and embedded is None, "record after final summary")
            last_event = event
            if event == "declaration":
                require(declared is None and record["ordinal"] == 0, "duplicate/late declaration")
                payload = mapping(record["payload"])
                declared, gates = mapping(payload["case"]), mapping(payload["gates"])
                require(payload["mode"] == summary["mode"], "declaration backend mismatch")
                require(
                    (declared["name"], declared["h"], declared["steps"])
                    == (summary["case"], h, steps),
                    "declaration/window mismatch",
                )
                for key, expected in ORIGINAL_GATES.items():
                    require(gates[key] == expected, "changed original gate " + key)
            elif event == "control-attempt":
                payload = mapping(record["payload"])
                controls.append((payload["step"], payload["time"]))
            elif event == "observation":
                payload = mapping(record["payload"])
                sample = mapping(payload["sample"])
                require(set(sample) == set(SAMPLE_FIELDS), "changed observation schema")
                require(
                    type(sample["step"]) is int and sample["step"] == len(samples),
                    "missing/duplicate/out-of-order observation",
                )
                require(
                    sample["step"] < steps and sample["time"] == sample["step"] * h,
                    "observation time differs from the saved step",
                )
                for key in SAMPLE_FIELDS[1:7]:
                    require(finite(sample[key]), "nonfinite observation " + key)
                require(
                    type(sample["outside"]) is bool and type(sample["positive"]) is bool,
                    "invalid saved event flags",
                )
                samples.append(dict(journal_ordinal=record["ordinal"], **sample))
            elif event == "terminal-geometry":
                terminal_geometries += 1
            elif event == "close-return":
                close_returns += 1
            elif event == "summary":
                embedded = decode_mappings(mapping(record["payload"])["summary"])

    require(declared is not None and gates is not None, "missing declaration")
    require(
        len(samples) == steps and samples[-1]["time"] + h == duration,
        "truncated observation trajectory",
    )
    require(
        controls == [(0, 0.0), (int(1 / h), 1.0), (int(1.125 / h), 1.125)],
        "changed saved control boundaries",
    )
    require(close_returns == 1 and terminal_geometries == 1, "missing close/terminal evidence")
    require(last_event == "summary" and embedded is not None, "missing final summary")
    # emit("summary") measures its own save time after serializing the journal
    # record. Only this already documented timing value changes before JSON close.
    inside_save, outside_save = embedded["costs"]["save_seconds"], summary["costs"]["save_seconds"]
    require(
        finite(inside_save) and finite(outside_save) and outside_save >= inside_save >= 0,
        "invalid closing save timing",
    )
    comparable = {key: value for key, value in summary.items() if key != "trace"}
    comparable["costs"] = dict(summary["costs"], save_seconds=inside_save)
    require(embedded == comparable, "closed summary differs beyond its own save timing")
    trace = summary["trace"]
    require(
        (records, logical_bytes, digest.hexdigest(), trace_path.stat().st_size)
        == (
            trace["records"],
            trace["logical_bytes"],
            trace["logical_sha256"],
            trace["compressed_bytes"],
        ),
        "trace bytes/hash/count mismatch",
    )
    require(summary_path.read_bytes() == summary_raw, "summary changed during extraction")
    source = {
        "directory": str(directory),
        "summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
        "gzip_sha256": sha256(trace_path),
        "gzip_bytes": trace_path.stat().st_size,
        "logical_sha256": digest.hexdigest(),
        "logical_bytes": logical_bytes,
        "records": records,
        "observation_rows": len(samples),
        "first_entry_time": samples[0]["time"],
        "last_entry_time": samples[-1]["time"],
        "accepted_terminal_time": duration,
        "source_summary": summary,
        "declared_gates": gates,
        "closing_save_timing": {
            "journal_seconds": inside_save,
            "external_summary_seconds": outside_save,
            "reason": "runner adds the final summary serialization time after emitting it",
        },
    }
    return source, samples


def write_csv(path, runs):
    fields = ("mode", "case", "source_status", "journal_ordinal", *SAMPLE_FIELDS)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source, samples in runs:
            summary = source["source_summary"]
            for sample in samples:
                writer.writerow(
                    dict(
                        mode=summary["mode"],
                        case=summary["case"],
                        source_status=summary["status"],
                        **sample,
                    )
                )


def render(path, runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    try:
        for axis in axes:
            for index, (begin, end) in enumerate(WINDOWS):
                axis.axvspan(
                    begin,
                    end,
                    color="#86b89b",
                    alpha=0.16,
                    label="Original support windows" if index == 0 else None,
                )
            axis.axvspan(*UNLOAD, color="#e6b971", alpha=0.18, label="Unloading command")
            for boundary in UNLOAD:
                axis.axvline(boundary, color="#9c6c27", linewidth=0.8, linestyle=":")
            axis.grid(alpha=0.22)
            axis.set_xlim(0, 3)
        for index, (source, samples) in enumerate(runs):
            summary, color = source["source_summary"], ("#175b96", "#bf5638")[index]
            mode, style = summary["mode"], ("-", "--")[index]
            times = [sample["time"] for sample in samples]
            axes[0].plot(
                times,
                [s["support_force"] for s in samples],
                label=mode + " entry support",
                color=color,
                linestyle=style,
            )
            axes[1].plot(
                times,
                [s["gap"] for s in samples],
                label=mode + " entry gap",
                color=color,
                linestyle=style,
            )
            axes[2].plot(
                times,
                [s["rigid_normal_speed"] for s in samples],
                label=mode + " rigid",
                color=color,
                linestyle=style,
            )
            axes[2].plot(
                times,
                [s["center_normal_speed"] for s in samples],
                label=mode + " free center",
                color=color,
                linestyle=":",
                alpha=0.85,
            )
            for window in summary["observations"]["windows"]:
                axes[0].hlines(
                    window["mean_force"],
                    window["begin"],
                    window["end"],
                    colors=color,
                    linestyles=style,
                    linewidth=2.8,
                )
        axes[0].axhline(LOAD, color="#383838", linewidth=1, label="Target load 16 N")
        axes[0].axhline(
            ORIGINAL_GATES["peak"],
            color="#9c2436",
            linestyle=":",
            label="Instantaneous peak limit 128 N",
        )
        for begin, end in WINDOWS:
            axes[0].fill_between(
                [begin, end], *ORIGINAL_GATES["support"], color="#43805c", alpha=0.25
            )
        axes[0].set_ylabel("Support force (N)")
        axes[0].set_title("Saved entry force; shaded 12.8-19.2 N gate applies to WINDOW MEAN")
        axes[1].axhline(LAYER_WIDTH, color="#575757", linestyle=":", label="Layer width 1/16 m")
        axes[1].axhline(0, color="#888888", linewidth=0.8, label="Zero geometric gap")
        axes[1].axhline(
            ORIGINAL_GATES["minimum_gap"],
            color="#9c2436",
            linestyle="--",
            label="Minimum gap -1/16 m",
        )
        axes[1].set_ylabel("Gap (m)")
        axes[2].axhline(0, color="#888888", linewidth=0.7)
        for sign in (-1, 1):
            axes[2].axhline(
                sign * ORIGINAL_GATES["RMS"],
                color="#9c2436",
                linestyle="--",
                label="RMS gate 0.1 m/s (not a pointwise gate)" if sign == 1 else None,
            )
        axes[2].set_ylabel("Normal velocity (m/s)")
        axes[2].set_xlabel("Accepted-entry time (s); unload +16 N from 1 to 1.125 s")
        for axis in axes:
            axis.legend(loc="upper right", fontsize=7, ncols=2)

        lines = []
        for source, _ in runs:
            summary = source["source_summary"]
            stats = "; ".join(
                f"[{w['begin']:.2f},{w['end']:.2f}): mean={w['mean_force']:.3f} N, "
                f"RMS rigid/center={w['rigid_velocity_rms']:.3f}/{w['center_velocity_rms']:.3f}"
                for w in summary["observations"]["windows"]
            )
            lines.append(f"{summary['mode']}: {stats}")
        figure.suptitle(
            "D4-N N1: interpret FAIL / CUDA FAIL — complete trajectory, steady support FAILED",
            fontsize=14,
            color="#8f1e2b",
        )
        footer = "\n".join(lines) + (
            "\nOriginal DIAGNOSTIC_ONLY / FAIL retained. Full 3 s windows; observations are t=0..T-h. "
            "No terminal sample invented.\nPlot/CSV extraction does not requalify physics or recompute forces."
        )
        figure.text(0.025, 0.015, footer, fontsize=8, va="bottom")
        figure.tight_layout(rect=(0, 0.12, 1, 0.96))
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("windows", nargs=2, type=Path, metavar="WINDOW_DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output
    require(output.is_absolute(), "output must be a fresh absolute directory")
    output = output.resolve()
    require(all(window.is_absolute() for window in args.windows), "input windows must be absolute")
    inputs = [window.resolve(strict=True) for window in args.windows]
    require(len(set(inputs)) == 2, "two distinct input windows required")
    require(
        not any(output == path or path in output.parents for path in inputs),
        "output must not be placed inside original window evidence",
    )
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "d4-normal-load-plot-v1",
        "render_status": "READING",
        "physics_requalified": False,
        "script_sha256": sha256(Path(__file__)),
        "input_directories": [str(path) for path in inputs],
        "sources": [],
    }
    try:
        runs = []
        for path in inputs:
            source, samples = load_window(path)
            receipt["sources"].append(source)
            runs.append((source, samples))
        require(
            {source["source_summary"]["mode"] for source, _ in runs} == {"interpret", "cuda"},
            "require one interpret and one CUDA window",
        )
        runs.sort(key=lambda run: run[0]["source_summary"]["mode"] != "interpret")
        first, second = (source["source_summary"] for source, _ in runs)
        require(
            all(first[key] == second[key] for key in ("case", "h", "T", "planned_steps")),
            "overlay inputs have different physical windows",
        )
        csv_path, png_path = output / "observations.csv", output / "normal-loads.png"
        write_csv(csv_path, runs)
        receipt["render_status"] = "RENDERING"
        render(png_path, runs)
        receipt.update(
            render_status="RENDERED",
            original_statuses=[source["source_summary"]["status"] for source, _ in runs],
            csv_rows=sum(len(samples) for _, samples in runs),
            outputs={
                path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in (csv_path, png_path)
            },
        )
    except BaseException as error:
        receipt.update(
            render_status="FAILED", error={"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        json_write(output / "plot-receipt.json", receipt)
    print(
        json.dumps(
            {
                "render_status": receipt["render_status"],
                "output": str(output),
                "original_statuses": receipt["original_statuses"],
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
