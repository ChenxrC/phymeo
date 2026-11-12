"""Compare two already FAILED N1 windows; no physics/reference is executed.

Usage: python compare_deformable_damping.py OLD_PLOT_DIR D6B_WINDOW_DIR --output FRESH_DIR
OLD_PLOT_DIR is closed run95: verify its receipt outputs, reuse only interpret
CSV samples, and never reopen the original D4 gzip. New D6-B is streamed once.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import plot_deformable_normal_loads as plot

require = plot.require
CSV_FIELDS = ("mode", "case", "source_status", "journal_ordinal", *plot.SAMPLE_FIELDS)


def reuse_original(directory):
    path = directory / "plot-receipt.json"
    raw = path.read_bytes()
    receipt = json.loads(raw)
    require(receipt["schema"] == "d4-normal-load-plot-v1" and receipt["render_status"] == "RENDERED", "original plot not closed")
    require(receipt["physics_requalified"] is False and receipt["original_statuses"] == ["FAIL", "FAIL"], "original results changed")
    require(set(receipt["outputs"]) == {"observations.csv", "normal-loads.png"}, "unknown original outputs")
    for name, expected in receipt["outputs"].items():
        output = directory / name
        require(output.stat().st_size == expected["bytes"] and plot.sha256(output) == expected["sha256"], "original output bytes/hash mismatch: " + name)
    sources = {source["source_summary"]["mode"]: source for source in receipt["sources"]}
    require(len(receipt["sources"]) == 2 and set(sources) == {"interpret", "cuda"}, "original source membership")
    for mode, source in sources.items():
        summary = source["source_summary"]
        require((summary["schema"], summary["case"], summary["status"], summary["qualification"]) == ("d4-normal-window-v1", "n1_sphere", "FAIL", "DIAGNOSTIC_ONLY"), "original qualification changed")
        require((summary["planned_steps"], summary["h"], summary["T"]) == (3072, 1 / 1024, 3), "original time grid changed")
        require(source["observation_rows"] == 3072 and source["accepted_terminal_time"] == 3, "original incomplete observations")
    csv_path = directory / "observations.csv"
    csv_raw = csv_path.read_bytes()
    require(hashlib.sha256(csv_raw).hexdigest() == receipt["outputs"]["observations.csv"]["sha256"], "CSV changed during read")
    reader = csv.DictReader(io.StringIO(csv_raw.decode("utf-8"), newline=""))
    require(tuple(reader.fieldnames) == CSV_FIELDS, "original CSV schema changed")
    counts, ordinals, selected = dict(interpret=0, cuda=0), dict(interpret=-1, cuda=-1), []
    for record in reader:
        mode = record["mode"]
        require(set(record) == set(CSV_FIELDS) and mode in counts, "extra CSV fields/backend")
        require(record["case"] == "n1_sphere" and record["source_status"] == "FAIL", "CSV source identity")
        sample = {key: float(record[key]) for key in plot.SAMPLE_FIELDS[1:7]}
        sample.update(step=int(record["step"]), journal_ordinal=int(record["journal_ordinal"]))
        for key in ("outside", "positive"):
            require(record[key] in ("True", "False"), "invalid CSV boolean")
            sample[key] = record[key] == "True"
        require(sample["step"] == counts[mode] < 3072 and sample["time"] == counts[mode] / 1024, "CSV lost ordered samples")
        require(all(plot.finite(sample[key]) for key in plot.SAMPLE_FIELDS[1:7]), "CSV nonfinite sample")
        require(ordinals[mode] < sample["journal_ordinal"] < sources[mode]["records"], "CSV original ordinal lost")
        ordinals[mode], counts[mode] = sample["journal_ordinal"], counts[mode] + 1
        if mode == "interpret":
            selected.append(sample)
    require(counts == {"interpret": 3072, "cuda": 3072} and receipt["csv_rows"] == 6144, "original CSV incomplete")
    require(path.read_bytes() == raw, "original receipt changed during reuse")
    provenance = dict(receipt_path=str(path), receipt_sha256=hashlib.sha256(raw).hexdigest(), receipt_bytes=len(raw), csv_path=str(csv_path), verified_outputs=receipt["outputs"], original_gzip_reopened=False)
    return sources["interpret"], selected, provenance


def write_csv(path, runs):
    fields = ("source", "runner_sha256", "cases_sha256", "material_dissipation", "mode", "case", "source_status", "qualification", "journal_ordinal", *plot.SAMPLE_FIELDS)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for identity, coefficient, source, samples in runs:
            summary = source["source_summary"]
            for sample in samples:
                writer.writerow(dict(source=identity, runner_sha256=summary["runner_sha256"], cases_sha256=summary["cases_sha256"], material_dissipation=coefficient, mode=summary["mode"], case=summary["case"], source_status=summary["status"], qualification=summary["qualification"], **sample))


def render(path, runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    try:
        for axis in axes:
            for begin, end in plot.WINDOWS:
                axis.axvspan(begin, end, color="#86b89b", alpha=0.16)
            axis.axvspan(*plot.UNLOAD, color="#e6b971", alpha=0.18)
            axis.set_xlim(0, 3)
            axis.grid(alpha=0.22)
        footer = []
        for index, (identity, coefficient, source, samples) in enumerate(runs):
            summary = source["source_summary"]
            label, color, style = identity + " / " + coefficient, ("#175b96", "#bf5638")[index], ("-", "--")[index]
            times = [row["time"] for row in samples]
            axes[0].plot(times, [row["support_force"] for row in samples], color=color, linestyle=style, label=label)
            axes[1].plot(times, [row["gap"] for row in samples], color=color, linestyle=style, label=label)
            axes[2].plot(times, [row["rigid_normal_speed"] for row in samples], color=color, linestyle=style, label=label + " rigid")
            axes[2].plot(times, [row["center_normal_speed"] for row in samples], color=color, linestyle=":", label=label + " center")
            stats = []
            for window in summary["observations"]["windows"]:
                axes[0].hlines(window["mean_force"], window["begin"], window["end"], colors=color, linestyles=style, linewidth=2.6)
                stats.append(f"[{window['begin']:.2f},{window['end']:.2f}): mean {window['mean_force']:.3f} N, RMS rigid/center {window['rigid_velocity_rms']:.3f}/{window['center_velocity_rms']:.3f}")
            footer.append(f"{identity}: FAIL / DIAGNOSTIC_ONLY; {coefficient}; runner {summary['runner_sha256'][:8]}\n" + "; ".join(stats))
        axes[0].axhline(plot.LOAD, color="#444444", linewidth=0.8, label="Target 16 N")
        axes[0].axhline(plot.ORIGINAL_GATES["peak"], color="#9c2436", linestyle=":", label="Peak limit 128 N")
        for begin, end in plot.WINDOWS:
            axes[0].fill_between([begin, end], *plot.ORIGINAL_GATES["support"], color="#43805c", alpha=0.25)
        axes[0].set(title="Original window MEAN gate 12.8-19.2 N (green); unload command (amber)", ylabel="Support force (N)")
        axes[1].axhline(plot.LAYER_WIDTH, color="#666666", linestyle=":", label="Layer width 1/16 m")
        axes[1].axhline(0, color="#888888", linewidth=0.6)
        axes[1].axhline(plot.ORIGINAL_GATES["minimum_gap"], color="#9c2436", linestyle="--", label="Minimum gap -1/16 m")
        axes[1].set_ylabel("Gap (m)")
        axes[2].axhline(0, color="#888888", linewidth=0.6)
        axes[2].set(title="Original window RMS <= 0.1 m/s per body: NOT a pointwise speed gate", ylabel="Normal speed (m/s)", xlabel="Saved entry time (s); complete 3072 samples, no invented terminal sample")
        for axis in axes:
            axis.legend(loc="upper right", fontsize=7, ncols=2)
        figure.suptitle("D4-N N1 interpret FAIL vs D6-B interpret FAIL — DIAGNOSTIC_ONLY", color="#8f1e2b", fontsize=14)
        figure.text(0.025, 0.015, "\n".join(footer) + "\nDifferent source versions and material models; extraction is not new qualification or a coefficient search.", fontsize=8, va="bottom")
        figure.tight_layout(rect=(0, 0.16, 1, 0.96))
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_plot", type=Path)
    parser.add_argument("damped_window", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(all(path.is_absolute() for path in (args.original_plot, args.damped_window, args.output)), "all paths must be absolute")
    output = args.output.resolve()
    inputs = [args.original_plot.resolve(), args.damped_window.resolve()]
    require(len(set(inputs)) == 2 and not any(output == p or p in output.parents for p in inputs), "fresh output outside both inputs required")
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema="d6b-damping-comparison-v1", render_status="READING", physics_requalified=False, script_sha256=plot.sha256(Path(__file__)), reader_sha256=plot.sha256(Path(plot.__file__)), inputs=list(map(str, inputs)), sources=[], outputs={})
    try:
        original, old_samples, reuse = reuse_original(inputs[0])
        receipt["verified_original_reuse"] = reuse
        receipt["sources"].append(original)
        damped, new_samples = plot.load_window(inputs[1], domain="d6b-c4096-n1-interpret")
        receipt["sources"].append(damped)
        require(damped["records"] == 30742 and damped["logical_sha256"] == "d117f5ad6932c3fbd0812d596a0f0b4c6723c6e2b8c79f98950a03b578dbe0a8", "not fixed run143")
        first, second = original["source_summary"], damped["source_summary"]
        require(all(first[key] == second[key] for key in ("mode", "h", "T", "planned_steps", "status", "qualification")), "comparison windows/statuses differ")
        runs = (("D4-N run50", "axial damping OFF", original, old_samples), ("D6-B run143", "4096 N s/m per edge", damped, new_samples))
        csv_path, png_path = output / "observations.csv", output / "damping-comparison.png"
        write_csv(csv_path, runs)
        receipt["render_status"] = "RENDERING"
        render(png_path, runs)
        receipt.update(render_status="RENDERED", csv_rows=len(old_samples) + len(new_samples), original_statuses=[first["status"], second["status"]], qualifications=[first["qualification"], second["qualification"]])
    except BaseException as error:
        receipt.update(render_status="FAILED", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        receipt["outputs"] = {p.name: dict(bytes=p.stat().st_size, sha256=plot.sha256(p)) for p in (output / "observations.csv", output / "damping-comparison.png") if p.is_file()}
        plot.json_write(output / "comparison-receipt.json", receipt)
    print(json.dumps(dict(render_status=receipt["render_status"], output=str(output), csv_rows=receipt["csv_rows"]), allow_nan=False))


if __name__ == "__main__":
    main()
