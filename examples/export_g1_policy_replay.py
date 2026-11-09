"""Export saved actual G1 physics journals to a self-contained Three.js replay.

Authoring checkpoint: this exporter has not been imported or executed here.
It streams the source once, retains only one pending accepted record, and writes
selected visualization frames directly to a new HTML file. It never loads a
reference motion, runs a controller or advances physics.

Example (parent serial queue only):
    python examples/export_g1_policy_replay.py --input run.jsonl.gz \\
        --scene g1_primitive_actuated.xml --output new-replay.html --stride 40 --world 0

The new HTML and its .summary.json sibling are exclusively created. The source
journal stays untouched, including all fields not duplicated in this visual artifact.

Text-only validation cases for the parent serial queue:
- ordinary accepted prefix followed by an atomic refusal: no extra frame;
- post-publication failure snapshot at next step: append actual q/v, force absent;
- postprocessing failure with context.last_accepted: append that later publication;
- reset snapshot at step zero: never replace a later accepted endpoint;
- duplicate equal snapshots: one frame; contradictory step/time/q/v: report error;
- attempted or entry q/v without step_number: never promote into replay frames.
"""

import argparse
import gzip
import hashlib
import io
import json
import math
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

from phymeo.builders.floating_mjcf import load_floating_mjcf
from phymeo.dynamics.floating_robot import FloatingRobotState, floating_kinematics

DEMOS = Path(__file__).resolve().parents[1] / "docs/demos"
MAX_LINE_BYTES = 32 * 1024 * 1024
MAX_SCENE_BYTES = 8 * 1024 * 1024
BLOCK_BYTES = 1024 * 1024


class _HashedReader(io.RawIOBase):
    def __init__(self, stream):
        super().__init__()
        self.stream, self.digest, self.bytes_read = stream, hashlib.sha256(), 0

    def readable(self):
        return True

    def tell(self):
        return self.bytes_read

    def readinto(self, target):
        count = self.stream.readinto(target)
        if count:
            self.digest.update(memoryview(target)[:count])
            self.bytes_read += count
        return count


class _HTML:
    def __init__(self, stream):
        self.stream, self.digest, self.bytes_written = stream, hashlib.sha256(), 0

    def write(self, value):
        data = value.encode("utf-8")
        self.stream.write(data)
        self.digest.update(data)
        self.bytes_written += len(data)


def _json(value):
    # Preserve Python's complete round-trip float representation in the dataset.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(name + " must be the complete finite physical vector")
    return result


def _initial_vector(record, name, world):
    value = record[name]
    if isinstance(value, dict):
        value = value[str(world)] if str(world) in value else value[world]
    return value


def _bind_scene(initialization, scene_path, scene_data):
    sources = initialization.get("sources", {})
    candidates = [sources[name] for name in ("scene", scene_path.name) if name in sources]
    if not candidates:
        raise ValueError("initialization.sources has no explicit scene binding")
    digest = hashlib.sha256(scene_data).hexdigest()
    for binding in candidates:
        if binding.get("sha256") != digest or binding.get("bytes", len(scene_data)) != len(scene_data):
            raise ValueError("--scene bytes differ from initialization.sources binding")
    # Load the captured, hash-bound bytes, not a file that may change during export.
    with tempfile.TemporaryDirectory(prefix="phymeo-g1-replay-model-") as directory:
        captured = Path(directory) / scene_path.name
        captured.write_bytes(scene_data)
        asset = load_floating_mjcf(captured)
    if len(asset.joint_names) != 29 or asset.model.body_count != 30 or len(asset.collisions) != 33:
        raise ValueError("expected the complete 29-joint/30-body/33-geometry actual G1 model")
    if any(row.shape.kind not in ("sphere", "capsule") for row in asset.collisions):
        raise ValueError("this G1 viewer renders only the actual sphere/capsule registrations")
    names = initialization.get("physical_joint_names")
    if names is not None and tuple(names) != asset.joint_names:
        raise ValueError("saved physical joint order differs from the bound scene")
    bodies = [asset.root_link, *asset.link_names]
    parents = [-1, *[int(parent) + 1 for parent in asset.model.children.parents]]
    model = dict(name=asset.name, body_names=bodies, body_parents=parents,
                 joint_names=list(asset.joint_names), geometry=[
                     dict(name=row.name, body_index=row.link_index + 1, link_name=row.link_name,
                          kind=row.shape.kind, radius_m=row.shape.radius,
                          local_vertices_m=row.local_vertices) for row in asset.collisions],
                 foot_body_indices={side: bodies.index(side + "_ankle_roll_link")
                                    for side in ("left", "right")})
    return asset, model, dict(path=str(scene_path), bytes=len(scene_data), sha256=digest,
                             initialization_bindings=candidates)


def _frame(asset, model, record, source_line, physical_tick, initial=False):
    state = record["state"]
    q = _vector(state["q"], 36, "saved q")
    v = _vector(state["v"], 35, "saved v")
    pose = floating_kinematics(asset.model, FloatingRobotState(q, v))
    geometry = [(pose.position[row.link_index + 1]
                 + np.asarray(row.local_vertices) @ pose.rotation[row.link_index + 1].T).tolist()
                for row in asset.collisions]
    support = record.get("support")
    load_time = record.get("load_time_s")
    if load_time is None and isinstance(support, dict):
        load_time = support.get("sample_time_s")
    return dict(source_line=source_line, physical_tick=physical_tick,
                initial=initial, time_s=record["time_s"], load_time_s=load_time,
                publication_without_env_transition=record.get("publication_without_env_transition", False),
                publication_evidence=record.get("publication_evidence"),
                state=dict(q=q.tolist(), v=v.tolist()), support=support,
                body_positions_m=pose.position.tolist(), geometry_points_m=geometry,
                foot_origins_m={side: pose.position[index].tolist()
                                for side, index in model["foot_body_indices"].items()},
                diagnostics={name: record[name] for name in
                             ("joint_rmse_rad", "root_height_m", "root_tilt_rad",
                              "root_xy_drift_m", "reference_xy_error_m",
                              "effort_caps_observed", "reaction_balance_observed") if name in record})


def _failure_publications(record, world, source_line):
    """Read actual public snapshots only; attempted and entry q/v are excluded."""
    event = record.get("event")
    containers = []
    if event == "failure_publication_snapshot":
        containers.append(("publications", record.get("publications", {})))
    for label, context in (
        ("context", record.get("context", {})),
        ("first_failure.context", (record.get("first_failure") or {}).get("context", {})),
    ):
        if isinstance(context, dict):
            containers.extend((label + "." + name, context.get(name, {}))
                              for name in ("failure_publications", "last_accepted"))
    for path, values in containers:
        if not isinstance(values, dict):
            continue
        value = values.get(str(world), values.get(world))
        if not isinstance(value, dict):
            continue
        step = value.get("step_number")
        # A reset publication at step zero is not a missing accepted endpoint.
        # Legacy context.tick or attempted/entry vectors are never promoted.
        if type(step) is not int or step <= 0:
            continue
        time = float(value["time_s"])
        if not math.isfinite(time) or time <= 0:
            raise ValueError("failure publication needs a positive finite actual time")
        state = value.get("state")
        if state is None:
            robots = value.get("robots")
            if not isinstance(robots, dict) or len(robots) != 1:
                raise ValueError("failure publication must identify the single actual G1 robot")
            state = next(iter(robots.values()))
        q = _vector(state["q"], 36, "published failure q")
        v = _vector(state["v"], 35, "published failure v")
        FloatingRobotState(q, v)  # Validate the saved storage boundary; no step.
        yield dict(time_s=time, step_number=step, state=dict(q=state["q"], v=state["v"]),
                   source_line=source_line, source_event=event, source_path=path)


def _latest_publication(prior, current):
    if prior is None:
        return current
    if current["step_number"] == prior["step_number"]:
        if (current["time_s"] != prior["time_s"] or current["state"] != prior["state"]):
            raise ValueError("conflicting actual failure publications at the same physical step")
        return prior
    if current["step_number"] > prior["step_number"]:
        if current["time_s"] <= prior["time_s"]:
            raise ValueError("failure publication counter/time ordering is inconsistent")
        return current
    if current["time_s"] >= prior["time_s"]:
        raise ValueError("failure publication counter/time ordering is inconsistent")
    return prior

def export(args):
    source, scene_path, output = args.input.resolve(), args.scene.resolve(), args.output.resolve()
    sidecar = output.with_suffix(".summary.json")
    if len({source, scene_path, output, sidecar}) != 4:
        raise ValueError("input, scene, HTML and summary must be different paths")
    if output.exists() or sidecar.exists():
        raise FileExistsError("refusing to replace an existing replay or summary")
    if scene_path.stat().st_size > MAX_SCENE_BYTES:
        raise ValueError("scene exceeds the bounded static-model input size")
    scene_data = scene_path.read_bytes()
    template = (DEMOS / "g1-physics-replay-template.html").read_text(encoding="utf-8")
    vendor = (DEMOS / "vendor/three-r160.min.js").read_text(encoding="utf-8")
    license_text = (DEMOS / "vendor/three-r160.LICENSE").read_text(encoding="utf-8")
    template = template.replace('<script src="./vendor/three-r160.min.js"></script>', "<script>\n" + vendor + "\n</script>")
    template = template.replace("</html>", "<!-- Three.js r160 license\n" + license_text + "\n-->\n</html>")
    if template.count("__G1_REPLAY_DATA__") != 1:
        raise ValueError("viewer template must contain exactly one data marker")
    prefix, suffix = template.split("__G1_REPLAY_DATA__")
    request = initialization = summary = failure = None
    asset = model = binding = writer = html_stream = summary_stream = None
    pending = failure_publication = None
    extra_publication_frames = 0
    selected_frames = accepted_records = lines = logical_bytes = 0
    last_tick = last_emitted_tick = 0
    last_time = 0.0
    trace_error = None
    logical_digest = hashlib.sha256()
    logical_complete = True
    initial_stat = source.stat()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as original:
            hashed = _HashedReader(original)
            buffered = io.BufferedReader(hashed, buffer_size=BLOCK_BYTES)
            compressed = buffered.peek(2)[:2] == b"\x1f\x8b"
            decoded = gzip.GzipFile(fileobj=buffered, mode="rb") if compressed else buffered
            while True:
                try:
                    line = decoded.readline(MAX_LINE_BYTES + 1)
                    if not line:
                        break
                    logical_digest.update(line)
                    logical_bytes += len(line)
                    lines += 1
                    if len(line) > MAX_LINE_BYTES:
                        raise ValueError("journal line exceeds the bounded 32 MiB record limit")
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("journal records must be JSON objects")
                    event = record.get("event")
                    if event in ("failure_publication_snapshot", "first_failure", "summary"):
                        for candidate in _failure_publications(record, args.world, lines):
                            failure_publication = _latest_publication(failure_publication, candidate)
                    if event == "request":
                        request = record
                    elif event == "initialization":
                        if initialization is not None:
                            raise ValueError("multiple initialization publications are unsupported")
                        asset, model, binding = _bind_scene(record, scene_path, scene_data)
                        initialization = record
                        requested_worlds = (request or {}).get("arguments", {}).get("worlds")
                        if requested_worlds is not None and args.world >= int(requested_worlds):
                            raise ValueError("selected world is outside the recorded request")
                        if "scene" in record.get("sources", {}) and args.world != 0:
                            raise ValueError("standalone G1 journals contain only logical world zero")
                        q = _initial_vector(record, "initial_q", args.world)
                        v = _initial_vector(record, "initial_v", args.world)
                        first = _frame(asset, model, dict(state=dict(q=q, v=v), time_s=0.0), lines, 0, initial=True)
                        html_stream = output.open("xb")
                        try:
                            summary_stream = sidecar.open("x", encoding="utf-8")
                        except Exception:
                            html_stream.close()
                            raise
                        writer = _HTML(html_stream)
                        writer.write(prefix)
                        writer.write('{"schema":"g1-actual-physics-replay-v1","model":' + _json(model) + ',"frames":[')
                        writer.write(_json(first))
                        selected_frames = 1
                    elif event == "accepted" and int(record.get("world", 0)) == args.world:
                        if asset is None:
                            raise ValueError("accepted state precedes hash-bound initialization")
                        tick = record.get("tick")
                        if type(tick) is not int or tick < 0 or tick != last_tick:
                            raise ValueError("selected-world accepted physical ticks must be contiguous from zero")
                        time = float(record["time_s"])
                        if not math.isfinite(time) or time <= last_time:
                            raise ValueError("accepted physical publication time must strictly advance")
                        _vector(record["state"]["q"], 36, "accepted q")
                        _vector(record["state"]["v"], 35, "accepted v")
                        last_tick, last_time = tick + 1, time
                        accepted_records += 1
                        pending = (record, lines, last_tick)
                        if last_tick % args.stride == 0:
                            writer.write("," + _json(_frame(asset, model, record, lines, last_tick)))
                            selected_frames += 1
                            last_emitted_tick = last_tick
                    elif event == "first_failure" and failure is None:
                        failure = record
                    elif event == "summary":
                        summary = record
                except (ValueError, KeyError, TypeError, OSError, EOFError, zlib.error) as error:
                    if writer is None:
                        raise
                    trace_error = dict(type=type(error).__name__, message=str(error), source_line=lines)
                    # Preserve the valid accepted prefix, then drain/hash the same
                    # stream without parsing or buffering the remaining journal.
                    try:
                        for block in iter(lambda: decoded.read(BLOCK_BYTES), b""):
                            logical_digest.update(block)
                            logical_bytes += len(block)
                    except (OSError, EOFError, zlib.error):
                        logical_complete = False
                    break
            # A damaged gzip may prevent logical decoding; the remaining original
            # compressed bytes still receive their own complete physical hash.
            for block in iter(lambda: buffered.read(BLOCK_BYTES), b""):
                pass
            if writer is None:
                raise ValueError("journal has no supported hash-bound initialization")
            if accepted_records == 0 and failure_publication is None:
                trace_error = trace_error or dict(type="NoAcceptedStates", message="selected world has no accepted physical state")
            if pending is not None and pending[2] != last_emitted_tick:
                record, line_number, tick = pending
                writer.write("," + _json(_frame(asset, model, record, line_number, tick)))
                selected_frames += 1
            if summary is not None:
                stats = summary.get("statistics") or summary.get("statistics_by_world", {}).get(str(args.world), {})
                duration = stats.get("accepted_duration_s")
                count = stats.get("accepted_steps")
                statistics_bases = [(last_time, accepted_records)]
                if failure_publication is not None and (
                    failure_publication["step_number"] > last_tick and failure_publication["time_s"] > last_time
                ):
                    statistics_bases.append((failure_publication["time_s"], failure_publication["step_number"]))
                if not any(
                    (duration is None or math.isclose(float(duration), time, rel_tol=1e-12, abs_tol=1e-12))
                    and (count is None or int(count) == steps) for time, steps in statistics_bases
                ):
                    trace_error = trace_error or dict(type="SummaryStateMismatch", message="summary statistics differ from ordinary accepted records and actual failure publication")
                requested = summary.get("requested_duration_s", summary.get("duration_s"))
                if summary.get("success") is True and requested is not None and not math.isclose(
                    float(requested), last_time, rel_tol=1e-10, abs_tol=1e-10
                ):
                    trace_error = trace_error or dict(type="IncompleteAcceptedWindow", message="reported success lacks the full requested accepted window")
            normal_accepted_duration = last_time
            normal_accepted_tick = last_tick
            if failure_publication is not None:
                published_step = failure_publication["step_number"]
                published_time = failure_publication["time_s"]
                if published_step > last_tick and published_time > last_time:
                    evidence = {name: failure_publication[name] for name in
                                ("source_line", "source_event", "source_path", "step_number", "time_s")}
                    terminal = dict(time_s=published_time, state=failure_publication["state"],
                                    support=None, load_time_s=None,
                                    publication_without_env_transition=True,
                                    publication_evidence=evidence)
                    writer.write("," + _json(_frame(asset, model, terminal,
                                 failure_publication["source_line"], published_step)))
                    selected_frames += 1
                    extra_publication_frames = 1
                    last_time, last_tick = published_time, published_step
                elif published_step == last_tick:
                    expected_state = None if pending is None else {
                        "q": pending[0]["state"]["q"], "v": pending[0]["state"]["v"]}
                    if published_time != last_time or failure_publication["state"] != expected_state:
                        trace_error = trace_error or dict(type="ConflictingPublication", message="failure snapshot disagrees with the ordinary accepted record")
                elif published_step > last_tick or published_time >= last_time:
                    trace_error = trace_error or dict(type="PublicationOrdering", message="failure snapshot counter and accepted time disagree")
            final_stat = source.stat()
            unchanged = (initial_stat.st_size, initial_stat.st_mtime_ns) == (final_stat.st_size, final_stat.st_mtime_ns)
            complete = summary is not None and trace_error is None and unchanged and logical_complete
            metadata = dict(
                world=args.world, stride_physical_ticks=args.stride,
                selected_frames=selected_frames, accepted_records=accepted_records,
                actual_accepted_duration_s=last_time, actual_published_duration_s=last_time,
                last_accepted_physical_tick=last_tick,
                ordinary_accepted_duration_s=normal_accepted_duration,
                ordinary_accepted_physical_tick=normal_accepted_tick,
                extra_failure_publication_frames=extra_publication_frames,
                failure_publication=failure_publication,
                journal_summary_present=summary is not None, journal_complete=complete,
                reported_success=None if summary is None else summary.get("success"),
                first_failure=failure, trace_error=trace_error,
                source=dict(path=str(source), original_bytes=hashed.bytes_read,
                            original_sha256=hashed.digest.hexdigest(), gzip=compressed,
                            decoded_jsonl_bytes=logical_bytes, decoded_jsonl_sha256=logical_digest.hexdigest(),
                            decoded_hash_complete=logical_complete, unchanged_during_export=unchanged,
                            processed_lines=lines), scene=binding,
                geometry_source="static hash-bound model and saved actual q/v only",
                frame_selection="initial, every Nth ordinary accepted tick, final ordinary accepted, strictly later actual failure publication",
                visual_fields="saved actual q/v and full support plus derived visual FK; all site/command/policy details remain in the unmodified original journal",
                force_display="input-state sampled force; resultant arrows at accepted foot origins are debug anchors, not actual application points",
                reference_motion_loaded=False, physics_advanced=False)
            writer.write('],"metadata":' + _json(metadata) + ',"initialization":' + _json(initialization)
                         + ',"request":' + _json(request) + ',"summary":' + _json(summary) + "}")
            writer.write(suffix)
            html_stream.flush()
            receipt = dict(schema="g1-actual-physics-replay-export-v1", export_complete=True,
                           output=str(output), html_bytes=writer.bytes_written,
                           html_sha256=writer.digest.hexdigest(), metadata=metadata,
                           actual_summary=summary)
            json.dump(receipt, summary_stream, ensure_ascii=False, allow_nan=False, indent=2)
            summary_stream.flush()
            return receipt
    finally:
        if html_stream is not None:
            html_stream.close()
        if summary_stream is not None:
            summary_stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--world", type=int, default=0)
    args = parser.parse_args()
    if args.stride < 1 or args.world < 0:
        parser.error("stride must be positive and world must be nonnegative")
    try:
        receipt = export(args)
    except Exception as error:
        print(json.dumps(dict(export_complete=False, error=type(error).__name__, message=str(error))), file=sys.stderr)
        return 1
    metadata = receipt["metadata"]
    print(json.dumps(dict(output=receipt["output"], html_bytes=receipt["html_bytes"],
                          selected_frames=metadata["selected_frames"], actual_duration_s=metadata["actual_accepted_duration_s"],
                          journal_complete=metadata["journal_complete"], reported_success=metadata["reported_success"])))
    return 0 if metadata["journal_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())