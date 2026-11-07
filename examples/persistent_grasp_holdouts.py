"""Run the four frozen grasp holdouts with explicitly declared tangent stations.

The original acquisition example owns every command, physical step, transition
and eight-gate verdict. This consumer configures the opt-in once and archives its
passive callbacks. A completed negative holdout is distinct from an incomplete
physical or observation run. No input tuning, prefix replay or alternate solver.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from time import perf_counter

import numpy as np

import phymeo as ph
from phymeo.solvers.tangent_history import StationKey

CASES = ("mimic-box", "mimic-zero-friction", "mimic-overload", "panda-sphere")
KT = 200.0
MAX_COMPRESSED_BYTES = 256 * 1024 * 1024
RESERVE_BYTES = 64 * 1024 * 1024
FOOTER_BYTES = 64 * 1024
PROTOCOL_SCHEMA = "grasp-protocol-observation-v1"
PUBLIC_FIELDS = {"state", "reading", "geometry", "history", "receipt", "events", "last_frame"}
TOKENS = ((101, 107, 131, 149), (211, 239, 263, 307))


def _example(filename):
    path = Path(__file__).resolve().with_name(filename + ".py")
    name = "persistent_grasp_original_" + filename
    if name not in sys.modules:
        spec = spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load original declaration: {path}")
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    if Path(module.__file__).resolve() != path:
        raise RuntimeError("original example module came from another path")
    return module


def _public(value):
    return _example("general_scene_friction").public_payload(value)


def _member(kind, value):
    row = dict(schema="persistent-grasp-stream-v1", kind=kind, value=_public(value))
    raw = (
        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    return gzip.compress(raw, mtime=0)


class ArchiveLimit(OSError):
    """A complete record did not fit the declared archive budget."""


class ObservationArchive:
    """Concatenated complete gzip members, with bounded failure-record reserve.

    The ordinary stream has 192 MiB; 64 MiB is reserved for the last unarchived
    record and a 64 KiB stop receipt. Total compressed bytes never exceed 256 MiB.
    Each record is bounded by reserve minus footer, including the final original
    result. If that bound itself fails, the stream is incomplete and says so.
    Failed appends truncate only this newly created member back to its last fully
    written boundary; an inability to restore that boundary is reported.
    """

    def __init__(
        self,
        path,
        *,
        max_bytes=MAX_COMPRESSED_BYTES,
        reserve_bytes=RESERVE_BYTES,
        footer_bytes=FOOTER_BYTES,
    ):
        if not 0 < footer_bytes < reserve_bytes < max_bytes:
            raise ValueError("invalid archive and failure reserve budgets")
        self.path = Path(path)
        self.max_bytes, self.reserve_bytes, self.footer_bytes = (
            max_bytes,
            reserve_bytes,
            footer_bytes,
        )
        self.bytes_written = self.records_written = 0
        self.pending = None
        self.prefix_intact = True
        self._stream = self.path.open("xb")

    def _append(self, payload):
        if not self.prefix_intact:
            raise OSError("previous incomplete member could not be rolled back")
        if self.bytes_written + len(payload) > self.max_bytes:
            raise ArchiveLimit("total compressed archive capacity exceeded")
        start = self.bytes_written
        try:
            written = self._stream.write(payload)
            if written != len(payload):
                raise OSError("short observation write")
            self._stream.flush()
        except BaseException:
            try:
                self._stream.seek(start)
                self._stream.truncate()
                self._stream.flush()
            except OSError:
                self.prefix_intact = False
            raise
        self.bytes_written += len(payload)
        self.records_written += 1

    def write(self, kind, value):
        payload = _member(kind, value)
        self.pending = (kind, payload)
        if len(payload) > self.reserve_bytes - self.footer_bytes:
            raise ArchiveLimit("one complete record exceeds the declared record limit")
        if self.bytes_written + len(payload) > self.max_bytes - self.reserve_bytes:
            raise ArchiveLimit("ordinary observation budget exhausted; stop without another step")
        self._append(payload)
        self.pending = None

    def stop(self, failure):
        """Keep a pending full record when it fits, then record the actual stop."""
        saved = self.pending is None
        if self.pending is not None:
            _kind, payload = self.pending
            if self.bytes_written + len(payload) <= self.max_bytes - self.footer_bytes:
                self._append(payload)
                self.pending = None
                saved = True
        footer = _member(
            "incomplete",
            dict(failure, last_record_saved=saved, complete_prefix_intact=self.prefix_intact),
        )
        if len(footer) > self.footer_bytes:
            raise ArchiveLimit("failure receipt exceeds its reserved capacity")
        self._append(footer)
        return saved

    def close(self):
        self._stream.close()


def _validate_record(record, outcome):
    expected = {"schema", "stage", "dt_s", "proposal", "entry", outcome}
    if outcome == "retained":
        expected.add("failure")
    if (
        not isinstance(record, dict)
        or set(record) != expected
        or record["schema"] != PROTOCOL_SCHEMA
    ):
        raise ValueError("unexpected original grasp callback protocol")
    for name in ("entry", outcome):
        if set(record[name]) != PUBLIC_FIELDS:
            raise ValueError("callback must contain all seven public observation fields")
    before, after = record["entry"]["reading"], record[outcome]["reading"]
    if outcome == "accepted":
        if (
            after.step_number != before.step_number + 1
            or after.load_time_s != before.time_s
            or after.time_s != before.time_s + record["dt_s"]
            or record["proposal"] is None
        ):
            raise ValueError("accepted callback has inconsistent physical clocks or request")
    elif _public(record["entry"]) != _public(record["retained"]):
        raise ValueError("rejected callback did not retain the complete entry state")


def _tangent_audit(observation, friction):
    history, receipt, events = observation["history"], observation["receipt"], observation["events"]
    checks = dict(
        receipt_present=receipt is not None,
        material_motion=True,
        application_motion=True,
        material_application=True,
        energy_residual=True,
        force_law=True,
        coulomb=True,
        spring_energy=True,
        actual_work=True,
        frictionless=True,
    )
    if receipt is None:
        return dict(
            checks=checks, active_stations=len(history.active), reason="missing accepted receipt"
        )
    checks.update(
        material_motion=receipt.material_motion_error_m <= 1e-6,
        application_motion=receipt.application_motion_error_m <= 1e-6,
        material_application=receipt.material_application_error_m <= 1e-6,
        energy_residual=abs(receipt.energy_residual_j) <= 1e-8,
    )
    work, rows = 0.0, []
    for i, active in enumerate(history.active):
        sample, update = active.sample, active.update
        found = [
            (site, component)
            for event in events.active
            if (event.identity.first, event.identity.second)
            == (sample.first.shape, sample.second.shape)
            for component in event.components
            for site, material in zip(component.load.sites, component.material, strict=True)
            if (
                material.first.local_point_m == sample.first.local_point_m
                and material.second.local_point_m == sample.second.local_point_m
            )
        ]
        if len(found) != 1:
            raise ValueError("accepted history does not identify one actual current material site")
        site, component = found[0]
        traction = np.asarray(site.force_on_second_n) - site.force_n * np.asarray(
            component.load.normal
        )
        elastic = np.asarray(update.elastic_displacement_m)
        energy = 0.5 * KT * float(elastic @ elastic)
        law_error = float(np.max(abs(traction + KT * elastic)))
        cone_excess = float(np.linalg.norm(traction) - component.law.friction * site.force_n)
        actual_work = float(traction @ receipt.actual_application_displacement_m[i])
        work += actual_work
        checks["force_law"] &= law_error <= 1e-9
        checks["coulomb"] &= cone_excess <= 1e-9
        checks["spring_energy"] &= abs(energy - update.stored_energy_j) <= 1e-12
        if friction == 0:
            checks["frictionless"] &= bool(
                np.all(traction == 0)
                and np.all(elastic == 0)
                and update.stored_energy_j == 0
                and update.plastic_slip_m == 0
            )
        rows.append(
            dict(
                identity=active.identity,
                normal_force_n=site.force_n,
                traction_n=traction,
                elastic_m=elastic,
                stored_energy_j=update.stored_energy_j,
                plastic_slip_m=update.plastic_slip_m,
                actual_work_j=actual_work,
                force_law_error_n=law_error,
                cone_excess_n=cone_excess,
            )
        )
    checks["actual_work"] = abs(work - receipt.actual_work_j) <= 1e-9
    return dict(
        checks=checks,
        active_stations=len(history.active),
        stations=rows,
        ended=history.ended,
        dropped_spring_energy_j=history.dropped_spring_energy_j,
        actual_work_j=work,
    )


class ProtocolRecorder:
    """Passive callbacks; no scene reference or control result is accepted."""

    def __init__(self, archive, *, friction):
        self.archive, self.friction = archive, friction
        self.last_record = self.last_kind = self.observer_failure = None
        self.last_saved = False
        self.accepted_records = self.rejected_records = 0
        self.checks = {}
        self._started = perf_counter()
        self._first_loaded_count = None
        self._progress_reported = False

    def _record(self, record, outcome):
        self.last_record, self.last_kind = record, outcome
        self.last_saved = False
        try:
            _validate_record(record, outcome)
            if record["entry"]["reading"].step_number != self.accepted_records:
                raise ValueError("callback omitted or repeated an accepted physical step")
            audit = None
            if outcome == "accepted":
                self.accepted_records += 1
                audit = _tangent_audit(record["accepted"], self.friction)
                for name, passed in audit["checks"].items():
                    self.checks[name] = self.checks.get(name, True) and bool(passed)
                if audit["active_stations"] and self._first_loaded_count is None:
                    self._first_loaded_count = self.accepted_records
            else:
                self.rejected_records += 1
            self.archive.write(outcome, dict(record=record, tangent_audit=audit))
            self.last_saved = True
            if (
                outcome == "accepted"
                and self._first_loaded_count is not None
                and self.accepted_records - self._first_loaded_count == 64
                and not self._progress_reported
            ):
                self._progress_reported = True
                print(
                    json.dumps(
                        dict(
                            event="first-load-plus-64-accepted",
                            elapsed_s=perf_counter() - self._started,
                            accepted_steps=record["accepted"]["reading"].step_number,
                            active_stations=audit["active_stations"],
                            clock_scope="protocol plus passive audit/JSON/gzip/IO",
                        )
                    ),
                    flush=True,
                )
        except Exception as error:
            self.observer_failure = dict(
                type=type(error).__name__, message=str(error), outcome=outcome
            )
            raise

    def accepted(self, record):
        self._record(record, "accepted")

    def rejected(self, record):
        self._record(record, "retained")


def declaration_arguments(info):
    pads = tuple(info["pads"])
    if len(pads) != 2 or any(len(pad.offsets_m) != 4 for pad in pads):
        raise ValueError("original holdout must retain its two four-point pad declarations")
    names = tuple((pad.shape_key.link_name, pad.shape_key.collision_name) for pad in pads)
    if info["case"] == "panda-sphere":
        expected = (
            ("panda_leftfinger", "panda_leftfinger_pad"),
            ("panda_rightfinger", "panda_rightfinger_pad"),
        )
        counter = ((0,), (0,))
    else:
        expected = (("left", "left_rubber"), ("right", "right_rubber"))
        counter = ((0, 1, 2, 3), (4, 5, 6, 7))
    if names != expected:
        raise ValueError("original authored pad order or names changed")
    return dict(
        robot=(0, 10),
        payload_key=info["primary_key"],
        pads=pads,
        station_keys={
            name: tuple(StationKey(0, token) for token in tokens)
            for name, tokens in zip(names, TOKENS, strict=True)
        },
        counter_features=dict(zip(names, counter, strict=True)),
        chart_ids={name: (1001 + i, 2001 + i) for i, name in enumerate(names)},
    )


def _run_protocol(protocol, scene, controller, info, archive):
    recorder = ProtocolRecorder(archive, friction=info["friction"])
    result, failure, status = None, None, "complete"
    result_archive = False
    try:
        result = protocol.run_case(
            scene, controller, info, on_accepted=recorder.accepted, on_rejected=recorder.rejected
        )
        result_archive = True
        if result["accepted_steps"] != recorder.accepted_records:
            raise ValueError("callback stream does not cover the original accepted-step count")
        archive.write("original_result", result)
        if result["execution_failure"] is not None:
            status = "physics_rejected"
    except Exception as error:
        status = (
            "observation_incomplete"
            if recorder.observer_failure is not None or result_archive
            else "execution_incomplete"
        )
        failure = dict(
            type=type(error).__name__,
            message=str(error),
            status=status,
            observer_failure=recorder.observer_failure,
        )
        if recorder.last_record is not None:
            outcome = recorder.last_kind
            snapshot = (
                recorder.last_record.get(outcome, {})
                if isinstance(recorder.last_record, dict)
                else {}
            )
            reading = snapshot.get("reading") if isinstance(snapshot, dict) else None
            failure.update(
                last_outcome=outcome,
                accepted_step=getattr(reading, "step_number", None),
                accepted_time_s=getattr(reading, "time_s", None),
                physical_state_not_rolled_back=None if reading is None else outcome == "accepted",
            )
        try:
            if (
                recorder.last_record is not None
                and not recorder.last_saved
                and archive.pending is None
            ):
                try:
                    archive.write("observer_failed_record", recorder.last_record)
                except ArchiveLimit:
                    pass  # stop() uses the reserved space for this complete pending member.
            failure["last_record_saved"] = archive.stop(failure)
        except Exception as archival_error:
            failure["stop_archive_error"] = dict(
                type=type(archival_error).__name__, message=str(archival_error)
            )
    return dict(
        status=status,
        original_result=result,
        failure=failure,
        accepted_callback_count=recorder.accepted_records,
        rejected_callback_count=recorder.rejected_records,
        tangent_checks=recorder.checks,
        complete_prefix_intact=archive.prefix_intact,
        compressed_bytes=archive.bytes_written,
        written_records=archive.records_written,
    )


def _source_manifest():
    root = Path(__file__).resolve().parents[1]
    package = (root / "src" / "phymeo").resolve()
    if Path(ph.__file__).resolve() != package / "__init__.py":
        raise RuntimeError("repository driver loaded phymeo from a different source tree")
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if (name == "phymeo" or name.startswith("phymeo.")) and filename is not None:
            if not Path(filename).resolve().is_relative_to(package):
                raise RuntimeError(f"mixed phymeo source: {name}: {filename}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    files = set((root / "src" / "phymeo").rglob("*.py"))
    files.update(
        Path(__file__).resolve().parent / (name + ".py")
        for name in (
            "robot_force_acquisition",
            "general_scene_friction",
            "persistent_grasp_holdouts",
        )
    )
    files.add(root / "tests" / "reference_newton" / "robot" / "mimic_jaws.urdf")
    files.add(Path(_example("robot_force_acquisition").PANDA_URDF))
    return dict(
        source_head=head,
        files=[
            dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(files)
        ],
    )


def run_holdout(case, out_dir):
    if case not in CASES:
        raise ValueError("unknown frozen holdout case")
    output = Path(out_dir)
    if not output.is_absolute():
        raise ValueError("out-dir must be a new absolute path")
    output.mkdir(parents=True, exist_ok=False)
    source = _source_manifest()
    protocol = _example("robot_force_acquisition")
    archive = ObservationArchive(output / "observations.jsonl.gz")
    started = perf_counter()
    summary = None
    try:
        scene, controller, info = protocol.make_case(case)
        with scene:
            declaration = declaration_arguments(info)
            registry = ph.material_stations_from_pads(scene, **declaration)
            scene.enable_coupled_tangent(registry, stiffness_n_per_m=KT)
            reading, geometry = scene.observe_scene()
            archive.write(
                "inputs",
                dict(
                    case=case,
                    source=source,
                    original_info=info,
                    declaration=declaration,
                    registry=registry,
                    stiffness_n_per_m=KT,
                    initial_state=scene.state,
                    initial_reading=reading,
                    initial_geometry=geometry,
                    archive_limit_bytes=MAX_COMPRESSED_BYTES,
                    failure_reserve_bytes=RESERVE_BYTES,
                ),
            )
            summary = _run_protocol(protocol, scene, controller, info, archive)
    except Exception as error:
        summary = dict(
            status="setup_or_archive_incomplete",
            failure=dict(type=type(error).__name__, message=str(error)),
            original_result=None,
            tangent_checks={},
            compressed_bytes=archive.bytes_written,
            complete_prefix_intact=archive.prefix_intact,
        )
        try:
            archive.stop(summary["failure"])
            summary["compressed_bytes"] = archive.bytes_written
        except Exception as archival_error:
            summary["failure"]["stop_archive_error"] = dict(
                type=type(archival_error).__name__, message=str(archival_error)
            )
    finally:
        archive.close()
    elapsed = perf_counter() - started
    source_after = _source_manifest()
    summary.update(
        schema="persistent-grasp-holdout-v1",
        case=case,
        elapsed_s=elapsed,
        source=source,
        source_after=source_after,
        source_unchanged=source_after == source,
        archive_path=str(archive.path.resolve()),
        max_compressed_bytes=MAX_COMPRESSED_BYTES,
        scope="HOST original acquisition protocol with explicit kt=200 pad stations",
        clock_scope="configuration/protocol/passive audit/JSON/gzip/IO; source hashing excluded",
    )
    original = summary.pop("original_result")
    # The complete original result is already a full gzip member, not a changed gate set.
    summary["original_checks"] = None if original is None else original["checks"]
    summary["original_outcome"] = (
        None
        if original is None
        else {
            key: value
            for key, value in original.items()
            if key not in {"trace", "events", "checks"}
        }
    )
    summary["passed"] = bool(
        summary["status"] == "complete"
        and summary["source_unchanged"]
        and original is not None
        and all(original["checks"].values())
        and all(summary["tangent_checks"].values())
    )
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_holdout(args.case, args.out_dir)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("case", "status", "passed", "elapsed_s", "compressed_bytes")
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
