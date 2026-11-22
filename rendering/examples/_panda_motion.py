"""Sample the real 2.25 s Panda archive without interpolation or physics steps.

At the defaults, 108 right-edge 48 Hz samples are presented at 24 fps (4.5 s,
nominal 0.5x). Physical timestamps and q/cube poses remain original records.
A distributed subset requires its SHA256 from an external trusted receipt.
"""

import hashlib
import json
import math
import re
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from phymeo_rendering.adapters.panda_replay import JOINT_NAMES, MODEL_SHA256, PandaReplaySource

ARCHIVE_SHA256 = "d74e42dd2d3f7735e197b0a845e602217b243a8513c73b7ab09c02995ccb666b"
SCHEMA = "phymeo-panda-motion-subset-v1"
CONVENTION = "right-edge-nearest-original-record-earlier-tie"
STATE_PHASE = (
    "post-step q/cube/xyzw/t; forces and pad_points_before_m belong to previous step input geometry"
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _rate(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive real number")
    _require(math.isfinite(value) and value > 0, f"invalid {name}")
    return Fraction(str(value))


def _schedule(record_count, dt_s, sample_hz):
    _require(type(record_count) is int and record_count > 0, "invalid archive record count")
    dt, rate = _rate(dt_s, "dt_s"), _rate(sample_hz, "sample_hz")
    _require(rate <= 1 / dt, "sample_hz exceeds original record frequency")
    duration = record_count * dt
    slots = math.ceil(duration * rate)
    indices, targets = [], []
    for k in range(slots):
        target = min(Fraction(k + 1) / rate, duration)
        coordinate = target / dt
        lower, remainder = divmod(coordinate.numerator, coordinate.denominator)
        tick = lower + int(2 * remainder > coordinate.denominator)
        index = min(record_count, max(1, tick)) - 1
        if indices and index == indices[-1]:
            # A short final partial slot can select the same endpoint. Keep it
            # once, with the final target, rather than invent a repeated state.
            targets[-1] = float(target)
        else:
            indices.append(index)
            targets.append(float(target))
    return tuple(indices), tuple(targets)


def sample_indices(*, record_count, dt_s, sample_hz=48):
    """Nearest original endpoint per right-edge target, exact-decimal tie rule."""
    return _schedule(record_count, dt_s, sample_hz)[0]


def _inputs(raw):
    values = json.loads(raw)
    _require(values.get("schema") == "panda-box-control-v1", "unknown archive input schema")
    _require(tuple(values.get("joint_names", ())) == JOINT_NAMES, "archive joint order mismatch")
    _require(values.get("urdf_sha256") == MODEL_SHA256, "archive URDF model mismatch")
    _require(
        values.get("steps") == 3600 and values.get("dt_s") == 0.000625,
        "foreign archive duration or timestep",
    )
    return values


def _raw_field(text, name):
    match = re.search(r'"' + re.escape(name) + r'"\s*:\s*', text)
    _require(match is not None, f"missing archive field {name}")
    _, end = json.JSONDecoder().raw_decode(text, match.end())
    return text[match.end() : end]


def _from_archive(raw, sample_hz, playback_fps):
    _require(_hash(raw) == ARCHIVE_SHA256, "original archive SHA256 mismatch")
    text = raw.decode("utf-8")
    inputs_raw = _raw_field(text, "inputs")
    inputs = _inputs(inputs_raw)
    indices, _ = _schedule(inputs["steps"], inputs["dt_s"], sample_hz)
    _rate(playback_fps, "playback_fps")
    selected = set(indices)
    match = re.search(r'"trace"\s*:\s*\[', text)
    _require(match is not None, "archive trace missing")
    pos = match.end()
    decoder = json.JSONDecoder()
    records = []
    for index in range(inputs["steps"]):
        while text[pos].isspace():
            pos += 1
        record, end = decoder.raw_decode(text, pos)
        _require(record["t"] == (index + 1) * inputs["dt_s"], "archive record time/index mismatch")
        if index in selected:
            records.append(dict(source_index=index, record_json=text[pos:end]))
        pos = end
        while text[pos].isspace():
            pos += 1
        if index + 1 < inputs["steps"]:
            _require(text[pos] == ",", "archive record count mismatch")
            pos += 1
        else:
            _require(text[pos] == "]", "archive record count mismatch")
    return dict(
        schema=SCHEMA,
        archive_sha256=ARCHIVE_SHA256,
        archive_record_count=inputs["steps"],
        inputs_json=inputs_raw,
        sample_hz=sample_hz,
        playback_fps=playback_fps,
        selection=CONVENTION,
        records=records,
    )


def _read(path):
    # Full known archive is 7.6 MB; reject accidental giant/foreign inputs before
    # parsing. This is file I/O only and does not load a physical runtime.
    with Path(path).open("rb") as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    _require(len(raw) <= 16 * 1024 * 1024, "motion source exceeds bounded archive size")
    return raw


def write_motion_subset(archive_path, destination, *, sample_hz=48, playback_fps=24):
    """Write a new portable JSON subset; save returned SHA in a trusted receipt.

    Output is exclusive (never silently overwrite previous evidence). It stores
    each complete selected record's original JSON tokens, including velocities,
    loads and input witnesses; q/cube/time are never reconstructed from knots.
    """
    payload = _from_archive(_read(archive_path), sample_hz, playback_fps)
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    with Path(destination).open("xb") as stream:
        stream.write(encoded)
    return dict(
        file=str(Path(destination).resolve()),
        sha256=_hash(encoded),
        archive_sha256=ARCHIVE_SHA256,
        frames=len(payload["records"]),
        sample_hz=sample_hz,
        playback_fps=playback_fps,
        selection=CONVENTION,
    )


class PandaMotionSource:
    """Prefetch selected original records through the existing checked replay FK.

    Accepts either the exact original archive or a generated subset accompanied
    by ``expected_subset_sha256`` from trusted documentation/receipt. A subset's
    self-declared archive hash alone is not proof of its source. Source indices
    remain original zero-based archive indices, never video-frame indices.
    """

    def __init__(
        self,
        scene,
        urdf_path,
        source_path,
        *,
        sample_hz=48,
        playback_fps=24,
        expected_subset_sha256=None,
    ):
        _rate(playback_fps, "playback_fps")
        raw = _read(source_path)
        digest = _hash(raw)
        if digest == ARCHIVE_SHA256:
            payload = _from_archive(raw, sample_hz, playback_fps)
        else:
            _require(
                expected_subset_sha256 is not None,
                "portable subset requires trusted SHA: expected_subset_sha256",
            )
            _require(
                isinstance(expected_subset_sha256, str) and digest == expected_subset_sha256,
                "motion subset SHA256 mismatch",
            )
            payload = json.loads(raw)
        _require(payload.get("schema") == SCHEMA, "unknown motion subset schema")
        _require(payload.get("archive_sha256") == ARCHIVE_SHA256, "foreign archive identity")
        _require(payload.get("selection") == CONVENTION, "foreign sampling convention")
        _require(
            payload.get("sample_hz") == sample_hz and payload.get("playback_fps") == playback_fps,
            "subset sampling/playback rates differ from requested rates",
        )
        inputs = _inputs(payload["inputs_json"])
        _require(payload.get("archive_record_count") == inputs["steps"], "archive count mismatch")
        indices, targets = _schedule(inputs["steps"], inputs["dt_s"], sample_hz)
        records = payload["records"]
        _require(
            isinstance(records, list) and tuple(r["source_index"] for r in records) == indices,
            "subset original record selection/order mismatch",
        )
        _require(
            all(
                type(r["source_index"]) is int and isinstance(r["record_json"], str)
                for r in records
            ),
            "invalid raw record representation",
        )
        # Reuse the existing public replay parser, provenance binding and URDF FK
        # unchanged. Temporary conversion files are removed before returning.
        excerpt = (
            '{"schema":"phymeo-panda-archive-excerpt-v1","inputs":'
            + payload["inputs_json"]
            + ',"records":['
            + ",".join(
                '{"source_index":' + str(r["source_index"]) + ',"record":' + r["record_json"] + "}"
                for r in records
            )
            + "]}"
        ).encode("utf-8")
        manifest = dict(
            schema="phymeo-panda-archive-manifest-v1",
            archive_sha256=ARCHIVE_SHA256,
            archive_record_count=inputs["steps"],
            excerpt_sha256=_hash(excerpt),
            model_sha256=MODEL_SHA256,
            live_physics_identity=None,
            base_pose=dict(position_m=[0.0, 0.0, 0.0], orientation_xyzw=[0.0, 0.0, 0.0, 1.0]),
            state_phase=STATE_PHASE,
            records=[
                dict(
                    source_index=r["source_index"],
                    raw_record_sha256=_hash(r["record_json"].encode("utf-8")),
                )
                for r in records
            ],
        )
        with TemporaryDirectory(prefix="phymeo-panda-replay-") as temporary:
            root = Path(temporary)
            (root / "trace.json").write_bytes(excerpt)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self._source = PandaReplaySource(scene, urdf_path, root / "trace.json")
        frames = tuple(self._source.frames())
        self._indices = {frame.key: i for i, frame in enumerate(frames)}
        self._source_digest = digest
        self._source_kind = "original-archive" if digest == ARCHIVE_SHA256 else "trusted-subset"
        self._timeline = dict(
            frames=len(frames),
            sample_hz=sample_hz,
            playback_fps=playback_fps,
            selection=CONVENTION,
            source_window_s=[0.0, inputs["steps"] * inputs["dt_s"]],
            first_record_time_s=frames[0].time_s,
            last_record_time_s=frames[-1].time_s,
            playback_duration_s=len(frames) / playback_fps,
            nominal_speed=playback_fps / sample_hz,
            source_indices=list(indices),
            target_times_s=list(targets),
            physical_times_s=[frame.time_s for frame in frames],
            presentation_times_s=[i / playback_fps for i in range(len(frames))],
            interpolation=False,
            repeated_records=False,
            loops=1,
            note="CFR presentation clock is separate from original endpoint times; partial final slots may be merged",
        )

    @property
    def geometry(self):
        return self._source.provenance(next(self._source.frames()))["geometry"]

    def scene(self):
        return self._source.scene()

    def frames(self):
        return self._source.frames()

    def timeline(self):
        return deepcopy(self._timeline)

    def provenance(self, frame):
        result = self._source.provenance(frame)
        index = self._indices[frame.key]
        result["motion"] = dict(
            source_kind=self._source_kind,
            source_file_sha256=self._source_digest,
            frame_index=index,
            target_time_s=self._timeline["target_times_s"][index],
            presentation_time_s=self._timeline["presentation_times_s"][index],
            sample_hz=self._timeline["sample_hz"],
            playback_fps=self._timeline["playback_fps"],
            selection=CONVENTION,
            interpolation=False,
        )
        return result
