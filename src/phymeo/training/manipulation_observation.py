"""Explicit HOST success-frame consumption of public, same-entry GPU observations.

This evaluator adapter is not an actor feature producer or an Env. Physics,
goal/history and reader cursor checkpoints remain distinct owners.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from types import MappingProxyType

import numpy as np

from .manipulation_tasks import Scene, SuccessFrame


def _require(condition, message):
    if not condition:
        raise ValueError("manipulation entry: " + message)


def _fingerprint(arrays, row):
    digest = sha256()
    for name in sorted(n for n in arrays if n.startswith("entry_")):
        digest.update(name.encode("ascii"))
        digest.update(arrays[name][row].tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SuccessReaderCheckpoint:
    binding: tuple
    records: tuple


class ManipulationSuccessReader:
    """Consume each public input tick at most once, with whole-read publication.

    scenes and episodes are evaluator declarations, never actor task IDs.
    read() performs an explicit diagnostic HOST readback of existing GPU output.
    Duplicate unchanged records are omitted; skipped or altered ticks reject.
    A reset requires a new episode and a committed per-world reset boundary.
    Restore pairs a cursor with a public frame from the restored physical capsule.
    """

    def __init__(self, workspace, scenes):
        self._lock = RLock()
        self._workspace = workspace
        binding = workspace.binding
        _require(
            binding.get("contact_tokens") is True
            and binding.get("entry_phase") == "accepted-step-input",
            "public accepted-entry contact observation is required",
        )
        _require(
            not any(row.get("area_pressure_pairs") for row in binding["source"]["buckets"]),
            "area contact needs an explicit body-pair success adapter",
        )
        self._keys = tuple(binding["world_keys"])
        _require(
            isinstance(scenes, Mapping)
            and set(scenes) == set(self._keys)
            and all(type(scenes[k]) is Scene for k in self._keys),
            "one declared Scene per world is required",
        )
        self._scenes = dict(scenes)
        source = binding["source"]
        addresses, identities = {}, []
        for world, model_key, bucket, _ in source["worlds"]:
            scene = scenes[world]
            native = source["buckets"][bucket]
            _require(
                native["primary_sha256"] == scene.source_sha256,
                "scene XML differs from observed source",
            )
            _require(source["timestep"] == scene.timestep_s, "scene timestep differs")
            meta = workspace.model_metadata[model_key]
            geometry = tuple(meta["geom_names"])
            body_ids = meta["geom_bodyid"]
            objects = frozenset(geometry.index(name) for name in scene.object_geom_names)
            bodies = {int(body_ids[g]) for g in objects}
            _require(len(bodies) == 1, "object geometries must share one body")
            body = next(iter(bodies))
            joints = [j for j, b in enumerate(meta["jnt_bodyid"]) if b == body]
            # Authored manipulation Scene objects are independent full FREE roots.
            # Read their original WXYZ directly; do not convert an endpoint matrix.
            _require(
                int(meta["body_parentid"][body]) == 0
                and len(joints) == 1
                and int(meta["jnt_type"][joints[0]]) == 0,
                "Scene object requires an independent complete FREE body",
            )
            qa = int(meta["jnt_qposadr"][joints[0]])
            fingers = tuple(geometry.index(name) for name in scene.finger_geom_names)
            support = geometry.index(scene.support_geom_name)
            addresses[world] = (body, qa, objects, fingers, support)
            identities.append(
                (
                    world,
                    scene.physical_binding,
                    model_key,
                    native["model_sha256"],
                    native["archive_sha256"],
                )
            )
        self._addresses = addresses
        self._binding = tuple(identities)
        self._last = {}

    @property
    def binding(self):
        return self._binding

    def read(self, frame, *, episodes):
        with self._lock:
            _require(
                isinstance(episodes, Mapping)
                and set(episodes) == set(self._keys)
                and all(type(episodes[k]) is int and episodes[k] >= 0 for k in self._keys),
                "one nonnegative episode number per world is required",
            )
            arrays = self._workspace.readback(frame)
            pending, output = dict(self._last), {}
            for row, key in enumerate(self._keys):
                valid, tick, after, count, active, overflow = map(int, arrays["entry_meta"][row])
                _require(after == frame.ticks[row], "entry/output tick differs")
                boundary = frame.world_boundaries[row]
                previous = self._last.get(key)
                episode = episodes[key]
                if valid == 0:
                    if after != 0:
                        _require(previous is None, "historical entry is missing; reset required")
                        continue
                    if previous is not None:
                        if boundary == previous[4]:
                            _require(
                                previous[1] == -1 and episode == previous[0],
                                "episode changed without a committed reset",
                            )
                        else:
                            _require(
                                boundary[2] in ("reset", "reset-states") and episode > previous[0],
                                "new episode requires reset; restore the paired cursor",
                            )
                    pending[key] = (episode, -1, 0.0, "", boundary)
                    continue
                _require(
                    valid == 1 and overflow == 0 and tick >= 0 and after == tick + 1,
                    "entry metadata is invalid",
                )
                _require(
                    0 <= active <= count <= arrays["entry_contact_ids"].shape[1],
                    "entry contact counts are invalid",
                )
                stamp = arrays["entry_clock"][row]
                time_s = float(stamp[0]) - float(stamp[1])
                episode = episodes[key]
                fingerprint = _fingerprint(arrays, row)
                if previous is None:
                    _require(tick == 0, "first consumption must start at input tick zero")
                elif boundary == previous[4]:
                    _require(episode == previous[0], "episode changed without a committed reset")
                    if tick == previous[1]:
                        _require(
                            (time_s, fingerprint) == previous[2:4],
                            "same tick has changed historical data",
                        )
                        continue
                    _require(tick == previous[1] + 1, "skipped or reversed physical tick")
                    if previous[1] >= 0:
                        _require(
                            abs(time_s - previous[2] - self._scenes[key].timestep_s) <= 2e-6,
                            "input time is not the next physical step",
                        )
                else:
                    _require(
                        boundary[2] in ("reset", "reset-states")
                        and episode > previous[0]
                        and tick == 0,
                        "new episode requires reset; restore the paired cursor",
                    )
                body, qa, objects, fingers, support = self._addresses[key]
                _require(arrays["entry_body_mask"][row, body] == 1, "object body is invalid")
                finger_loads, support_z = [0.0, 0.0], 0.0
                ids, data = arrays["entry_contact_ids"][row], arrays["entry_contact_data"][row]
                world_wrench = arrays["entry_contact_wrench_world"][row]
                for c in range(count):
                    _require(ids[c, 0] == 1, "contact prefix is invalid")
                    if ids[c, 1] == 0:
                        continue
                    first, second = map(int, ids[c, 2:4])
                    if first in objects and second not in objects:
                        other, sign = second, -1
                    elif second in objects and first not in objects:
                        other, sign = first, 1
                    else:
                        continue
                    normal_force = float(data[c, 13])
                    _require(normal_force >= 0, "negative contact normal load")
                    if other in fingers:
                        finger_loads[fingers.index(other)] += normal_force
                    if other == support:
                        support_z += sign * float(world_wrench[c, 2])
                value = SuccessFrame(
                    world=key,
                    episode=episode,
                    tick=tick,
                    time_s=time_s,
                    object_position=tuple(arrays["entry_body_position"][row, body]),
                    object_wxyz=tuple(arrays["entry_q"][row, qa + 3 : qa + 7]),
                    linear_velocity=tuple(arrays["entry_body_linear_velocity"][row, body]),
                    angular_velocity=tuple(arrays["entry_body_angular_velocity"][row, body]),
                    finger_normal_forces_n=tuple(finger_loads),
                    support_up_force_n=max(0.0, support_z),
                    object_key=self._scenes[key].object_key,
                )
                output[key] = value
                pending[key] = (episode, tick, time_s, fingerprint, boundary)
            self._last = pending
            return MappingProxyType(output)

    def checkpoint(self):
        """Cursor only; the caller must pair it with physics and goal/history."""
        with self._lock:
            return SuccessReaderCheckpoint(
                self._binding,
                tuple((key, *self._last[key]) for key in self._keys if key in self._last),
            )

    def restore(self, checkpoint, *, frame):
        """Validate the paired physical frame, then rebase operational boundaries."""
        with self._lock:
            _require(
                type(checkpoint) is SuccessReaderCheckpoint and checkpoint.binding == self._binding,
                "cursor source binding differs",
            )
            _require(type(checkpoint.records) is tuple, "cursor records must be immutable")
            arrays = self._workspace.readback(frame)
            pending = {}
            for record in checkpoint.records:
                _require(type(record) is tuple and len(record) == 6, "cursor record layout")
                key, episode, tick, time_s, digest, boundary = record
                _require(
                    type(key) is int and key in self._keys and key not in pending,
                    "cursor world identity",
                )
                _require(
                    type(episode) is int and episode >= 0 and type(tick) is int and tick >= -1,
                    "cursor episode/tick",
                )
                _require(
                    type(time_s) is float and np.isfinite(time_s) and time_s >= 0, "cursor time"
                )
                _require(
                    type(boundary) is tuple
                    and len(boundary) == 3
                    and type(boundary[0]) is str
                    and len(boundary[0]) == 32
                    and all(c in "0123456789abcdef" for c in boundary[0])
                    and type(boundary[1]) is int
                    and boundary[1] >= 0
                    and boundary[2] in ("initial", "reset", "reset-states", "restore"),
                    "cursor boundary",
                )
                row = self._keys.index(key)
                current_boundary = frame.world_boundaries[row]
                _require(
                    current_boundary == boundary or current_boundary[2] == "restore",
                    "cursor cannot bypass a committed reset boundary",
                )
                valid, observed_tick, after, *_ = map(int, arrays["entry_meta"][row])
                if tick == -1:
                    _require(
                        digest == "" and time_s == 0.0 and valid == 0 and after == 0,
                        "empty cursor differs from paired physical frame",
                    )
                else:
                    _require(
                        type(digest) is str
                        and len(digest) == 64
                        and all(c in "0123456789abcdef" for c in digest),
                        "cursor fingerprint",
                    )
                    stamp = arrays["entry_clock"][row]
                    _require(
                        valid == 1
                        and observed_tick == tick
                        and after == tick + 1
                        and time_s == float(stamp[0]) - float(stamp[1])
                        and digest == _fingerprint(arrays, row),
                        "cursor differs from paired physical frame",
                    )
                pending[key] = (episode, tick, time_s, digest, frame.world_boundaries[row])
            for row, key in enumerate(self._keys):
                if key not in pending:
                    _require(
                        arrays["entry_meta"][row, 0] == 0
                        and arrays["entry_meta"][row, 2] == 0
                        and frame.world_boundaries[row][2] in ("initial", "restore"),
                        "unconsumed cursor requires an empty physical frame",
                    )
            self._last = pending
