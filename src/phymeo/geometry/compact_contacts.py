"""Sparse pair-contact buffers projected from the validated world collision broad phase."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import groupby

import numpy as np

from ..constraints.collision_shapes import MAX_SHAPES, _integer
from .batched_collision import WorldCollisionWorkspace
from .collision_pipeline import CollisionDetection, CollisionPair, _detect_pairs

__all__ = [
    "CompactContactBuffer",
    "ContactBufferLimits",
    "CompactWorldContactWorkspace",
    "compact_contacts",
]
_MAX = MAX_SHAPES * (MAX_SHAPES - 1) // 2
_CONTACT = {"hit", "touching", "overlap"}
_CLEAR = {"separated", "filtered", "adjacent"}
# For these shape pairs, code 5 returns separation before any narrow validation.
# Keep broad rejections for every other/new kind: invalid hulls, closed meshes,
# plates or cells can legitimately produce diagnostics even when AABBs separate.
_SAFE_AABB_KINDS = {"sphere", "rounded_particle", "particle", "capsule", "triangle", "plane"}


@dataclass(frozen=True)
class ContactBufferLimits:
    """Independent capacities for the published shape-pair contacts and diagnostic rows."""

    contacts: int = _MAX
    diagnostics: int = _MAX

    def __post_init__(self):
        for name in ("contacts", "diagnostics"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))


@dataclass(frozen=True)
class CompactContactBuffer:
    """Full source CollisionPair records, retaining typed witnesses, patches and time brackets."""

    status: str
    backend: str
    required_candidates: int
    required_contacts: int | None
    required_diagnostics: int | None
    contacts: tuple[CollisionPair, ...]
    diagnostics: tuple[CollisionPair, ...]
    overflow: tuple[str, ...]


def _candidate_refusal(backend, count, status="capacity_exceeded"):
    return CompactContactBuffer(
        status,
        backend,
        count,
        None,
        None,
        (),
        (),
        ("candidates",) if status == "capacity_exceeded" else (),
    )


def _project(backend, count, rows, limits):
    contacts = []
    diagnostics = []
    n = d = 0
    for row in rows:
        positive = row.status in _CONTACT
        if positive:
            n += 1
            if n <= limits.contacts:
                contacts.append(row)
        attention = row.status not in _CONTACT and row.status not in _CLEAR
        if positive:
            attention = row.reason in ("support_depth_unresolved", "nonunique_normal") or (
                row.witness is None and (row.sweep is None or row.sweep.witness is None)
            )
        if attention:
            d += 1
            if d <= limits.diagnostics:
                diagnostics.append(row)
    overflow = tuple(
        name
        for name, size, limit in (
            ("contacts", n, limits.contacts),
            ("diagnostics", d, limits.diagnostics),
        )
        if size > limit
    )
    if overflow:
        return CompactContactBuffer("capacity_exceeded", backend, count, n, d, (), (), overflow)
    return CompactContactBuffer("ok", backend, count, n, d, tuple(contacts), tuple(diagnostics), ())


def compact_contacts(detection, limits=None):
    """Extract precisely the compact shape-pair contract from an ordinary full detection."""
    if not isinstance(detection, CollisionDetection):
        raise TypeError("expected CollisionDetection")
    limits = ContactBufferLimits() if limits is None else limits
    if not isinstance(limits, ContactBufferLimits):
        raise TypeError("expected ContactBufferLimits")
    if detection.status != "ok":
        return _candidate_refusal(
            detection.backend, detection.required_candidates, detection.status
        )
    return _project(detection.backend, detection.required_candidates, detection.pairs, limits)


class CompactWorldContactWorkspace(WorldCollisionWorkspace):
    """Single-page same-world solver records with the original broad/narrow semantics.

    CUDA reduces bounds and classifies the existing pair table. HOST selects candidate
    and validation slots from its returned flags before creating any collision records.
    The pair-slot table and the declared per-world candidate capacity retain their original bounds.
    """

    def __init__(self, worlds, backend="host", *, cache_root=None):
        super().__init__(worlds, backend, cache_root=cache_root)
        guarded = np.array([s.kind not in _SAFE_AABB_KINDS for s in self._model.shapes], dtype=bool)
        self._check_aabb = (
            (guarded[self._pairs[:, 0]] | guarded[self._pairs[:, 1]]) if guarded.any() else None
        )
        if self._check_aabb is not None:
            self._check_aabb.flags.writeable = False
        self._default_limits = ContactBufferLimits()
        self._active_limits = None
        self._compact_counts = Counter()

    def detect(self, motion_by_world, policy_by_world, *, limits_by_world=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("world collision workspace is closed")
            supplied = {} if limits_by_world is None else limits_by_world
            if not isinstance(supplied, Mapping):
                raise TypeError("limits_by_world must be a world mapping")
            overrides = {}
            for key, value in supplied.items():
                key = _integer(key, "world")
                if key not in self._keys:
                    raise KeyError(key)
                if not isinstance(value, ContactBufferLimits):
                    raise TypeError("every supplied output limit must be ContactBufferLimits")
                overrides[key] = value
            self._active_limits = overrides
            try:
                return super().detect(motion_by_world, policy_by_world)
            finally:
                self._active_limits = None

    def _empty_results(self, paths, rules, bounds):
        return {key: _project(self._backend, 0, (), self._default_limits) for key in self._order}

    def _results(self, work, paths, rules, bounds, flags):
        counts = {part.key: int(np.count_nonzero(flags[part.pairs] == 0)) for part in self._worlds}

        def mode(part):
            rule = rules[part.key]
            return None if counts[part.key] > rule.candidate_capacity else rule.proximity_patches

        results = {}
        for proximity, members in groupby(self._worlds, mode):
            parts = tuple(members)
            if proximity is None:
                for part in parts:
                    results[part.key] = _candidate_refusal(self._backend, counts[part.key])
                continue
            start, end = parts[0].pairs.start, parts[-1].pairs.stop
            local = flags[start:end]
            keep = local == 0
            if self._check_aabb is not None:
                keep |= (local == 5) & self._check_aabb[start:end]
            selected = np.flatnonzero(keep)
            size = len(selected)
            self._compact_counts["compact_selected_slots"] += size
            self._compact_counts["compact_validation_slots"] += size - sum(
                counts[p.key] for p in parts
            )
            self._compact_counts["compact_skipped_slots"] += len(local) - size
            if not size:
                rows = ()
                slots = selected
            elif size == len(local):
                rows = _detect_pairs(work, paths, local, {}, proximity, pair_start=start)
                slots = None
            else:
                slots = selected + start
                rows = _detect_pairs(work, paths, local[selected], {}, proximity, pair_slots=slots)
            for part in parts:
                if slots is None:
                    left, right = part.pairs.start - start, part.pairs.stop - start
                else:
                    left, right = np.searchsorted(slots, (part.pairs.start, part.pairs.stop))
                limits = self._active_limits.get(part.key, self._default_limits)
                results[part.key] = _project(
                    self._backend, counts[part.key], rows[left:right], limits
                )
        return {key: results[key] for key in self._order}

    def stats(self):
        with self._lock:
            return dict(
                super().stats(),
                **{
                    k: self._compact_counts[k]
                    for k in (
                        "compact_selected_slots",
                        "compact_validation_slots",
                        "compact_skipped_slots",
                    )
                },
            )
