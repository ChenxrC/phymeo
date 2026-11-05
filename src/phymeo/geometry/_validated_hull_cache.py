"""Bounded workspace-owned copies of successfully validated exact-pose hulls.

This prepares material geometry only. Query policy, world-coordinate accuracy,
closest features, contact witnesses and continuous-path proofs remain consumers'
responsibility. There is one entry per registered hull, not a pose history.
"""

from copy import deepcopy
from dataclasses import fields, is_dataclass
from itertools import islice
from struct import pack

import numpy as np

from ..constraints.collision_shapes import (
    MAX_SHAPES,
    MAX_VERTICES,
    CollisionMotion,
    CollisionShape,
)
from . import _convex_hull


def _representation(value):
    """Exact registration representation, independent of dataclass equality."""
    if is_dataclass(value):
        return (type(value), tuple((f.name, _representation(getattr(value, f.name))) for f in fields(value)))
    if isinstance(value, float):
        return (float, pack("<d", value))
    if isinstance(value, tuple):
        return (tuple, tuple(_representation(v) for v in value))
    if value is None or isinstance(value, (str, int, bool)):
        return (type(value), value)
    # Legal hull registrations have no arrays or other geometry assets. A
    # modified registration must not acquire an unbound mutable extension.
    raise ValueError("unsupported registered hull representation")


def _array_key(value):
    # CollisionMotion constructs owned, finite FP64 arrays. Do not silently
    # promote an externally replaced representation into that validated domain.
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.float64)
        or value.ndim != 2
        or value.shape[1] != 3
        or not np.isfinite(value).all()
    ):
        return None
    return (value.dtype.str, value.shape, value.strides, value.tobytes(order="C"))


def _motion_key(path):
    if type(path) is not CollisionMotion:
        return None
    start = _array_key(path.start)
    end = None if path.end is None else _array_key(path.end)
    if start is None or (path.end is not None and end is None):
        return None
    if path.moving:
        return None
    # Distinguish equal endpoints, no endpoints, and zero-angle turns, even
    # though all are stationary. Never infer reuse from a rigid transform or
    # local shape equivalence rather than the actual stored world vertices.
    return (
        start,
        end,
        _representation(path.turn),
        _representation(path.body_orientation_xyzw),
    )


def _builder_key(builder):
    # A class can keep its identity while its actual validation constructor is
    # replaced (including legitimate instrumentation). Function factories are
    # bound by identity, not by a name or an equality operator.
    return (builder, builder.__init__ if isinstance(builder, type) else None)


def _same_builder(first, second):
    return first[0] is second[0] and first[1] is second[1]


class ValidatedHullCache:
    """One successful stationary preparation per bound hull under its query lock.

    ``get`` always returns a private deep copy. The original builder result is
    copied before storage as well, so neither consumers nor a factory retaining
    its result can mutate a future query. Only successful preparations are kept;
    a replacement or bypass evicts that hull's preceding entry before building.
    """

    def __init__(self, shapes):
        registered = tuple(islice(iter(shapes), MAX_SHAPES + 1))
        if (
            len(registered) > MAX_SHAPES
            or not all(isinstance(s, CollisionShape) for s in registered)
            or len({s.key for s in registered}) != len(registered)
            or sum(len(s.vertex_ids) for s in registered) > MAX_VERTICES
        ):
            raise ValueError("invalid registered hull shape table or capacity")
        self._registered = {
            id(s): (s, _representation(s)) for s in registered if s.kind == "hull"
        }
        self._entries = {}
        self._attempts = self._builds = self._hits = 0

    def get(self, shape, path, builder=None):
        """Return exact material preparation; resolve the live builder on every call."""
        registered = self._registered.get(id(shape))
        if registered is None or registered[0] is not shape:
            raise ValueError("query requires the owning workspace's registered hull")
        token = id(shape)
        try:
            representation = _representation(shape)
        except (ValueError, TypeError):
            self._entries.pop(token, None)
            raise
        if representation != registered[1]:
            self._entries.pop(token, None)
            raise ValueError("registered hull registration changed after binding")
        builder = _convex_hull.HullShape if builder is None else builder
        factory = _builder_key(builder)
        try:
            motion = _motion_key(path)
        except (ValueError, TypeError, AttributeError):
            # Malformed replacements go through the real constructor each time,
            # never through a previously successful cached representation.
            motion = None
        entry = self._entries.get(token)
        if motion is not None and entry is not None and entry[0] == motion and _same_builder(entry[1], factory):
            result = deepcopy(entry[2])
            self._hits += 1
            return result
        self._entries.pop(token, None)
        self._attempts += 1
        prepared = builder(shape, path)
        self._builds += 1
        if motion is None:
            return prepared
        owned = deepcopy(prepared)
        result = deepcopy(owned)
        self._entries[token] = (motion, factory, owned)
        return result

    def stats(self):
        return dict(
            validated_hull_attempts=self._attempts,
            validated_hull_builds=self._builds,
            validated_hull_hits=self._hits,
        )

    def __len__(self):
        return len(self._entries)

    def clear(self):
        self._entries.clear()
