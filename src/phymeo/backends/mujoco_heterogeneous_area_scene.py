"""Automatic area geometry for a transactional heterogeneous TYX batch.

Current TYX body poses feed HOST finite-surface clipping. This wrapper does not
add device clipping, CCD or a new material law, and does not own batch lifetime.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .mujoco_area_pressure_scene import (
    BodyAreaPressureAsset,
    _check_intersector,
    _checked_publication_frame,
    _geometry_requests,
    _intersect_requests,
)
from .mujoco_area_pressure_state import asset_digest
from .mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
from .mujoco_heterogeneous_checkpoint import MAX_BYTES, MuJoCoTyxHeterogeneousCheckpoint

_MAGIC = b"PHYMEOTYXHETAREASCENE1\n"


def _asset_digest(assets):
    entries = tuple((key, asset_digest(assets[key])) for key in sorted(assets))
    raw = json.dumps(entries, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, init=False)
class MuJoCoTyxHeterogeneousAreaSceneCheckpoint:
    """Immutable checked envelope binding model-local assets to batch state."""

    _bytes: bytes

    def __init__(self, raw):
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise ValueError("expected heterogeneous scene capsule bytes")
        size = raw.nbytes if isinstance(raw, memoryview) else len(raw)
        if size > MAX_BYTES + 256:
            raise ValueError("heterogeneous scene capsule size")
        raw = bytes(raw)
        if not raw.startswith(_MAGIC):
            raise ValueError("heterogeneous scene capsule header")
        digest, sep, payload = raw[len(_MAGIC) :].partition(b"\n")
        if not sep or digest != hashlib.sha256(payload).hexdigest().encode("ascii"):
            raise ValueError("heterogeneous scene capsule checksum")
        asset, sep, capsule = payload.partition(b"\n")
        if not sep or len(asset) != 64 or any(c not in b"0123456789abcdef" for c in asset):
            raise ValueError("heterogeneous scene asset digest")
        MuJoCoTyxHeterogeneousCheckpoint.from_bytes(capsule)
        object.__setattr__(self, "_bytes", raw)

    @classmethod
    def from_bytes(cls, raw):
        return cls(raw)

    @classmethod
    def make(cls, assets, checkpoint):
        if type(checkpoint) is not MuJoCoTyxHeterogeneousCheckpoint:
            raise TypeError("expected heterogeneous batch checkpoint")
        payload = _asset_digest(assets).encode("ascii") + b"\n" + checkpoint.to_bytes()
        return cls(_MAGIC + hashlib.sha256(payload).hexdigest().encode("ascii") + b"\n" + payload)

    def to_bytes(self):
        return self._bytes

    @property
    def asset_sha256(self):
        return self._bytes[len(_MAGIC) + 65 :].split(b"\n", 1)[0].decode("ascii")

    @property
    def batch_checkpoint(self):
        return MuJoCoTyxHeterogeneousCheckpoint.from_bytes(
            self._bytes[len(_MAGIC) + 65 :].split(b"\n", 1)[1]
        )


class MuJoCoTyxHeterogeneousAreaPressureScene:
    """Per-model immutable assets; current geometry and physics commit together."""

    def __init__(self, batch, assets, *, intersector=None, publication_frame="world"):
        self._publication_frame = _checked_publication_frame(publication_frame)
        if not isinstance(batch, MuJoCoTyxHeterogeneousBatch):
            raise TypeError("expected MuJoCoTyxHeterogeneousBatch")
        if not isinstance(assets, Mapping):
            raise ValueError("assets must map each area model key to its ordered assets")
        assets = dict(assets)
        with batch._lock:
            batch._check()
            _check_intersector(intersector, batch._mode)
            self._intersector = intersector
            models = {spec.key: spec for spec in batch._models if spec.area_pressure_pairs}
            if not models or set(assets) != set(models):
                raise ValueError("assets must cover exactly the declared area model keys")
            owned = {}
            for key, spec in models.items():
                rows = assets[key]
                if not isinstance(rows, tuple) or not 1 <= len(rows) <= 16:
                    raise ValueError("assets must contain 1..16 ordered pairs per model")
                if any(not isinstance(a, BodyAreaPressureAsset) for a in rows):
                    raise ValueError("expected BodyAreaPressureAsset")
                copies = tuple(
                    BodyAreaPressureAsset(a.identity, a.pressure_local, a.rigid_local) for a in rows
                )
                pairs = tuple((a.identity.compliant_body, a.identity.rigid_body) for a in copies)
                if pairs != spec.area_pressure_pairs:
                    raise ValueError("assets must match all declared area pairs in order")
                if spec.area_friction_laws and tuple(a.identity for a in copies) != tuple(
                    law.identity for law in spec.area_friction_laws
                ):
                    raise ValueError("area friction requires complete body/material identities")
                owned[key] = copies
            self.batch = batch
            self._assets = MappingProxyType(owned)
            self._asset_sha256 = _asset_digest(self._assets)
            self._empty = MappingProxyType({})
            self._last_geometry = self._empty
            self._geometry_revision = None

    @property
    def publication_frame(self):
        """Coordinate origin for clipping publication across all worlds."""
        return self._publication_frame

    @property
    def assets(self):
        return self._assets

    @property
    def asset_sha256(self):
        """Stable identity of owned model-local assets, independent of live state."""
        if _asset_digest(self._assets) != self._asset_sha256:
            raise ValueError("owned area assets changed after scene construction")
        return self._asset_sha256

    @property
    def last_geometry(self):
        """Last accepted input geometry, invalidated by any external revision."""
        with self.batch._lock:
            self.batch._check()
            if self.batch._revision != self._geometry_revision:
                return self._empty
            return self._last_geometry

    def step(self, h, *, expected_revision):
        batch = self.batch
        with batch._transaction():
            if type(expected_revision) is not int or expected_revision != batch._revision:
                raise ValueError("stale heterogeneous area scene revision")
            if not np.isfinite(h) or float(h) != batch._h:
                raise ValueError("scene timestep must equal every compiled timestep")
            requests_by_world = {}
            # Reset/restore replace owners. Resolve the current buckets each call;
            # models sharing a bucket may still carry different local materials.
            for owner in batch._buckets:
                if not owner._area_pressure_pairs:
                    continue
                assets_by_world = {
                    key: self._assets[batch._layout[key].model_key] for key in owner.world_keys
                }
                requests_by_world.update(
                    _geometry_requests(owner, assets_by_world, self.publication_frame)
                )
            generated, surfaces = _intersect_requests(requests_by_world, self._intersector)
            publication = MappingProxyType(
                {key: generated[key] for key in batch._world_keys if key in generated}
            )
            accepted = batch._step_area_pressure_locked(
                h, surfaces, expected_revision=expected_revision
            )
            # No fallible work remains after the all-bucket physical commit.
            self._last_geometry = publication
            self._geometry_revision = batch._revision
            return accepted

    def checkpoint(self):
        with self.batch._lock:
            self.batch._check()
            return MuJoCoTyxHeterogeneousAreaSceneCheckpoint.make(
                self._assets, self.batch.checkpoint()
            )

    def restore(self, checkpoint, *, expected_revision):
        with self.batch._lock:
            self.batch._check()
            if type(checkpoint) is not MuJoCoTyxHeterogeneousAreaSceneCheckpoint:
                raise ValueError("expected heterogeneous area scene checkpoint")
            if checkpoint.asset_sha256 != self._asset_sha256:
                raise ValueError("heterogeneous scene asset binding differs")
            result = self.batch.restore(
                checkpoint.batch_checkpoint, expected_revision=expected_revision
            )
            self._last_geometry = self._empty
            self._geometry_revision = None
            return result

    def reset(self, keys=None):
        with self.batch._lock:
            self.batch._check()
            before = self.batch._revision
            result = self.batch.reset(keys)
            if self.batch._revision != before:
                self._last_geometry = self._empty
                self._geometry_revision = None
            return result
