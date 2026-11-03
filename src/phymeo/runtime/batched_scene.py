"""Keyed physical steps across resident rigid motion and cloth--link contact worlds."""

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType

from ..backends._world_step_layout import world_key
from ..backends.tyx_world_batch import FreeSphere, PrismaticSlider, TyxWorldBatch
from ..backends.tyx_world_contact import ClothLinkWorld, TyxWorldContactBatch, WorldContactResult


@dataclass(frozen=True)
class BatchedStepResult:
    """Accepted flags in registration order and the complete actual contact reports."""

    accepted: Mapping[int, bool]
    contact_results: Mapping[int, WorldContactResult]


class TyxBatchedScene:
    """Advance supported independent worlds in bounded solver groups.

    FreeSphere and PrismaticSlider share one device launch. ClothLinkWorld uses the
    batched material evaluation and its physical acceptance checks. An invalid
    world may retain its last state while others advance. A failed device call
    leaves both groups at their previous published frame. Controls set explicitly
    before a call stay held. Reading checkpoints is a separate operation.
    """

    def __init__(self, worlds, mode="cuda", *, cache_root=None):
        self._lock = RLock()
        self._closed = False
        self._completed = self._failed = 0
        resident, material = [], []
        self._is_material = {}
        for raw_key, spec in worlds:
            key = world_key(raw_key)
            if key in self._is_material:
                raise ValueError("duplicate world key")
            if not isinstance(spec, (FreeSphere, PrismaticSlider, ClothLinkWorld)):
                raise TypeError("expected FreeSphere, PrismaticSlider or ClothLinkWorld")
            contact = isinstance(spec, ClothLinkWorld)
            self._is_material[key] = contact
            (material if contact else resident).append((key, spec))
        with ExitStack() as resources:
            self._material = resources.enter_context(
                TyxWorldContactBatch(material, mode, cache_root=cache_root)
            )
            self._resident = resources.enter_context(
                TyxWorldBatch(resident, mode, cache_root=cache_root)
            )
            self._initial_material = self._material.snapshot()
            self._resources = resources.pop_all()

    def _open(self):
        if self._closed:
            raise RuntimeError("batched scene is closed")

    def _selection(self, keys):
        chosen = list(self._is_material) if keys is None else [world_key(k) for k in keys]
        if len(chosen) != len(set(chosen)):
            raise ValueError("duplicate selected world")
        for k in chosen:
            if k not in self._is_material:
                raise KeyError(k)
        resident = [k for k in chosen if not self._is_material[k]]
        material = [k for k in chosen if self._is_material[k]]
        return chosen, resident, material

    def set_controls(self, by_world):
        """Stage persistent COM or drive controls for selected resident worlds."""
        with self._lock:
            self._open()
            self._resident.set_controls(by_world)

    def step(self, h, contact_loads=None, *, site_capacity=None):
        """Compute one common substep, then publish its per-world results."""
        with self._lock:
            self._open()
            loads = {} if contact_loads is None else contact_loads
            try:
                points, contact = self._material._stage_step(h, loads, site_capacity=site_capacity)
                flags = dict.fromkeys(self._is_material, False)
                flags.update((k, entry.accepted) for k, entry in contact.items())
                result = BatchedStepResult(MappingProxyType(flags), MappingProxyType(contact))
                # This is the last operation that can fail on the device. It only
                # swaps the resident state buffers after its status download succeeds.
                flags.update(self._resident.step(h))
            except Exception:
                self._failed += 1
                raise
            self._material._publish_points(points)
            self._completed += 1
            return result

    def snapshot(self, keys=None):
        """Read only the requested world checkpoints in the order supplied."""
        with self._lock:
            self._open()
            ordered, resident, material = self._selection(keys)
            points = self._material.snapshot(material)
            points.update(self._resident.snapshot(resident))
            return {k: points[k] for k in ordered}

    def restore(self, by_world):
        """Validate both groups before publishing any restored material state."""
        with self._lock:
            self._open()
            if not isinstance(by_world, Mapping):
                raise TypeError("checkpoints must be keyed by world")
            _, resident, material = self._selection(by_world)
            points = self._material._stage_restore({k: by_world[k] for k in material})
            self._resident.restore({k: by_world[k] for k in resident})
            self._material._publish_points(points)

    def reset(self, keys=None):
        """Reset only selected worlds, including drive histories and material audits."""
        with self._lock:
            self._open()
            _, resident, material = self._selection(keys)
            points = self._material._stage_restore({k: self._initial_material[k] for k in material})
            self._resident.reset(resident)
            self._material._publish_points(points)

    def stats(self):
        with self._lock:
            self._open()
            return dict(
                worlds=len(self._is_material),
                completed_steps=self._completed,
                failed_steps=self._failed,
                resident=self._resident.stats(),
                material=self._material.stats(),
            )

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._resources.close()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args):
        self.close()
