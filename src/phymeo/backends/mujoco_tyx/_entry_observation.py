"""Workspace-owned observation of persistent accepted-entry device records.

Only constructor-time model identities are uploaded. Refresh reads the current
physical owner's accepted entry bank, never HOST receipts or solver scratch.
The workspace owns synchronization, failure handling and atomic publication.
"""

import numpy as np

from ..mujoco_device_controls import _require
from ._body_observation import BODY_FIELDS, BodyObserver
from ._entry_storage import ENTRY_FIELDS, entry_layout

ENTRY_BODY_FIELDS = tuple("entry_" + name for name in BODY_FIELDS)


def _identities(owner):
    """Keep every declared candidate identity, in original producer order."""
    plan, model = owner._plan, owner._model
    mixed = bool(getattr(plan, "constraint_contact", False))
    rows = []
    if mixed or getattr(plan, "normal_contact", False):
        count = int(plan.nslot) if mixed else len(plan.slot_ids)
        for slot in plan.slot_ids[:count]:
            pair, feature, geom1, geom2 = map(int, slot)
            rows.append(
                (
                    geom1,
                    geom2,
                    int(model.geom_bodyid[geom1]),
                    int(model.geom_bodyid[geom2]),
                    feature,
                    int(plan.pair_condim[pair]) if mixed else 1,
                )
            )
    elif plan.pair is not None:
        geom1, geom2 = map(int, plan.pair)
        rows.append(
            (geom1, geom2, int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2]), 0, 1)
        )
    values = np.asarray(rows, dtype=np.int32).reshape(-1, 6) if rows else np.zeros((1, 6), np.int32)
    return len(rows), values


class EntryObserver:
    """Copy, validate and derive public entry fields in the workspace's draft bank."""

    def __init__(self, workspace):
        from ._entry_observation_kernels import finish, scatter

        self.workspace = workspace
        self.scatter, self.finish = scatter, finish
        batch, dims = workspace._batch, workspace._dimensions
        self.capacity = max(1, *(owner._contact_capacity for owner in batch._buckets))
        count, bodies = len(batch.world_keys), dims["nbody"]
        layout = {
            "entry_q": ((count, dims["nq"]), np.float32),
            "entry_v": ((count, dims["nv"]), np.float32),
            "entry_act": ((count, dims["na"]), np.float32),
            "entry_clock": ((count, 4), np.float32),
            "entry_meta": ((count, 6), np.int32),
            "entry_contact_ids": ((count, self.capacity, 8), np.int32),
            "entry_contact_data": ((count, self.capacity, 20), np.float32),
            "entry_body_mask": ((count, bodies), np.int32),
            "entry_contact_wrench_world": ((count, self.capacity, 6), np.float32),
            "entry_contact_body_wrench": ((count, self.capacity, 2, 6), np.float32),
        }
        for bank in workspace._banks:
            for name, (shape, dtype) in layout.items():
                _require(name not in bank, "duplicate entry observation field")
                bank[name] = workspace._buffer(np.zeros(shape, dtype))
        self.sources = []
        self.safe = []
        self.identities = []
        for owner in batch._buckets:
            shape = entry_layout(
                owner._plan,
                len(owner.world_keys),
                (owner._contact_capacity, owner._constraint_capacity),
            )
            self.sources.append(shape)
            self._source(owner, len(self.sources) - 1)
            self.safe.append(
                tuple(
                    workspace._buffer(np.zeros(shape[name][0], np.float32))
                    for name in ("entry_q", "entry_v")
                )
            )
            size, identities = _identities(owner)
            self.identities.append((size, workspace._buffer(identities)))
        self.bodies = BodyObserver(workspace, prefix="entry_")

    def _source(self, owner, index):
        """Recheck current owners after reset/restore without reading device data."""
        ws, device = self.workspace, owner._device
        expected = self.sources[index]
        actual = entry_layout(
            owner._plan,
            len(owner.world_keys),
            (owner._contact_capacity, owner._constraint_capacity),
        )
        _require(actual == expected, "entry source layout changed")
        _require(set(device.entry) == set(expected), "entry source fields differ")
        values = {name: device.entry[name][device.active] for name in ENTRY_FIELDS}
        checks = [(values[name], shape, dtype) for name, (shape, dtype) in expected.items()]
        checks.extend(
            (
                (device.q[device.active], expected["entry_q"][0], np.dtype("float32")),
                (device.v[device.active], expected["entry_v"][0], np.dtype("float32")),
                (device.time[device.active], (len(owner.world_keys), 2), np.dtype("float32")),
            )
        )
        for value, shape, dtype in checks:
            if ws._mode == "interpret":
                _require(
                    isinstance(value, np.ndarray)
                    and value.shape == shape
                    and value.dtype == dtype
                    and value.flags.c_contiguous,
                    "entry source array layout",
                )
            else:
                _require(
                    isinstance(value, ws._tx.CudaArray)
                    and value.shape == shape
                    and np.dtype(value.host_dtype) == dtype
                    and value.device_ordinal == ws._ordinal,
                    "entry source device array layout",
                )
                value._ensure_alive()
        return values

    def _check_status(self, index):
        ws = self.workspace
        if ws._stream is None:
            ws._runtime.sync()
            status = ws._status[index].copy()
        else:
            ws._stream.synchronize()
            status = ws._status[index].to_numpy(stream=ws._stream)
            ws._stream.synchronize()
            ws._counts["status_downloaded_bytes"] += status.nbytes
        _require(not np.any(status), "invalid accepted-entry observation")

    def refresh_bucket(self, index, owner, destination, options):
        """Stage one bucket; the caller holds leaf locks and the source barrier."""
        ws, device, model = self.workspace, owner._device, owner._model
        source = self._source(owner, index)
        target, dims = ws._banks[destination], ws._dimensions
        safe_q, safe_v = self.safe[index]
        identity_count, identities = self.identities[index]
        with ws._tx.use_runtime(ws._runtime):
            self.scatter(
                len(owner.world_keys),
                model.nq,
                model.nv,
                model.na,
                model.nbody,
                model.ngeom,
                owner._contact_capacity,
                identity_count,
                dims["nq"],
                dims["nv"],
                dims["na"],
                self.capacity,
                ws._rows[index],
                identities,
                *(source[name] for name in ENTRY_FIELDS),
                device.q[device.active],
                device.v[device.active],
                device.time[device.active],
                safe_q,
                safe_v,
                *(target[name] for name in ENTRY_FIELDS),
                ws._status[index],
                **options,
            )
            ws._counts["kernels"] += 1
            # FK initializes its status, so transport errors must be consumed first.
            self._check_status(index)
        self.bodies.refresh_bucket(index, owner, destination, options, q=safe_q, v=safe_v)
        with ws._tx.use_runtime(ws._runtime):
            self.finish(
                len(owner.world_keys),
                model.nbody,
                dims["nbody"],
                self.capacity,
                ws._rows[index],
                target["entry_meta"],
                target["entry_contact_ids"],
                target["entry_contact_data"],
                *(target[name] for name in ENTRY_BODY_FIELDS),
                target["entry_body_mask"],
                target["entry_contact_wrench_world"],
                target["entry_contact_body_wrench"],
                ws._status[index],
                **options,
            )
            ws._counts["kernels"] += 1
            self._check_status(index)
