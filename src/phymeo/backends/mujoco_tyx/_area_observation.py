"""Optional area fields in the common accepted-entry observation frame."""

from types import MappingProxyType

import numpy as np

from ..mujoco_area_pressure_device import AREA_FIELDS
from ..mujoco_device_controls import _require


class AreaObserver:
    def __init__(self, workspace):
        from ._area_observation_kernels import scatter

        self.workspace, self.scatter = workspace, scatter
        owners = workspace._batch._buckets
        self.capacity = max(
            [1, *(o._constraint_capacity for o in owners if o._area_pressure_pairs)]
        )
        n = len(workspace._batch.world_keys)
        layout = {
            "area_meta": ((n, 4), np.int32),
            "area_ids": ((n, self.capacity, 6), np.int32),
            "area_data": ((n, self.capacity, 15), np.float32),
            "area_measure_words": ((n, self.capacity, 2), np.int32),
        }
        for bank in workspace._banks:
            for name, (shape, dtype) in layout.items():
                _require(name not in bank, "duplicate area observation field")
                bank[name] = workspace._buffer(np.zeros(shape, dtype))
        declared = np.zeros((n, 1), np.int32)
        for row, key in enumerate(workspace._batch.world_keys):
            owner = owners[workspace._batch.layout[key].bucket]
            declared[row, 0] = bool(owner._area_pressure_pairs)
        workspace._masks["area_declared_mask"] = workspace._buffer(declared)

    def refresh_bucket(self, index, owner, destination, options):
        if not owner._area_pressure_pairs:
            return
        ws = self.workspace
        # Re-resolve the current owner after boundary operations. The existing
        # source validator checks all optional fields, shapes, devices and life.
        source = ws._entry_observer._source(owner, index)
        target = ws._banks[destination]
        with ws._tx.use_runtime(ws._runtime):
            self.scatter(
                len(owner.world_keys),
                max(1, owner._constraint_capacity),
                self.capacity,
                len(owner._plan.area_pair_names),
                owner._model.nbody,
                ws._rows[index],
                owner._device.area_body_ids,
                source["entry_meta"],
                *(source[name] for name in AREA_FIELDS),
                *(target[name] for name in AREA_FIELDS),
                ws._status[index],
                **options,
            )
            ws._counts["kernels"] += 1
            ws._entry_observer._check_status(index)

    def catalogs(self):
        """Identity metadata only; numerical values always come from devices."""
        batch = self.workspace._batch
        result = {}
        for key in batch.world_keys:
            owner = batch._buckets[batch.layout[key].bucket]
            receipt = owner._receipts[key]
            result[key] = (
                ()
                if not owner._area_pressure_pairs or receipt is None
                else tuple(
                    MappingProxyType(
                        {
                            name: pair[name]
                            for name in ("identity", "stamp", "sample_ids", "source_ids")
                        }
                    )
                    for pair in receipt.raw_contact_fields["area_pressure"]
                )
            )
        return MappingProxyType(result)
