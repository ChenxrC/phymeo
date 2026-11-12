"""Prepared body observations; borrow accepted geometry, never run another FK."""

from types import MappingProxyType

import numpy as np

from ._resident_cleanup import cleanup_failed_init
from ._resident_field_read import ResidentFieldReadPlan

# code, components, unit, reference
FIELDS = {
    "position": (0, 3, "m", "body_origin"),
    "com_position": (3, 3, "m", "body_com"),
    "rotation": (6, 9, "1", "body_to_world_row_major"),
    "linear_velocity": (15, 3, "m/s", "body_origin"),
    "com_velocity": (18, 3, "m/s", "body_com"),
    "angular_velocity": (21, 3, "rad/s", "body"),
}


class ResidentBodyReadPlan(ResidentFieldReadPlan):
    """Same completion, validation and retryable cleanup as coordinate field reads.

    Fixed ordinary trees expose moving links in model order, not a synthetic
    anchor body with fabricated COM. Translation-only spheres expose one COM body
    with identity rotation. Ordinary floating trees expose the massive root then
    moving child links; accepted COM twists supply velocities without another FK.
    """

    def __init__(self, batch, keys, fields):
        self._initialize_owner(batch)
        self._tree = None
        self._floating = None
        self._twists = ()
        self._geometry = ()
        self._topology = ()
        with batch._lock:
            batch._check_open()
            selected = tuple(batch._slots) if keys is None else tuple(batch._keys(keys))
            fields = tuple(fields)
            if not fields or any(type(f) is not str or f not in FIELDS for f in fields):
                raise ValueError(f"body fields must select from {tuple(FIELDS)}")
            if len(fields) != len(set(fields)):
                raise ValueError("duplicate body fields are not allowed")
            descriptions = []
            for key in selected:
                if key in batch._tree_specs:
                    descriptions.append(("fixed_tree", batch._tree_specs[key].n))
                elif key in batch._floating_specs:
                    descriptions.append(("floating_tree", batch._floating_specs[key].b))
                elif batch._bindings[batch._slots[key]].kind == 0:
                    descriptions.append(("translation_sphere", 1))
                else:
                    raise NotImplementedError(
                        "body read requires spheres or ordinary fixed/floating trees"
                    )
            if descriptions and any(d != descriptions[0] for d in descriptions):
                raise ValueError("selected worlds require one homogeneous body layout")
            family, count = descriptions[0] if descriptions else ("empty", 0)
            columns, metadata = [], {}
            for name in fields:
                code, width, unit, reference = FIELDS[name]
                metadata[name] = MappingProxyType(
                    dict(
                        offset=len(columns),
                        width=count * width,
                        shape=(count, width),
                        unit=unit,
                        frame="world",
                        reference=reference,
                        access="read",
                        sample="accepted_endpoint",
                    )
                )
                columns.extend((b, code + c) for b in range(count) for c in range(width))
            self._shape = (len(selected), len(columns))
            self._layout = MappingProxyType(
                dict(
                    contract_version=1,
                    world_keys=selected,
                    shape=self._shape,
                    dtype="float32",
                    fields=MappingProxyType(metadata),
                    body_count=count,
                    family=family,
                    body_order=(
                        "root_then_model_links"
                        if family == "floating_tree"
                        else "model_links"
                        if family == "fixed_tree"
                        else "single_com_body"
                    ),
                    ticks_shape=(len(selected),),
                    ticks_dtype="int32",
                    ticks_unit="accepted_substeps",
                    lifetime="plan_and_batch_open",
                    completion="synchronous_host",
                )
            )
            if family == "fixed_tree":
                self._tree = batch._trees
                self._sources = self._tree.states
                self._geometry = self._tree.fk
                self._topology = (
                    self._tree.meta,
                    self._tree.parents,
                    self._tree.kinds,
                    self._tree.anchor,
                )
                rows = [(self._tree.rows[k], batch._slots[k]) for k in selected]
            elif family == "floating_tree":
                self._floating = batch._floating_trees
                self._sources = self._floating.states
                self._geometry = self._floating.fk
                self._twists = self._floating.twist
                self._topology = (self._floating.anchor,)
                rows = [(self._floating.rows[k], batch._slots[k]) for k in selected]
            else:
                self._sources = batch._states
                rows = [(batch._slots[k], batch._slots[k]) for k in selected]
            batch._readback_plans.add(self)
            try:
                if selected:
                    from ._tyx_resident_body_read import (
                        make_body_read_kernels,
                        make_floating_body_read_kernels,
                    )

                    self._kernels = (
                        make_floating_body_read_kernels()
                        if self._floating is not None
                        else make_body_read_kernels(family == "fixed_tree")
                    )
                    with batch._device():
                        ids = np.asarray(rows, np.int32)
                        cols = np.asarray(columns, np.int32)
                        self._indices = self._allocate(ids.shape, np.int32, ids)
                        self._columns = self._allocate(cols.shape, np.int32, cols)
                        if batch._mode == "cuda":
                            self._output = self._allocate(self._shape, np.float32)
                        self._ticks = self._allocate((len(selected),), np.int32)
                        output = (
                            self._output
                            if batch._mode == "cuda"
                            else np.empty(self._shape, np.float32)
                        )
                        for bank in (0, 1):
                            self._launch_read(bank, output, self._ticks)
                        if batch._mode == "cuda":
                            batch._sync()
            except BaseException as error:
                cleanup_failed_init(self, error)
                raise

    def _launch_read(self, bank, output, ticks):
        if self._floating is not None:
            self._batch._launch(
                self._kernels[bank],
                *self._shape,
                self._indices,
                self._columns,
                self._batch._ticks[bank],
                *self._topology,
                self._geometry[bank],
                self._twists[bank],
                output,
                ticks,
            )
            return
        extra = (*self._topology, self._geometry[bank]) if self._tree is not None else ()
        self._batch._launch(
            self._kernels[bank],
            *self._shape,
            self._indices,
            self._columns,
            self._sources[bank],
            self._batch._ticks[bank],
            *extra,
            output,
            ticks,
        )

    def _borrowed_arrays(self):
        return (*super()._borrowed_arrays(), *self._geometry, *self._topology, *self._twists)

    def _clear_bindings(self):
        super()._clear_bindings()
        self._tree = None
        self._floating = None
        self._twists = ()
        self._geometry = self._topology = ()
