"""Cold physical-field layout and direct caller-owned HOST output for resident worlds."""

from types import MappingProxyType

import numpy as np

from ._resident_cleanup import ReleaseStack, cleanup_failed_init, close_dependencies


def _description(batch, key):
    slot = batch._slots[key]
    binding = batch._bindings[slot]
    if key in batch._tree_specs:
        storage = batch._trees
        types = tuple(batch._tree_specs[key].joint_types)
        return storage.states, storage.rows[key], types, "joint", "actuator_drive", True
    if key in batch._floating_specs:
        storage = batch._floating_trees
        types = tuple(batch._floating_specs[key].joint_types)
        # Child coordinates/actual motor output only; the physical free root is
        # exposed by body observations, never invented as six scalar motors.
        return storage.states, storage.rows[key], types, "joint", "actuator_drive", True
    if binding.tree_spec is not None:
        raise NotImplementedError(
            "field read requires scalar worlds or ordinary fixed/floating trees"
        )
    if binding.kind == 0:
        types, frame, effort = ("translation",) * 3, "world_com", "external_com_force"
    else:
        types = ("revolute" if binding.kind in (3, 4) else "prismatic",)
        frame, effort = "joint", "actuator_drive"
    return batch._states, slot, types, frame, effort, False


def _host_buffer(value, shape, dtype, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} requires a NumPy array with dtype {np.dtype(dtype)}")
    if value.shape != shape or not value.flags.c_contiguous or not value.flags.writeable:
        raise ValueError(f"{name} requires writable C-contiguous shape {shape}")


class ResidentFieldReadPlan:
    """One selected layout, persistent gather scratch and borrowed accepted banks.

    No HOST physical snapshot is produced. A CUDA read performs one blocking D2H
    for values, plus one for ticks only when requested. Caller outputs may be
    partially written if a device/transfer failure occurs; physics is unchanged.
    """

    contract_version = 1

    def _initialize_owner(self, batch):
        self._batch = batch
        self._closed = self._closing = False
        self._resources = ReleaseStack()
        self._device_bytes = self._host_bytes = self._allocations = 0
        self._reads = self._failed_reads = self._transfers = self._copied_bytes = 0
        self._indices = self._columns = self._output = self._ticks = self._kernels = None
        self._sources = ()

    def __init__(self, batch, keys, fields):
        self._initialize_owner(batch)
        with batch._lock:
            batch._check_open()
            selected = tuple(batch._slots) if keys is None else tuple(batch._keys(keys))
            fields = tuple(fields)
            allowed = {"position", "velocity", "applied_effort"}
            if not fields or any(type(f) is not str or f not in allowed for f in fields):
                raise ValueError("fields must select position, velocity or applied_effort")
            if len(fields) != len(set(fields)):
                raise ValueError("duplicate fields are not allowed")
            descriptions = [_description(batch, key) for key in selected]
            if descriptions:
                source, _, types, frame, effort, tree = descriptions[0]
                if any(
                    row[0] is not source or row[2:] != descriptions[0][2:]
                    for row in descriptions[1:]
                ):
                    raise ValueError(
                        "selected worlds require one homogeneous physical field layout"
                    )
                self._sources = source
            else:
                types, frame, effort, tree = (), "empty", "unavailable", False
            size = len(types)
            self._shape = (len(selected), len(fields) * size)
            metadata, columns = {}, []
            for name in fields:
                start = len(columns)
                index = {"position": 0, "velocity": 1, "applied_effort": 2}[name]
                columns.extend((i, index) if tree else (0, 3 * index + i) for i in range(size))
                units = tuple(
                    ("rad" if t == "revolute" else "m")
                    if name == "position"
                    else ("rad/s" if t == "revolute" else "m/s")
                    if name == "velocity"
                    else ("N m" if t == "revolute" else "N")
                    for t in types
                )
                metadata[name] = MappingProxyType(
                    dict(
                        offset=start,
                        width=size,
                        units=units,
                        frame=frame,
                        reference="body_com" if frame == "world_com" else "generalized_coordinate",
                        sample="last_accepted_substep"
                        if name == "applied_effort"
                        else "accepted_endpoint",
                        source=effort if name == "applied_effort" else "physical_state",
                        access="read",
                        initial_effort="zero_until_step_or_explicit_restore"
                        if name == "applied_effort"
                        else None,
                    )
                )
            self._layout = MappingProxyType(
                dict(
                    contract_version=1,
                    world_keys=selected,
                    shape=self._shape,
                    dtype="float32",
                    fields=MappingProxyType(metadata),
                    coordinate_types=types,
                    ticks_shape=(len(selected),),
                    ticks_dtype="int32",
                    ticks_unit="accepted_substeps",
                    lifetime="plan_and_batch_open",
                    completion="synchronous_host",
                )
            )
            batch._readback_plans.add(self)
            try:
                if selected:
                    from ._tyx_resident_read_kernels import make_field_read_kernels

                    self._kernels = make_field_read_kernels(tree)
                    indices = np.array(
                        [
                            (d[1], batch._slots[k])
                            for k, d in zip(selected, descriptions, strict=True)
                        ],
                        np.int32,
                    )
                    cols = np.array(columns, np.int32).reshape(-1, 2)
                    with batch._device():
                        self._indices = self._allocate(indices.shape, np.int32, indices)
                        self._columns = self._allocate(cols.shape, np.int32, cols)
                        if batch._mode == "cuda":
                            self._output = self._allocate(self._shape, np.float32)
                        self._ticks = self._allocate((len(selected),), np.int32)
                        warm_output = (
                            self._output
                            if batch._mode == "cuda"
                            else np.empty(self._shape, np.float32)
                        )
                        for bank in (0, 1):
                            batch._launch(
                                self._kernels[bank],
                                *self._shape,
                                self._indices,
                                self._columns,
                                self._sources[bank],
                                batch._ticks[bank],
                                warm_output,
                                self._ticks,
                            )
                        if batch._mode == "cuda":
                            batch._sync()
            except BaseException as error:
                cleanup_failed_init(self, error)
                raise

    @property
    def layout(self):
        return self._layout

    def _allocate(self, shape, dtype, initial=None):
        batch = self._batch
        if batch._mode == "interpret":
            array = np.empty(shape, dtype) if initial is None else initial.copy()
            self._host_bytes += array.nbytes
            return array
        array = (
            batch._tx.cuda_empty(shape, dtype) if initial is None else batch._tx.cuda_array(initial)
        )
        self._resources.callback(self._free, array, array.nbytes)
        self._device_bytes += array.nbytes
        self._allocations += 1
        batch._counts["device_allocations"] += 1
        batch._counts["device_owned_bytes"] += array.nbytes
        batch._counts["readback_plan_owned_bytes"] += array.nbytes
        if initial is not None:
            batch._counts["readback_index_uploaded_bytes"] += initial.nbytes
        return array

    def _free(self, array, size):
        array.free()
        self._device_bytes -= size
        self._batch._counts["device_owned_bytes"] -= size
        self._batch._counts["readback_plan_owned_bytes"] -= size

    def _check_open(self):
        if self._closed or self._closing:
            raise RuntimeError("resident field read plan is closed or closing")
        self._batch._check_open()

    def _borrowed_arrays(self):
        return (*self._sources, self._indices, self._columns, self._output, self._ticks)

    def _launch_read(self, bank, output, ticks):
        self._batch._launch(
            self._kernels[bank],
            *self._shape,
            self._indices,
            self._columns,
            self._sources[bank],
            self._batch._ticks[bank],
            output,
            ticks,
        )

    def _clear_bindings(self):
        self._sources = ()
        self._indices = self._columns = self._output = self._ticks = self._kernels = None

    def read_into(self, values, *, ticks=None):
        """Fill validated HOST arrays in frozen order; return the accepted state epoch."""
        batch = self._batch
        with batch._lock:
            self._check_open()
            _host_buffer(values, self._shape, np.float32, "values")
            if ticks is not None:
                _host_buffer(ticks, (self._shape[0],), np.int32, "ticks")
                if np.shares_memory(values, ticks):
                    raise ValueError("values and ticks must not alias")
            if self._shape[0]:
                borrowed = (
                    *batch._host_alias_arrays(),
                    *batch._ticks,
                    *self._borrowed_arrays(),
                )
                for caller in (values, ticks):
                    if caller is not None and any(
                        isinstance(a, np.ndarray) and np.shares_memory(caller, a) for a in borrowed
                    ):
                        raise ValueError("caller output must not alias resident storage")
            try:
                if self._shape[0]:
                    with batch._device():
                        bank = batch._active
                        device = batch._mode == "cuda"
                        self._launch_read(
                            bank,
                            self._output if device else values,
                            self._ticks if device or ticks is None else ticks,
                        )
                        if device:
                            batch._sync()
                            self._output.copy_to_host(values)
                            self._transfers += 1
                            self._copied_bytes += values.nbytes
                            if ticks is not None:
                                self._ticks.copy_to_host(ticks)
                                self._transfers += 1
                                self._copied_bytes += ticks.nbytes
            except Exception:
                self._failed_reads += 1
                batch._counts["readback_plan_failed_reads"] += 1
                raise
            self._reads += 1
            batch._counts["readback_plan_reads"] += 1
            return batch._state_epoch

    def stats(self):
        with self._batch._lock:
            return dict(
                closed=self._closed,
                closing=self._closing,
                reads=self._reads,
                failed_reads=self._failed_reads,
                device_allocations=self._allocations,
                device_owned_bytes=self._device_bytes,
                host_workspace_bytes=self._host_bytes,
                transfers=self._transfers,
                downloaded_bytes=self._copied_bytes,
            )

    def close(self):
        with self._batch._lock:
            if self._closed:
                return
            self._closing = True
            self._batch._ensure_cleanup_ready()
            close_dependencies(self)
            self._resources.close()
            self._batch._readback_plans.discard(self)
            self._clear_bindings()
            self._host_bytes = 0
            self._closed, self._closing = True, False

    def __enter__(self):
        with self._batch._lock:
            self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
