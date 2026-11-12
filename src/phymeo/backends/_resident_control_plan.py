"""Prepared HOST control columns, independent of consumer action schemas."""

from types import MappingProxyType

import numpy as np


def _binding(batch, key):
    slot = batch._slots[key]
    binding = batch._bindings[slot]
    if key in batch._tree_specs:
        storage = batch._trees
        return storage, storage.rows[key], tuple(batch._tree_specs[key].joint_types), "tree"
    if binding.tree_spec is not None:
        raise NotImplementedError("prepared controls require scalar worlds or ordinary fixed trees")
    if not binding.kind:
        return batch, slot, (), "sphere"
    if binding.kind in (1, 2):
        return batch, slot, ("prismatic",), "slider"
    return batch, slot, ("revolute",), "rotor"


def _field_columns(fields, types, kind):
    metadata, columns = {}, []
    known = {"target_position": 0, "target_velocity": 1, "feedforward": 2, "external_effort": 3}
    for field in fields:
        start = len(columns)
        if kind == "sphere":
            if field != "com_force":
                raise ValueError("sphere controls expose only com_force")
            columns.extend(range(3))
            units, frame, meaning = ("N",) * 3, "world_com", "external_force_at_com"
        elif field in known:
            index = known[field]
            columns.extend(10 * j + index for j in range(len(types)))
            units = tuple(
                ("rad" if t == "revolute" else "m")
                if field == "target_position"
                else ("rad/s" if t == "revolute" else "m/s")
                if field == "target_velocity"
                else ("N m" if t == "revolute" else "N")
                for t in types
            )
            frame = "joint"
            meaning = (
                "total_axial_load"
                if kind == "slider" and field == "external_effort"
                else "held_command"
            )
        elif field in ("com_force", "com_torque") and kind != "slider":
            index = 4 if field == "com_force" else 7
            columns.extend(10 * j + index + c for j in range(len(types)) for c in range(3))
            units = ("N" if field == "com_force" else "N m",) * (3 * len(types))
            frame, meaning = "world_com", "external_load"
        else:
            raise ValueError(f"unsupported {kind} control field {field!r}")
        metadata[field] = MappingProxyType(
            dict(
                offset=start,
                width=len(columns) - start,
                units=units,
                frame=frame,
                reference="body_com" if frame == "world_com" else "generalized_coordinate",
                source=meaning,
                sample="current_held_input",
                access="read_write",
            )
        )
    return metadata, np.array(columns, np.intp)


class ResidentControlPlan:
    """Validated partial control updates, with selected-only rollback scratch.

    Values are commands. They do not publish physical states or actuator output.
    Device commands retain the existing one whole-owner upload on next step.
    """

    contract_version = 1

    def __init__(self, batch, keys, fields):
        self._batch = batch
        self._closed = False
        self._writes = self._reads = self._rollbacks = self._failed_writes = 0
        with batch._lock:
            batch._check_open()
            selected = tuple(batch._slots) if keys is None else tuple(batch._keys(keys))
            fields = tuple(fields)
            if (
                not fields
                or any(type(f) is not str for f in fields)
                or len(set(fields)) != len(fields)
            ):
                raise ValueError("fields require unique explicit control names")
            bindings = [_binding(batch, key) for key in selected]
            if not bindings:
                raise ValueError("prepared controls require a nonempty world selection")
            owner, _, types, kind = bindings[0]
            if any(row[0] is not owner or row[2:] != bindings[0][2:] for row in bindings):
                raise ValueError("selected controls require a homogeneous physical layout")
            metadata, columns = _field_columns(fields, types, kind)
            self._owner = owner
            self._owner_shape = owner._held.shape
            stride = int(np.prod(self._owner_shape[1:]))
            rows = np.array([binding[1] for binding in bindings], np.intp)
            self._indices = (rows[:, None] * stride + columns).reshape(-1)
            self._shape = (len(selected), len(columns))
            self._previous = np.empty(self._shape, np.float32)
            self._magnitude = np.empty(self._shape, np.float32)
            self._valid = np.empty(self._shape, bool)
            self._zero = np.empty(self._shape, bool)
            self._layout = MappingProxyType(
                dict(
                    contract_version=1,
                    world_keys=selected,
                    shape=self._shape,
                    dtype="float32",
                    fields=MappingProxyType(metadata),
                    coordinate_types=types,
                    publication="held_input_only",
                    device_upload="next_step_whole_owner",
                    delay="target_position/target_velocity/feedforward use the model delay; external loads are instantaneous",
                )
            )
            batch._control_plans.add(self)

    @property
    def layout(self):
        return self._layout

    def _current(self):
        if self._closed:
            raise RuntimeError("resident control plan is closed")
        self._batch._check_open()
        current = self._owner._held
        if (
            current.shape != self._owner_shape
            or current.dtype != np.float32
            or not current.flags.c_contiguous
            or not current.flags.writeable
        ):
            raise RuntimeError("resident control owner layout changed; rebuild plan")
        return current.reshape(-1)

    def _validate_buffer(self, values, *, output=False):
        if not isinstance(values, np.ndarray) or values.dtype != np.float32:
            raise TypeError("controls require a float32 NumPy array")
        if (
            values.shape != self._shape
            or not values.flags.c_contiguous
            or (output and not values.flags.writeable)
        ):
            raise ValueError(f"controls require C-contiguous shape {self._shape}")
        if any(
            np.shares_memory(values, a)
            for a in (
                *self._batch._host_alias_arrays(),
                self._previous,
                self._magnitude,
                self._valid,
                self._zero,
            )
        ):
            raise ValueError("caller controls must not alias resident-owned storage")

    def write(self, values):
        """Precheck the entire batch, then replace only selected held columns."""
        batch = self._batch
        with batch._lock:
            current = self._current()
            self._validate_buffer(values)
            np.abs(values, out=self._magnitude)
            np.greater_equal(self._magnitude, np.finfo(np.float32).tiny, out=self._valid)
            np.equal(values, 0, out=self._zero)
            np.logical_or(self._valid, self._zero, out=self._valid)
            np.isfinite(values, out=self._zero)
            np.logical_and(self._valid, self._zero, out=self._valid)
            if not self._valid.all():
                raise ValueError("controls must be finite normal FP32 or zero")
            np.take(current, self._indices, out=self._previous.reshape(-1), mode="clip")
            np.equal(values, self._previous, out=self._valid)
            if self._valid.all():
                self._writes += 1
                return
            try:
                np.put(current, self._indices, values.reshape(-1), mode="clip")
            except Exception:
                self._failed_writes += 1
                try:
                    np.put(current, self._indices, self._previous.reshape(-1), mode="clip")
                    self._rollbacks += 1
                except Exception as error:
                    batch._control_error = error
                raise
            self._owner._dirty = True
            self._writes += 1

    def read_into(self, values):
        """Read current held commands into caller HOST storage; no CUDA synchronization."""
        with self._batch._lock:
            current = self._current()
            self._validate_buffer(values, output=True)
            np.take(current, self._indices, out=values.reshape(-1), mode="clip")
            self._reads += 1

    def stats(self):
        with self._batch._lock:
            return dict(
                closed=self._closed,
                writes=self._writes,
                reads=self._reads,
                failed_writes=self._failed_writes,
                rollbacks=self._rollbacks,
                host_workspace_bytes=0
                if self._closed
                else sum(
                    a.nbytes
                    for a in (
                        self._indices,
                        self._previous,
                        self._magnitude,
                        self._valid,
                        self._zero,
                    )
                ),
                rollback_bytes_per_write=int(np.prod(self._shape)) * 4,
                device_allocations=0,
                immediate_upload_bytes=0,
                immediate_synchronizations=0,
            )

    def close(self):
        with self._batch._lock:
            if self._closed:
                return
            self._batch._control_plans.discard(self)
            self._indices = self._previous = self._magnitude = self._valid = self._zero = None
            self._owner = None
            self._closed = True

    def __enter__(self):
        with self._batch._lock:
            self._current()
        return self

    def __exit__(self, *_):
        self.close()
