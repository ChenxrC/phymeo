"""Opt-in TYX batch of independent translating spheres and scalar fixed-base sliders."""

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from threading import RLock

import numpy as np

from ._world_step_layout import (
    PROPERTY_WIDTH,
    STATE_WIDTH,
    FreeSphere,
    PrismaticSlider,
    WorldCheckpoint,
    as_fp32,
    bind_world,
    world_key,
)

__all__ = ["FreeSphere", "PrismaticSlider", "TyxWorldBatch", "WorldCheckpoint"]


class TyxWorldBatch:
    """One device kernel advances all worlds using a pair of persistent state buffers.

    Controls stay held until changed. A step returns a compact per-world acceptance map.
    Numeric rejection retains the affected world's physical state, tick and delayed packet.
    A failed batch submission leaves its input buffer current. Reading the committed state
    after a device loss requires a functioning driver or a previously saved checkpoint.
    """

    def __init__(self, worlds, mode="cuda", *, cache_root=None):
        if mode not in ("cuda", "interpret"):
            raise ValueError("resident mode must be cuda or interpret")
        self._mode = mode
        self._lock = RLock()
        self._closed = False
        self._unavailable = False
        self._resources = ExitStack()
        self._active = 0
        self._dirty = False
        self._counts = dict.fromkeys(
            (
                "device_allocations",
                "device_owned_bytes",
                "topology_uploaded_bytes",
                "initial_uploaded_bytes",
                "control_uploaded_bytes",
                "restore_uploaded_bytes",
                "status_downloaded_bytes",
                "summary_downloaded_bytes",
                "snapshot_downloaded_bytes",
                "kinematics_downloaded_bytes",
                "launch_attempts",
                "submitted_kernels",
                "synchronizations",
                "completed_steps",
                "accepted_world_steps",
                "rejected_world_steps",
                "failed_steps",
                "recovery_sync_attempts",
            ),
            0,
        )
        self._slots = {}
        self._bindings = []
        props, initial_rows, commands = [], [], []
        for key, spec in worlds:
            key = world_key(key)
            if key in self._slots:
                raise ValueError(f"duplicate world key {key}")
            if len(self._bindings) == np.iinfo(np.int32).max:
                raise ValueError("world count exceeds the TYX i32 interface")
            binding, properties, row, command = bind_world(key, spec)
            self._slots[key] = len(self._bindings)
            self._bindings.append(binding)
            props.append(properties)
            initial_rows.append(row)
            commands.append(command)
        self._count = len(self._bindings)
        self._held = np.asarray(commands, np.float32).reshape(self._count, 4)
        self._initial = {
            b.world: b.checkpoint(row, 0, command)
            for b, row, command in zip(self._bindings, initial_rows, commands, strict=True)
        }
        if not self._count:
            return
        import tyx as tx

        from . import _tyx_world_kernels as kernels

        self._tx, self._kernels = tx, kernels
        self._advance_kernels = (
            kernels.advance_worlds,
            kernels.advance_worlds_other_buffer if mode == "cuda" else kernels.advance_worlds,
        )
        self._runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )
        self._resources.callback(self._runtime.close)
        try:
            with tx.use_runtime(self._runtime):
                n = self._count
                self._kind = self._buffer(
                    (n,), np.int32, [b.kind for b in self._bindings], "topology"
                )
                self._props = self._buffer((n, PROPERTY_WIDTH), np.float32, props, "topology")
                self._commands = self._buffer((n, 4), np.float32, self._held, "initial")
                self._states = (
                    self._buffer((n, STATE_WIDTH), np.float32, initial_rows, "initial"),
                    self._buffer((n, STATE_WIDTH), np.float32),
                )
                self._ticks = (
                    self._buffer((n,), np.int32, np.zeros(n, np.int32), "initial"),
                    self._buffer((n,), np.int32),
                )
                self._status = self._buffer((n,), np.int32)
                self._summary = self._buffer((n, 2), np.float32)
                self._row = self._buffer((STATE_WIDTH,), np.float32)
                self._row_tick = self._buffer((1,), np.int32)
                self._restore_state = self._buffer((n, STATE_WIDTH), np.float32)
                self._restore_tick = self._buffer((n,), np.int32)
        except BaseException:
            self.close()
            raise

    def _buffer(self, shape, dtype, initial=None, category=None):
        host = (
            None if initial is None else np.asarray(initial, dtype=dtype, order="C").reshape(shape)
        )
        if self._mode == "interpret":
            return np.empty(shape, dtype=dtype) if host is None else host.copy()
        dev = self._tx.cuda_empty(shape, dtype) if host is None else self._tx.cuda_array(host)
        self._resources.callback(dev.free)
        self._counts["device_allocations"] += 1
        self._counts["device_owned_bytes"] += dev.nbytes
        if host is not None:
            self._counts[category + "_uploaded_bytes"] += host.nbytes
        return dev

    def _upload(self, buffer, data, category):
        if self._mode == "cuda":
            buffer.copy_from(data)
            self._counts[category + "_uploaded_bytes"] += data.nbytes
        else:
            np.copyto(buffer, data)

    def _read(self, buffer, category):
        if self._mode == "interpret":
            return buffer.copy()
        result = buffer.to_numpy()
        self._counts[category + "_downloaded_bytes"] += result.nbytes
        return result

    def _read_status(self):
        result = self._read(self._status, "status")
        if result.shape != (self._count,) or not np.all((result == 0) | (result == 1)):
            raise RuntimeError("TYX returned an invalid world acceptance vector")
        return result

    def _launch(self, kernel, *args):
        self._counts["launch_attempts"] += 1
        kernel(*args, mode=self._mode)
        self._counts["submitted_kernels"] += 1

    def _sync(self):
        self._runtime.sync()
        self._counts["synchronizations"] += 1

    @contextmanager
    def _device(self):
        try:
            with self._tx.use_runtime(self._runtime):
                yield
        except Exception:
            # The accepted buffer remains separate, but the next writer can reuse
            # the standby buffer only after the driver confirms prior work drained.
            self._counts["recovery_sync_attempts"] += 1
            try:
                self._sync()
            except Exception:
                self._unavailable = True
            raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError("resident world batch is closed")
        if self._unavailable:
            raise RuntimeError("device cannot synchronize; rebuild from a saved checkpoint")

    def _keys(self, keys):
        result = list(self._slots) if keys is None else [world_key(k) for k in keys]
        if len(set(result)) != len(result):
            raise ValueError("duplicate selection key")
        for key in result:
            if key not in self._slots:
                raise KeyError(key)
        return result

    def set_controls(self, by_world):
        """Validate all supplied controls, then stage a single upload on the next step."""
        with self._lock:
            self._check_open()
            if not isinstance(by_world, Mapping):
                raise TypeError("controls must be a world mapping")
            keys = self._keys(by_world)
            held = self._held.copy()
            for key in keys:
                i = self._slots[key]
                held[i] = self._bindings[i].packet(by_world[key])
            if not np.array_equal(held, self._held):
                self._held, self._dirty = held, True

    def step(self, h):
        """Advance one positive physical substep and return accepted worlds by key."""
        with self._lock:
            self._check_open()
            dt = float(as_fp32(h, "physical substep", ()))
            if dt <= 0:
                raise ValueError("physical substep must be positive")
            if not self._count:
                return {}
            src, dst = self._active, 1 - self._active
            try:
                with self._device(), np.errstate(all="ignore"):
                    if self._dirty:
                        self._upload(self._commands, self._held, "control")
                        self._dirty = False
                    self._launch(
                        self._advance_kernels[src],
                        self._count,
                        dt,
                        self._kind,
                        self._props,
                        self._commands,
                        self._states[src],
                        self._ticks[src],
                        self._states[dst],
                        self._ticks[dst],
                        self._status,
                    )
                    self._sync()
                    status = self._read_status()
                results = {
                    b.world: bool(value) for b, value in zip(self._bindings, status, strict=True)
                }
            except Exception:
                self._counts["failed_steps"] += 1
                raise
            self._active = dst
            accepted = int(np.count_nonzero(status))
            self._counts["completed_steps"] += 1
            self._counts["accepted_world_steps"] += accepted
            self._counts["rejected_world_steps"] += self._count - accepted
            return results

    def snapshot(self, keys=None):
        """Read selected accepted states and held controls. A subset downloads only those rows."""
        with self._lock:
            self._check_open()
            keys = self._keys(keys)
            if not keys:
                return {}
            state, tick = self._states[self._active], self._ticks[self._active]
            with self._device():
                if len(keys) == self._count:
                    rows, ticks = self._read(state, "snapshot"), self._read(tick, "snapshot")
                    result = {}
                    for key in keys:
                        i = self._slots[key]
                        result[key] = self._bindings[i].checkpoint(rows[i], ticks[i], self._held[i])
                    return result
                result = {}
                for key in keys:
                    i = self._slots[key]
                    self._launch(
                        self._kernels.read_world, i, state, tick, self._row, self._row_tick
                    )
                    self._sync()
                    row = self._read(self._row, "snapshot")
                    t = self._read(self._row_tick, "snapshot")[0]
                    result[key] = self._bindings[i].checkpoint(row, t, self._held[i])
                return result

    def kinematics(self, keys=None):
        """Read owned FP32 position and velocity rows with their world keys, widths and ticks."""
        from ._world_kinematics import read_kinematics

        return read_kinematics(self, keys)

    def summary(self):
        """Read two FP32 values per world: kinetic energy and sphere bottom z or slider q."""
        with self._lock:
            self._check_open()
            if not self._count:
                return {}
            with self._device(), np.errstate(all="ignore"):
                self._launch(
                    self._kernels.observe_worlds,
                    self._count,
                    self._kind,
                    self._props,
                    self._states[self._active],
                    self._summary,
                )
                self._sync()
                summary = self._read(self._summary, "summary")
            if not np.isfinite(summary).all():
                raise FloatingPointError("resident summary is not representable in FP32")
            return {
                b.world: {
                    "kinetic_energy_j": float(v[0]),
                    "q_m" if b.kind else "bottom_z_m": float(v[1]),
                }
                for b, v in zip(self._bindings, summary, strict=True)
            }

    def restore(self, by_world):
        """Replace chosen physical states, ticks, drive histories and their held controls."""
        with self._lock:
            self._check_open()
            if not isinstance(by_world, Mapping):
                raise TypeError("checkpoints must be a world mapping")
            keys = self._keys(by_world)
            if not keys:
                return
            states = np.zeros((self._count, STATE_WIDTH), np.float32)
            ticks = np.full(self._count, -1, np.int32)
            held = self._held.copy()
            for key in keys:
                i = self._slots[key]
                states[i], ticks[i], held[i] = self._bindings[i].unpack(by_world[key])
            src, dst = self._active, 1 - self._active
            with self._device():
                self._upload(self._restore_state, states, "restore")
                self._upload(self._restore_tick, ticks, "restore")
                self._launch(
                    self._kernels.restore_worlds,
                    self._count,
                    self._restore_state,
                    self._restore_tick,
                    self._states[src],
                    self._ticks[src],
                    self._states[dst],
                    self._ticks[dst],
                )
                self._sync()
            self._active = dst
            self._dirty = self._dirty or not np.array_equal(held, self._held)
            self._held = held

    def reset(self, keys=None):
        """Return selected worlds to their initial accepted state and command."""
        with self._lock:
            self._check_open()
            self.restore({key: self._initial[key] for key in self._keys(keys)})

    def stats(self):
        with self._lock:
            return {
                **self._counts,
                "mode": self._mode,
                "world_count": self._count,
                "state_buffer_count": 2 if self._count else 0,
                "current_buffer": self._active,
            }

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._resources.close()
                self._counts["device_owned_bytes"] = 0
