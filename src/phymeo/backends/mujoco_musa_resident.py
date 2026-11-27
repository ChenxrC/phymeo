"""Exclusive device-resident training on the admitted MuJoCo TYX MUSA solvers."""

import numpy as np

from ..training.resident_stepper import PhysicsStepFailure


class _BankView:
    """A generation-bound borrow. Tensor transfers copy; no solver pointer escapes."""

    def __init__(self, session, buffer, *, writable=False):
        self._session = session
        self._buffer = buffer
        self._generation = session._generation
        self._writable = writable
        self.shape, self.dtype = buffer.shape, buffer.dtype

    def _live(self):
        self._session._check()
        if self._generation != self._session._generation:
            raise RuntimeError("MUSA resident view expired; request the current buffers")
        return self._buffer

    def copy_to_torch(self, out, *, stream=None):
        with self._session._owner._lock:
            return self._live().copy_to_torch(out, stream=stream)

    def to_torch(self, *, stream=None):
        with self._session._owner._lock:
            return self._live().to_torch(stream=stream)

    def to_numpy(self):
        """Explicit host observation for auditing. Counted separately from solver status."""
        with self._session._owner._lock:
            value = self._live().to_numpy()
            self._session._explicit_readback += value.nbytes
            return value

    def copy_from_torch(self, tensor, *, stream=None):
        with self._session._owner._lock:
            buffer = self._live()
            if not self._writable:
                raise TypeError("only the resident ctrl buffer is writable")
            return buffer.copy_from_torch(tensor, stream=stream)


class MusaResidentSession:
    """One model, fixed ordered world rows, native FP32 coordinates and motors.

    ``buffers()`` gives qpos [world,max(8,nq)], qvel [world,max(8,nv)], ctrl
    [world,nu], and time [world,2]. Time is the compensated pair sum-error.
    Only the native nq/nv columns are state. After a successful step, tree
    solvers also expose ``entry_bias`` [world,max(8,nv)] at the *input* pose.
    Each step/reset invalidates all previous views. Callers may retain owned
    Torch copies and may hold a control for multiple steps. No external loads,
    per-step FK mirrors or Torch autograd through physics.

    The solver synchronizes before each all-world commit. The automatic D2H
    is one int32 status per world. Reset copies the saved session-entry pose,
    velocity, compensated clock and zero motor command wholly on device.
    """

    def __init__(self, owner):
        self._owner = owner
        self._device = owner._device
        self.world_keys = owner.world_keys
        self.nq, self.nv, self.nu = map(int, (owner._model.nq, owner._model.nv, owner._model.nu))
        self.timestep = owner.timestep
        self._generation = self._steps = self._resets = self._failures = 0
        self._status_readback = self._explicit_readback = 0
        self._ticks = 0
        self._bias_valid = False

    def _check(self):
        if getattr(self._owner, "_musa_resident_owner", None) is not self:
            raise RuntimeError("MUSA resident session is no longer active")

    def _start(self):
        with self._owner._lock:
            self._check()
            device = self._device
            from .mujoco_tyx._control_kernels import check_resident_ctrl

            self._check_ctrl = check_resident_ctrl
            self._applied = device.applied_input if hasattr(device, "applied_input") else device.applied
            for target in (device.controls, self._applied, device.body_force):
                target.copy_from(np.zeros(target.shape, target.dtype))
            # These four training-workspace allocations belong to the same
            # MusaRuntime and are released by it with the solver. The solver's
            # physical array manifest excludes external task workspaces.
            self._initial = []
            for source in (device.q[device.active], device.v[device.active],
                           device.time[device.active], device.controls):
                backup = device.runtime.empty(source.shape, source.dtype)
                backup.copy_from_device(source)
                self._initial.append(backup)

    def buffers(self):
        with self._owner._lock:
            self._check()
            d = self._device
            views = dict(qpos=_BankView(self, d.q[d.active]), qvel=_BankView(self, d.v[d.active]),
                         time=_BankView(self, d.time[d.active]), ctrl=_BankView(self, d.controls, writable=True))
            if self._bias_valid and hasattr(d, "bias"):
                views["entry_bias"] = _BankView(self, d.bias)
            return views

    def contact_buffers(self):
        """Read the most recently accepted mixed-constraint solve at its input pose.

        The leading count masks inactive contact rows. The map selects the
        compiled slot supplying the geom IDs and frame. Wrenches act on geom2
        in that frame. Device views expire on the next step or reset.
        """
        with self._owner._lock:
            self._check()
            if not self._bias_valid:
                raise RuntimeError("resident contact observations require an accepted step")
            d = self._device
            if not hasattr(d, "contact_wrench") or getattr(d.plan, "area_friction_laws", ()):
                raise NotImplementedError("resident contact observations require native mixed constraint rows")
            if not hasattr(self, "_contact_slots"):
                ids = np.asarray(d.plan.slot_ids[:d.plan.nslot], dtype=np.int32)
                self._contact_slots = np.frombuffer(ids.tobytes(), dtype=np.int32).reshape(ids.shape)
            return dict(counts=_BankView(self, d.constraint_counts), contact_map=_BankView(self, d.contact_map),
                        contact_wrench=_BankView(self, d.contact_wrench), normal_rows=_BankView(self, d.normal_rows),
                        slot_ids=self._contact_slots)

    def step(self):
        """Attempt one compiled physical timestep for every world, then publish atomically."""
        with self._owner._lock:
            self._check()
            if self._ticks >= np.iinfo(np.int32).max:
                raise OverflowError("resident logical tick exceeds i32")
            self._generation += 1
            self._bias_valid = False
            d = self._device
            try:
                # Pass the actual resident control/input banks. Their identity
                # suppresses staging only on MUSA, across admitted solver types.
                d.advance(d.controls, self._applied, d.body_force)
                d.launch_kernel(self._check_ctrl, d.nworld, self.nu, d.controls, d.status)
                status = d._get(d.status)
                d.counts["completed_kernels"] += 1
                self._status_readback += status.nbytes
                if np.any(status):
                    rejected = {key: int(code) for key, code in zip(self.world_keys, status) if code}
                    raise ValueError(f"per-world TYX solver status: {rejected}")
            except Exception as error:
                self._failures += 1
                raise PhysicsStepFailure(
                    f"MUSA resident candidate rejected before publication: {error}",
                    accepted_worlds=dict.fromkeys(self.world_keys, False),
                ) from error
            d.publish()
            self._steps += 1
            self._ticks += 1
            self._bias_valid = True
            return self.buffers()

    def reset(self):
        """Restore all rows to their session-entry states, keeping the compiled owner."""
        with self._owner._lock:
            self._check()
            self._generation += 1
            self._bias_valid = False
            d = self._device
            target = 1 - d.active
            destinations = (d.q[target], d.v[target], d.time[target], d.controls)
            for destination, source in zip(destinations, self._initial, strict=True):
                destination.copy_from_device(source)
            d.publish()
            self._ticks = 0
            self._resets += 1
            return self.buffers()

    def report(self):
        """Cumulative hardware transport counters and session-specific readback accounting."""
        with self._owner._lock:
            self._check()
            report = self._device.runtime.report()
            report.update(world_keys=list(self.world_keys), active_bank=self._device.active,
                          resident_steps=self._steps, accepted_world_steps=self._steps*len(self.world_keys),
                          failed_batches=self._failures, resident_resets=self._resets,
                          episode_steps=self._ticks, status_downloaded_bytes=self._status_readback,
                          explicit_readback_bytes=self._explicit_readback,
                          reset_storage_bytes=sum(buffer.nbytes for buffer in self._initial))
            return report
