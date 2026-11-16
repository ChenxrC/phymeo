"""Opt-in TYX batch of spheres, scalar joints, fixed and floating complete trees."""

from collections.abc import Mapping
from contextlib import contextmanager
from threading import RLock
from types import MappingProxyType

import numpy as np

from ._resident_affine_tree import AffineRobotTree
from ._resident_cleanup import ReleaseStack, cleanup_failed_init, close_dependencies
from ._resident_floating_affine_tree import FloatingAffineRobotTree, FloatingAffineTreeInputs
from ._resident_floating_tree import FloatingRobotTree, FloatingTreeInputs
from ._resident_revolute_world import RevoluteRotor
from ._resident_tree_world import FixedRobotTree
from ._world_step_layout import (
    PROPERTY_WIDTH,
    STATE_WIDTH,
    FreeSphere,
    PrismaticSlider,
    SpherePlaneCheckpoint,
    SpherePlaneWorld,
    WorldCheckpoint,
    as_fp32,
    bind_world,
    world_key,
)

__all__ = [
    "AffineRobotTree",
    "FreeSphere",
    "FixedRobotTree",
    "FloatingRobotTree",
    "FloatingAffineRobotTree",
    "PrismaticSlider",
    "RevoluteRotor",
    "SpherePlaneCheckpoint",
    "SpherePlaneWorld",
    "TyxWorldBatch",
    "WorldCheckpoint",
    "BatchStepRejected",
]


_BOUNDARY_RUNTIME_QUARANTINE = {}
_BOUNDARY_RUNTIME_QUARANTINE_LOCK = RLock()


class BatchStepRejected(RuntimeError):
    """An all-world substep was rejected; no candidate state was published."""

    def __init__(self, rejected_worlds):
        self.rejected_worlds = tuple(rejected_worlds)
        super().__init__(
            f"resident batch rejected {len(self.rejected_worlds)} world(s): "
            f"{self.rejected_worlds[:8]}; no worlds published"
        )


class TyxWorldBatch:
    """Device passes advance independent worlds using a pair of persistent state buffers.

    Controls stay held until changed. A step returns a compact per-world acceptance map.
    By default rejection retains only the affected world's state, tick and delayed packet.
    require_all=True retains every world's accepted state if any candidate is rejected.
    A failed batch submission leaves its input buffer current. Reading the committed state
    after a device loss requires a functioning driver or a previously saved checkpoint.
    """

    publication_contract_version = 1

    def __init__(self, worlds, mode="cuda", *, cache_root=None, initial_controls=None):
        if mode not in ("cuda", "interpret"):
            raise ValueError("resident mode must be cuda or interpret")
        self._mode = mode
        self._lock = RLock()
        self._closed = False
        self._closing = False
        self._unavailable = False
        self._control_error = None
        self._cleanup_drained = False
        self._resources = ReleaseStack()
        self._host_buffers = []
        self._readback_plans = set()
        self._control_plans = set()
        self._reset_plans = set()
        self._collision_plans = set()
        self._contact_bundles = set()
        self._boundary_plans = set()
        self._boundary_runtime_leases = {}
        self._boundary_unknown = {}
        self._boundary_reading = False
        self._boundary_closing = False
        self._boundary_close_pending = False
        self._collision_owner = object()
        self._topology_generation = 0
        self._state_epoch = 0
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
                "reset_plan_uploaded_bytes",
                "reset_plan_initial_uploaded_bytes",
                "status_downloaded_bytes",
                "summary_downloaded_bytes",
                "snapshot_downloaded_bytes",
                "kinematics_downloaded_bytes",
                "readback_index_uploaded_bytes",
                "readback_plan_owned_bytes",
                "readback_plan_reads",
                "readback_plan_failed_reads",
                "launch_attempts",
                "submitted_kernels",
                "synchronizations",
                "completed_steps",
                "accepted_world_steps",
                "rejected_world_steps",
                "failed_steps",
                "aborted_batch_steps",
                "recovery_sync_attempts",
                "revolute_kernel_launches",
            ),
            0,
        )
        self._slots = {}
        self._bindings = []
        self._trees = None
        self._tree_specs = {}
        self._affine_trees = None
        self._affine_specs = {}
        self._floating_trees = None
        self._floating_specs = {}
        self._floating_affine_trees = None
        self._floating_affine_specs = {}
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
            if binding.tree_spec is not None:
                specs = (
                    self._floating_affine_specs
                    if binding.kind in (11, 12)
                    else self._floating_specs
                    if binding.kind in (9, 10)
                    else self._affine_specs
                    if binding.kind in (7, 8)
                    else self._tree_specs
                )
                specs[key] = binding.tree_spec
            props.append(properties)
            initial_rows.append(row)
            commands.append(command)
        self._count = len(self._bindings)
        self._command_width = max((b.command_width for b in self._bindings), default=4)
        self._property_width = max((len(row) for row in props), default=PROPERTY_WIDTH)
        self._held = np.zeros((self._count, self._command_width), np.float32)
        properties = np.zeros((self._count, self._property_width), np.float32)
        for i, (command, row) in enumerate(zip(commands, props, strict=True)):
            self._held[i, : len(command)] = command
            properties[i, : len(row)] = row
        self._initial = {}
        for binding, row, command in zip(self._bindings, initial_rows, commands, strict=True):
            spec = binding.tree_spec
            if spec is None:
                point = binding.checkpoint(row, 0, command)
            elif binding.kind in (11, 12):
                from ..solvers.floating_tree import FloatingRobotInputs

                control = FloatingAffineTreeInputs(
                    spec.graph_token,
                    spec.initial_drive,
                    FloatingRobotInputs(
                        spec.initial_effort.astype(np.float64),
                        spec.initial_loads[:, :3].astype(np.float64),
                        spec.initial_loads[:, 3:].astype(np.float64),
                    ),
                )
                point = spec.checkpoint(
                    binding.world,
                    spec.initial_root,
                    spec.initial_state,
                    spec.physical.initial_state,
                    spec.initial_generalized,
                    spec.physical.initial_generalized,
                    0,
                    control,
                )
            elif binding.kind in (9, 10):
                from ..solvers.floating_tree import FloatingRobotInputs

                control = FloatingTreeInputs(
                    spec.initial_drive,
                    FloatingRobotInputs(
                        spec.initial_effort.astype(np.float64),
                        spec.initial_loads[:, :3].astype(np.float64),
                        spec.initial_loads[:, 3:].astype(np.float64),
                    ),
                )
                point = spec.checkpoint(
                    binding.world,
                    spec.initial_root,
                    spec.initial_state,
                    spec.initial_generalized,
                    0,
                    control,
                )
            elif binding.kind in (7, 8):
                point = spec.checkpoint(
                    binding.world,
                    spec.initial_state,
                    spec.physical.initial_state,
                    0,
                    spec.initial_drive,
                    spec.initial_loads,
                )
            else:
                point = spec.checkpoint(binding.world, spec.initial_state, 0, spec.initial_command)
            self._initial[binding.world] = point
        if not self._count:
            return
        import tyx as tx

        from . import _tyx_world_kernels as kernels

        self._tx, self._kernels = tx, kernels
        self._advance_kernels = (
            kernels.advance_worlds,
            kernels.advance_worlds_other_buffer if mode == "cuda" else kernels.advance_worlds,
        )
        self._rotor_kernels = None
        if any(b.kind in (3, 4) for b in self._bindings):
            from ._tyx_resident_revolute_kernels import make_revolute_step_kernels

            self._rotor_kernels = make_revolute_step_kernels()
        self._plane_kernels = None
        if any(b.plane_height is not None for b in self._bindings):
            from ._tyx_sphere_plane_kernel import (
                project_sphere_planes,
                project_sphere_planes_other_buffer,
            )

            self._plane_kernels = (
                project_sphere_planes,
                project_sphere_planes_other_buffer if mode == "cuda" else project_sphere_planes,
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
                self._props = self._buffer(
                    (n, self._property_width), np.float32, properties, "topology"
                )
                self._commands = self._buffer(
                    (n, self._command_width), np.float32, self._held, "initial"
                )
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
                if self._tree_specs:
                    from ._resident_tree_storage import TreeDeviceStorage

                    self._trees = TreeDeviceStorage(self, self._tree_specs)
                if self._affine_specs:
                    from ._resident_affine_storage import AffineTreeDeviceStorage

                    self._affine_trees = AffineTreeDeviceStorage(self, self._affine_specs)
                if self._floating_specs:
                    from ._resident_floating_storage import FloatingTreeDeviceStorage

                    self._floating_trees = FloatingTreeDeviceStorage(self, self._floating_specs)
                if self._floating_affine_specs:
                    from ._resident_floating_affine_storage import FloatingAffineTreeDeviceStorage

                    self._floating_affine_trees = FloatingAffineTreeDeviceStorage(
                        self, self._floating_affine_specs
                    )
            if initial_controls is not None:
                self.set_controls(initial_controls)
                # Construction only: freeze the complete authored reset baseline,
                # including held commands. Never recapture it in a hot reset.
                self._initial = self.snapshot()
        except BaseException as error:
            cleanup_failed_init(self, error)
            raise

    def _tree_storages(self):
        """All tree laws share this batch's transaction, never a second owner."""
        return tuple(
            (specs, storage)
            for specs, storage in (
                (self._tree_specs, self._trees),
                (self._affine_specs, self._affine_trees),
                (self._floating_specs, self._floating_trees),
                (self._floating_affine_specs, self._floating_affine_trees),
            )
            if storage is not None
        )

    def _buffer(self, shape, dtype, initial=None, category=None):
        host = (
            None if initial is None else np.asarray(initial, dtype=dtype, order="C").reshape(shape)
        )
        if self._mode == "interpret":
            result = np.empty(shape, dtype=dtype) if host is None else host.copy()
            self._host_buffers.append(result)
            return result
        dev = self._tx.cuda_empty(shape, dtype) if host is None else self._tx.cuda_array(host)
        self._resources.callback(self._free_buffer, dev, dev.nbytes)
        self._counts["device_allocations"] += 1
        self._counts["device_owned_bytes"] += dev.nbytes
        if host is not None:
            self._counts[category + "_uploaded_bytes"] += host.nbytes
        return dev

    def _free_buffer(self, buffer, size):
        buffer.free()
        self._counts["device_owned_bytes"] -= size

    def _host_alias_arrays(self):
        """Current HOST owners, including unselected coordinate families."""
        yield from self._host_buffers
        yield self._held
        for _, storage in self._tree_storages():
            yield from storage._host_controls()
        for plan in self._reset_plans:
            yield from plan._host_arrays()

    def _ensure_cleanup_ready(self):
        # Only a failed recovery sync needs another completion attempt. Normal
        # resident operations already complete before returning to the caller.
        if self._unavailable and not self._cleanup_drained:
            self._sync()
            self._cleanup_drained = True

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

    def _check_open(self, *, allow_boundary_loan=False):
        if self._closed:
            raise RuntimeError("resident world batch is closed")
        if self._boundary_unknown:
            raise RuntimeError("boundary resource UNKNOWN_RELEASE; batch cannot be reused")
        if self._boundary_close_pending:
            raise RuntimeError("boundary cleanup pending; only close may be retried")
        if self._boundary_reading and not allow_boundary_loan:
            raise RuntimeError("resident mutation/reentry during a boundary source loan")
        if self._closing:
            raise RuntimeError("resident world batch is closed or closing")
        if self._control_error is not None:
            raise RuntimeError(
                "resident control rollback failed; close and rebuild"
            ) from self._control_error
        if self._unavailable:
            raise RuntimeError("device cannot synchronize; rebuild from a saved checkpoint")

    def _register_boundary_runtime_lease(self, lease):
        """Retain a query buffer's runtime dependency before its first allocation."""
        with self._lock:
            self._check_open(allow_boundary_loan=True)
            runtime = getattr(self, "_runtime", None)
            if runtime is None or getattr(lease, "runtime", None) is not runtime:
                raise ValueError("boundary lease must borrow this exact batch runtime")
            if any(value is lease for value in self._boundary_runtime_leases.values()):
                raise ValueError("boundary runtime lease is already registered")
            token = object()
            self._boundary_runtime_leases[token] = lease
            return token

    def _release_boundary_runtime_lease(self, token):
        """Called only by a normally completed ledger's no-op runtime close."""
        with self._lock:
            if token in self._boundary_unknown:
                raise RuntimeError("boundary UNKNOWN_RELEASE token cannot be unregistered")
            if token not in self._boundary_runtime_leases:
                raise ValueError("unknown boundary runtime lease token")
            del self._boundary_runtime_leases[token]

    def _quarantine_boundary_runtime_lease(self, token, ledger):
        with self._lock:
            if token not in self._boundary_runtime_leases:
                raise ValueError("unknown boundary runtime lease token")
            if ledger.state != "UNKNOWN_RELEASE":
                raise ValueError("only UNKNOWN_RELEASE ledgers may quarantine the runtime")
            previous = self._boundary_unknown.get(token)
            if previous is not None and previous is not ledger:
                raise ValueError("boundary token already belongs to another unknown ledger")
            self._boundary_unknown[token] = ledger
            self._unavailable = True
            with _BOUNDARY_RUNTIME_QUARANTINE_LOCK:
                _BOUNDARY_RUNTIME_QUARANTINE[id(self)] = self

    @contextmanager
    def _borrow_boundary_collision_state(self, *, owner=None, topology_generation=None):
        """A read-only accepted loan also rejects same-thread producer reentry."""
        with self._lock:
            self._check_open()
            with self._borrow_collision_state(
                owner=owner, topology_generation=topology_generation
            ) as accepted:
                self._boundary_reading = True
                try:
                    yield accepted
                finally:
                    self._boundary_reading = False

    def _close_boundary_resources(self):
        """Keep the complete parent stack until every dependent lease is closed.

        A failure before ledger entry may be retried. An opaque free failure is
        sticky and its DeviceResources ledger never invokes that callback again.
        Independent plans are still visited; the parent runtime is not a sibling.
        """
        if self._boundary_closing:
            raise RuntimeError("recursive boundary cleanup")
        self._boundary_closing = True
        self._boundary_close_pending = True
        errors = []
        try:
            for plan in tuple(self._boundary_plans):
                try:
                    plan.close()
                except BaseException as error:
                    errors.append(error)
            if errors:
                raise BaseExceptionGroup("boundary cleanup incomplete; parent retained", errors)
            if self._boundary_unknown:
                errors = [
                    failure.error
                    for ledger in self._boundary_unknown.values()
                    for failure in ledger.failures
                ]
                raise BaseExceptionGroup("boundary UNKNOWN_RELEASE; parent retained", errors)
            if self._boundary_runtime_leases:
                raise RuntimeError("boundary runtime leases pending; parent retained")
            self._boundary_plans.clear()
            self._boundary_close_pending = False
        finally:
            self._boundary_closing = False

    @contextmanager
    def _borrow_collision_state(self, *, owner=None, topology_generation=None):
        """Loan one accepted buffer under the existing lock and device transaction."""
        from ._resident_collision_plan import AcceptedCollisionState

        with self._lock:
            self._check_open()
            if owner is not None and owner is not self._collision_owner:
                raise ValueError("collision view belongs to a different owner")
            if topology_generation is not None and topology_generation != self._topology_generation:
                raise ValueError("collision topology generation changed")
            common = (
                self._collision_owner,
                self._topology_generation,
                self._state_epoch,
                self._active,
                tuple(self._slots),
                tuple(self._bindings),
                MappingProxyType(self._slots),
            )
            if not self._count:
                yield AcceptedCollisionState(*common, None, None, None, None, self._mode, None)
                return
            with self._device():
                yield AcceptedCollisionState(
                    *common,
                    self._states[self._active],
                    self._ticks[self._active],
                    self._props,
                    self._runtime,
                    self._mode,
                    self._tx,
                    tree=None if self._trees is None else self._trees.borrow_geometry(self._active),
                    affine_tree=(
                        None
                        if self._affine_trees is None
                        else self._affine_trees.borrow_geometry(self._active)
                    ),
                    floating_tree=(
                        None
                        if self._floating_trees is None
                        else self._floating_trees.borrow_geometry(self._active)
                    ),
                    floating_affine_tree=(
                        None
                        if self._floating_affine_trees is None
                        else self._floating_affine_trees.borrow_geometry(self._active)
                    ),
                )

    def prepare_collision(
        self, model, bindings, *, candidate_capacity=4096, tree_dynamic_pairs=False
    ):
        """Bind accepted finite geometry, optionally including nonadjacent tree groups."""
        from ._resident_collision_plan import ResidentCollisionPlan

        return ResidentCollisionPlan(
            self,
            model,
            bindings,
            candidate_capacity=candidate_capacity,
            tree_dynamic_pairs=tree_dynamic_pairs,
        )

    def collision(self, plan):
        """Query a collision plan owned by this exact resident batch."""
        from ._resident_collision_plan import collision_for

        return collision_for(self, plan)

    def prepare_convex_query(
        self,
        broad,
        *,
        pairs=None,
        pair_capacity=None,
        gjk_iterations=64,
        epa_iterations=192,
        epa_vertex_capacity=64,
        epa_face_capacity=128,
        memory_limit_bytes=67108864,
        site_capacity=None,
    ):
        """Prepare bounded device distances and bilateral material witnesses."""
        from ._device_convex_query import ConvexQueryPlan

        return ConvexQueryPlan(
            self,
            broad,
            pairs=pairs,
            pair_capacity=pair_capacity,
            gjk_iterations=gjk_iterations,
            epa_iterations=epa_iterations,
            epa_vertex_capacity=epa_vertex_capacity,
            epa_face_capacity=epa_face_capacity,
            memory_limit_bytes=memory_limit_bytes,
            site_capacity=site_capacity,
        )

    def convex_query(self, plan):
        """Query only a plan owned by this exact batch, without a physical step."""
        from ._device_convex_query import convex_query_for

        return convex_query_for(self, plan)

    def prepare_boundary_query(self, model, bindings, *, limits, memory_limit_bytes):
        """Prepare finite boundary distances from this batch's accepted device state.

        Limits and memory capacity are explicit. The plan borrows this runtime;
        query does not advance physics, resolve contact forces, or certify CCD.
        """
        from ._resident_boundary_query import ResidentBoundaryQueryPlan

        return ResidentBoundaryQueryPlan(
            self, model, bindings, limits=limits, memory_limit_bytes=memory_limit_bytes
        )

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
            tree_stages = [
                (
                    storage,
                    storage.validate_controls({key: by_world[key] for key in keys if key in specs}),
                )
                for specs, storage in self._tree_storages()
            ]
            for key in keys:
                if self._bindings[self._slots[key]].tree_spec is not None:
                    continue
                i = self._slots[key]
                packet = self._bindings[i].packet(by_world[key])
                held[i].fill(0)
                held[i, : len(packet)] = packet
            if not np.array_equal(held, self._held):
                self._held, self._dirty = held, True
            for storage, stage in tree_stages:
                storage.commit_controls(stage)

    def prepare_control_write(self, keys=None, *, fields):
        """Prepare selected HOST command fields; execution uploads on the next step."""
        from ._resident_control_plan import ResidentControlPlan

        return ResidentControlPlan(self, keys, fields)

    def prepare_contacts(
        self,
        model,
        bindings,
        materials_by_pair,
        *,
        candidate_capacity=4096,
        site_capacity=128,
        sampling="discrete_entry",
        tree_dynamic_pairs=False,
        narrow_phase="host_patch",
        device_patch_site_capacity=None,
    ):
        """Bind finite compliant/velocity-friction contact to accepted full tree state."""
        from ._resident_contact_plan import ResidentContactPlan

        return ResidentContactPlan(
            self,
            model,
            bindings,
            materials_by_pair,
            candidate_capacity=candidate_capacity,
            site_capacity=site_capacity,
            sampling=sampling,
            tree_dynamic_pairs=tree_dynamic_pairs,
            narrow_phase=narrow_phase,
            device_patch_site_capacity=device_patch_site_capacity,
        )

    def prepare_contact_bundle(
        self,
        pages,
        *,
        site_capacity=128,
        sampling="discrete_entry",
        tree_dynamic_pairs=False,
        narrow_phase="host_patch",
        device_patch_site_capacity=None,
    ):
        """Register whole-world pages for one step, with an optional device query per page."""
        from ._resident_contact_bundle import ResidentContactBundle

        return ResidentContactBundle(
            self,
            pages,
            site_capacity=site_capacity,
            sampling=sampling,
            tree_dynamic_pairs=tree_dynamic_pairs,
            narrow_phase=narrow_phase,
            device_patch_site_capacity=device_patch_site_capacity,
        )

    def step_contacts(self, plan, h, *, require_all=False):
        """Sample entry contact, solve full mechanics and publish one transaction."""
        from ._resident_contact_bundle import ResidentContactBundle
        from ._resident_contact_plan import ResidentContactPlan

        if not isinstance(plan, (ResidentContactPlan, ResidentContactBundle)):
            raise TypeError("expected a ResidentContactPlan or ResidentContactBundle")
        if plan._batch is not self:
            raise ValueError("contact plan belongs to a different owner")
        return self._step(h, contacts=plan, require_all=require_all)

    def step(self, h, *, require_all=False):
        """Advance one substep; optionally publish only if all worlds accept it.

        The default preserves per-world acceptance. With require_all=True, a
        rejected candidate raises BatchStepRejected without swapping any bank.
        Explicitly staged controls stay held; no state checkpoint is downloaded.
        """
        return self._step(h, require_all=require_all)

    def _step(self, h, *, contacts=None, require_all=False):
        with self._lock:
            self._check_open()
            if type(require_all) is not bool:
                raise TypeError("require_all must be bool")
            if contacts is not None:
                contacts._check_open()
            dt = float(as_fp32(h, "physical substep", ()))
            if dt <= 0:
                raise ValueError("physical substep must be positive")
            if not self._count:
                return {}
            src, dst = self._active, 1 - self._active
            for _, storage in self._tree_storages():
                storage.invalidate_diagnostics()
            try:
                with self._device(), np.errstate(all="ignore"):
                    if self._dirty:
                        self._upload(self._commands, self._held, "control")
                    for _, storage in self._tree_storages():
                        storage.upload_controls()
                    # Clear dirty only after every command bank was uploaded.
                    self._dirty = False
                    for _, storage in self._tree_storages():
                        storage.controls_uploaded()
                    attempt = None if contacts is None else contacts._begin_attempt(dt, src, dst)
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
                    if self._rotor_kernels is not None:
                        self._launch(
                            self._rotor_kernels[src],
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
                        self._counts["revolute_kernel_launches"] += 1
                    for _, storage in self._tree_storages():
                        if contacts is not None and contacts._handles(storage):
                            contacts._stage_tree(storage, attempt, dt, src, dst)
                        else:
                            storage.stage_advance(
                                dt, src, dst, self._ticks[src], self._ticks[dst], self._status
                            )
                    if contacts is not None:
                        finalize_trees = getattr(contacts, "_finalize_trees", None)
                        if finalize_trees is not None:
                            finalize_trees(attempt, dt, src, dst)
                    if self._plane_kernels is not None:
                        self._launch(
                            self._plane_kernels[src],
                            self._count,
                            self._kind,
                            self._props,
                            self._states[src],
                            self._ticks[src],
                            self._states[dst],
                            self._ticks[dst],
                            self._status,
                        )
                    self._sync()
                    status = self._read_status()
                    for _, storage in self._tree_storages():
                        storage.check_advance(status)
                    if contacts is not None:
                        candidate = contacts._read_candidate(attempt, dst)
                        contact_commit = contacts._finish_attempt(attempt, candidate, status)
                        contacts._check_attempt(attempt)
                # Completion has already succeeded. Reject outside _device so
                # ordinary numerical rejection adds no recovery synchronization.
                if require_all and not np.all(status):
                    self._counts["aborted_batch_steps"] += 1
                    raise BatchStepRejected(
                        b.world
                        for b, value in zip(self._bindings, status, strict=True)
                        if not value
                    )
                results = {
                    b.world: bool(value) for b, value in zip(self._bindings, status, strict=True)
                }
            except Exception:
                for _, storage in self._tree_storages():
                    storage.invalidate_diagnostics()
                self._counts["failed_steps"] += 1
                raise
            self._active = dst
            self._state_epoch += 1
            for _, storage in self._tree_storages():
                storage.finish_advance(status)
            if contacts is not None:
                contacts._publish(contact_commit)
            accepted = int(np.count_nonzero(status))
            self._counts["completed_steps"] += 1
            self._counts["accepted_world_steps"] += accepted
            self._counts["rejected_world_steps"] += self._count - accepted
            return results

    def snapshot(self, keys=None):
        """Read complete accepted states; tree bank readback is counted separately."""
        with self._lock:
            self._check_open()
            keys = self._keys(keys)
            if not keys:
                return {}
            state, tick = self._states[self._active], self._ticks[self._active]
            tree_keys = [
                key for key in keys if self._bindings[self._slots[key]].tree_spec is not None
            ]
            if tree_keys:
                result = {}
                for specs, storage in self._tree_storages():
                    selected = [key for key in tree_keys if key in specs]
                    if selected:
                        result.update(storage.snapshot(selected, self._active, tick))
                legacy = [key for key in keys if key not in tree_keys]
                if legacy:
                    result.update(self.snapshot(legacy))
                return {key: result[key] for key in keys}
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

    def _selected_tree_keys(self, keys):
        selected = (
            [b.world for b in self._bindings if b.tree_spec is not None]
            if keys is None
            else self._keys(keys)
        )
        if any(self._bindings[self._slots[key]].tree_spec is None for key in selected):
            raise ValueError("tree observation requires tree world keys")
        return selected

    def tree_kinematics(self, keys=None):
        """Own complete accepted mixed R/P coordinates and FK at the same tick."""
        with self._lock:
            self._check_open()
            selected = self._selected_tree_keys(keys)
            result = {}
            for specs, storage in self._tree_storages():
                picked = [key for key in selected if key in specs]
                if picked:
                    result.update(
                        storage.kinematics(picked, self._active, self._ticks[self._active])
                    )
            return {key: result[key] for key in selected}

    def tree_diagnostics(self, keys=None):
        """Return detached entry-time dynamics for the latest accepted attempt."""
        with self._lock:
            self._check_open()
            selected = self._selected_tree_keys(keys)
            result = {}
            for specs, storage in self._tree_storages():
                picked = [key for key in selected if key in specs]
                if picked:
                    result.update(storage.diagnostics(picked))
            return {key: result[key] for key in selected}

    def prepare_readback(self, keys, *, kind="kinematics"):
        """Prepare reusable selected-world observation buffers; existing reads stay unchanged."""
        from ._world_readback_plan import WorldReadbackPlan

        return WorldReadbackPlan(self, keys, kind=kind)

    def prepare_field_read(self, keys=None, *, fields=("position", "velocity")):
        """Freeze a homogeneous physical field layout for direct caller HOST output."""
        from ._resident_field_read import ResidentFieldReadPlan

        return ResidentFieldReadPlan(self, keys, fields)

    def prepare_body_read(self, keys=None, *, fields=("position", "rotation")):
        """Prepare accepted-endpoint body observations into caller-owned HOST arrays."""
        from ._resident_body_read import ResidentBodyReadPlan

        return ResidentBodyReadPlan(self, keys, fields)

    def readback(self, plan):
        """Read an owned frame from a plan bound to this exact batch instance."""
        from ._world_readback_plan import readback_for

        return readback_for(self, plan)

    def summary(self):
        """Read energy and sphere bottom z, slider q_m or revolute q_rad."""
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
            result = {
                b.world: {
                    "kinetic_energy_j": float(v[0]),
                    ("q_rad" if b.kind in (3, 4) else "q_m" if b.kind else "bottom_z_m"): float(
                        v[1]
                    ),
                }
                for b, v in zip(self._bindings, summary, strict=True)
                if b.tree_spec is None
            }
            for specs, storage in self._tree_storages():
                result.update(storage.summary(list(specs), self._active))
            return {key: result[key] for key in self._slots}

    def restore(self, by_world):
        """Replace chosen physical states, ticks, drive histories and their held controls."""
        return self._restore(by_world)

    def _restore(self, by_world, *, before_commit=None):
        """Internal composition hook runs only after all staged restore checks."""
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
            tree_stages = [
                (
                    storage,
                    storage.validate_restore({key: by_world[key] for key in keys if key in specs}),
                )
                for specs, storage in self._tree_storages()
            ]
            for key in keys:
                if self._bindings[self._slots[key]].tree_spec is not None:
                    continue
                i = self._slots[key]
                states[i], ticks[i], control = self._bindings[i].unpack(by_world[key])
                held[i].fill(0)
                held[i, : len(control)] = control
            src, dst = self._active, 1 - self._active
            try:
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
                    for storage, stage in tree_stages:
                        storage.stage_restore(stage, src, dst, self._ticks[src], self._ticks[dst])
                    self._sync()
                    for _, storage in self._tree_storages():
                        storage.check_restore()
                    if before_commit is not None:
                        before_commit()
            except Exception:
                for _, storage in self._tree_storages():
                    storage.invalidate_diagnostics()
                raise
            self._active = dst
            self._state_epoch += 1
            self._dirty = self._dirty or not np.array_equal(held, self._held)
            self._held = held
            for storage, stage in tree_stages:
                storage.commit_restore(stage)

    def reset(self, keys=None):
        """Return selected worlds to their initial accepted state and command."""
        with self._lock:
            self._check_open()
            self.restore({key: self._initial[key] for key in self._keys(keys)})

    def prepare_reset(self):
        """Freeze initial states for repeated HOST-mask reset without snapshots."""
        from ._resident_reset_plan import ResidentResetPlan

        return ResidentResetPlan(self)

    def stats(self):
        with self._lock:
            convex_queries = tuple(
                query for broad in self._collision_plans for query in broad._convex_dependents
            )
            return {
                **self._counts,
                "publication_contract_version": self.publication_contract_version,
                "closed": self._closed,
                "closing": self._closing,
                "mode": self._mode,
                "world_count": self._count,
                "state_buffer_count": 2 if self._count else 0,
                "current_buffer": self._active,
                "readback_plan_live_count": len(self._readback_plans),
                "control_plan_live_count": len(self._control_plans),
                "reset_plan_live_count": len(self._reset_plans),
                "collision_plan_live_count": len(self._collision_plans),
                "contact_bundle_live_count": len(self._contact_bundles),
                "convex_query_plan_live_count": len(convex_queries),
                "convex_query_live_bytes": sum(
                    int(query.stats["live_bytes"]) for query in convex_queries
                ),
                "state_epoch": self._state_epoch,
                "tree_world_count": len(self._tree_specs),
                "affine_tree_world_count": len(self._affine_specs),
                "floating_world_count": len(self._floating_specs),
                "floating_affine_world_count": len(self._floating_affine_specs),
                "floating_affine_tree_status_codes": (
                    {}
                    if self._floating_affine_trees is None
                    else dict(self._floating_affine_trees.status_codes)
                ),
                "floating_tree_status_codes": (
                    {} if self._floating_trees is None else dict(self._floating_trees.status_codes)
                ),
                "affine_tree_status_codes": (
                    {} if self._affine_trees is None else dict(self._affine_trees.status_codes)
                ),
                "tree_status_codes": {} if self._trees is None else dict(self._trees.status_codes),
            }

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._boundary_reading:
                raise RuntimeError("cannot close during a boundary source loan")
            self._close_boundary_resources()
            self._closing = True
            self._ensure_cleanup_ready()
            close_dependencies(self)
            # Bundles depend on pages; pages depend on queries and batch storage.
            for owners in (
                self._contact_bundles,
                self._collision_plans,
                self._readback_plans,
                self._control_plans,
                self._reset_plans,
            ):
                for plan in tuple(owners):
                    plan.close()
            self._resources.close()
            self._host_buffers.clear()
            self._counts["device_owned_bytes"] = 0
            self._closed = True
            self._closing = False
