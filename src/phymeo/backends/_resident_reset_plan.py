"""Reusable initial-state reset with masked control and accepted-bank publication."""

from types import MappingProxyType

import numpy as np

from ._resident_cleanup import ReleaseStack, cleanup_failed_init, close_dependencies


class ResidentResetPlan:
    """One batch's frozen initial states, not a snapshot of its current physics.

    Dense HOST bool selection follows layout.world_keys. CUDA execution uploads
    only one int32 tick/mask vector and copies to the existing candidate bank.
    """

    contract_version = 1

    def __init__(self, batch):
        self._batch = batch
        self._closed = self._closing = False
        self._resources = ReleaseStack()
        self._arrays = []
        self._controls = []
        self._device_bytes = self._allocations = 0
        self._resets = self._failures = self._uploads = self._uploaded_bytes = 0
        self._syncs = 0
        self._metadata_bank = 0
        self._floating = None
        with batch._lock:
            batch._check_open()
            if batch._affine_specs or batch._floating_affine_specs:
                raise NotImplementedError(
                    "prepared reset requires scalar worlds or ordinary fixed/floating trees"
                )
            self._tree = batch._trees
            n = batch._count
            self._mask = np.zeros(n, bool)
            self._ticks_host = np.full(n, -1, np.int32)
            self._initial_ticks = np.zeros(n, np.int32)
            self._tree_rows = np.array(
                [batch._slots[w] for w in self._tree.worlds] if self._tree else [], np.intp
            )
            self._tree_mask = np.zeros(len(self._tree_rows), bool)
            self._layout = MappingProxyType(
                dict(
                    contract_version=1,
                    world_keys=tuple(batch._slots),
                    mask_shape=(n,),
                    mask_dtype="bool",
                    completion="synchronous_host",
                    publication="all_selected_or_none",
                    source="construction_initial_state_and_controls",
                    device_upload="one_int32_mask_vector",
                )
            )
            batch._reset_plans.add(self)
            try:
                if n:
                    self._prepare()
            except BaseException as error:
                cleanup_failed_init(self, error)
                raise

    @property
    def layout(self):
        return self._layout

    def _control(self, owner, initial, mask, attribute="_held"):
        self._controls.append(
            (owner, attribute, initial, np.empty_like(initial), np.empty(initial.shape, bool), mask)
        )

    def _prepare(self):
        from ._tyx_resident_reset_kernels import make_reset_kernels

        batch, tree = self._batch, self._tree
        initial = np.zeros((batch._count, 12), np.float32)
        held = np.zeros_like(batch._held)
        for key, slot in batch._slots.items():
            if key in batch._tree_specs or key in batch._floating_specs:
                self._initial_ticks[slot] = batch._initial[key].tick
            else:
                initial[slot], self._initial_ticks[slot], command = batch._bindings[slot].unpack(
                    batch._initial[key]
                )
                held[slot, : len(command)] = command
        self._control(batch, held, self._mask[:, None])
        self._scalar_kernels, self._tree_kernels = make_reset_kernels()
        with batch._device():
            self._initial = self._allocate(initial)
            self._selected_ticks = self._allocate(self._ticks_host)
            if tree:
                state = np.zeros((tree.count, tree.nmax, 7), np.float32)
                held = np.zeros_like(tree._held)
                for key, row in tree.rows.items():
                    spec = tree.specs[key]
                    state[row, : spec.n], held[row, : spec.n], _ = spec.restore_row(
                        batch._initial[key], key
                    )
                self._control(tree, held, self._tree_mask[:, None, None])
                self._tree_initial = self._allocate(state)
                self._tree_fk = self._allocate(np.zeros((tree.count, tree.nmax, 18), np.float32))
                self._tree_energy = self._allocate(np.zeros(tree.count, np.float32))
                # Only scratch frame/flags/spatial are borrowed. Accepted geometry,
                # diagnostic mass/Jacobians/vector/info and state are not overwritten.
                batch._launch(
                    tree.kernels["geometry"][0],
                    tree.count,
                    tree.meta,
                    tree.parents,
                    tree.kinds,
                    tree.static,
                    tree.zero,
                    self._tree_initial,
                    self._tree_fk,
                    tree.frame,
                    tree.flags,
                )
                batch._launch(
                    tree.kernels["energy"][0],
                    tree.count,
                    tree.meta,
                    tree.parents,
                    tree.kinds,
                    tree.static,
                    self._tree_initial,
                    self._tree_fk,
                    tree.spatial,
                    self._tree_energy,
                    tree.flags,
                )
                batch._counts["tree_fk_kernel_launches"] += 1
                batch._counts["tree_energy_kernel_launches"] += 1
                self._validity = [{w: False for w in tree.worlds} for _ in (0, 1)]
                self._codes = [{w: "RESTORED" for w in tree.worlds} for _ in (0, 1)]
                self._code_views = tuple(MappingProxyType(c) for c in self._codes)
            if batch._floating_trees is not None:
                from ._resident_floating_reset import FloatingResetBinding

                self._floating = FloatingResetBinding(batch._floating_trees, self._allocate)
                for attribute, initial_control, mask in self._floating.controls:
                    self._control(batch._floating_trees, initial_control, mask, attribute)
            # Warm both argument directions into standby only; accepted banks stay
            # unchanged. A discarded standby bank is not an authoritative state.
            active, standby = batch._active, 1 - batch._active
            for bank in (0, 1):
                self._launch(bank, active, standby)
            batch._sync()
            if tree:
                tree.check_restore()
            if self._floating is not None:
                self._floating.storage.check_restore()

    def _allocate(self, initial):
        batch = self._batch
        if batch._mode == "interpret":
            array = initial.copy()
        else:
            array = batch._tx.cuda_array(initial)
            self._resources.callback(self._free, array, array.nbytes)
            self._device_bytes += array.nbytes
            self._allocations += 1
            batch._counts["device_allocations"] += 1
            batch._counts["device_owned_bytes"] += array.nbytes
            batch._counts["reset_plan_initial_uploaded_bytes"] += array.nbytes
        self._arrays.append(array)
        return array

    def _free(self, array, size):
        array.free()
        self._device_bytes -= size
        self._batch._counts["device_owned_bytes"] -= size

    def _check_open(self):
        if self._closed or self._closing:
            raise RuntimeError("resident reset plan is closed or closing")
        self._batch._check_open()

    def _launch(self, kernel_bank, src, dst):
        batch, tree = self._batch, self._tree
        batch._launch(
            self._scalar_kernels[kernel_bank],
            batch._count,
            self._initial,
            self._selected_ticks,
            batch._states[src],
            batch._ticks[src],
            batch._states[dst],
            batch._ticks[dst],
        )
        if tree:
            batch._launch(
                self._tree_kernels[kernel_bank],
                tree.count,
                tree.meta,
                self._selected_ticks,
                self._tree_initial,
                self._tree_fk,
                self._tree_energy,
                tree.states[src],
                tree.fk[src],
                tree.energy[src],
                tree.states[dst],
                tree.fk[dst],
                tree.energy[dst],
            )
        if self._floating is not None:
            self._floating.launch(kernel_bank, self._selected_ticks, src, dst)

    def _host_arrays(self):
        yield self._mask
        yield self._ticks_host
        yield self._initial_ticks
        yield self._tree_rows
        yield self._tree_mask
        if self._floating is not None:
            yield self._floating.rows
            yield self._floating.mask
        for _, _, initial, previous, compare, _ in self._controls:
            yield initial
            yield previous
            yield compare
        if self._batch._mode == "interpret":
            yield from self._arrays

    def _select(self, mask):
        if mask is None:
            self._mask.fill(True)
        else:
            if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
                raise TypeError("reset mask requires a bool NumPy array")
            if mask.shape != self._mask.shape or not mask.flags.c_contiguous:
                raise ValueError("reset mask requires the C-contiguous layout shape")
            if any(
                np.shares_memory(mask, a)
                for a in (*self._batch._host_alias_arrays(), *self._host_arrays())
            ):
                raise ValueError("reset mask must not alias resident storage")
            np.copyto(self._mask, mask)
        self._ticks_host.fill(-1)
        np.copyto(self._ticks_host, self._initial_ticks, where=self._mask)
        np.take(self._mask, self._tree_rows, out=self._tree_mask, mode="clip")
        if self._floating is not None:
            self._floating.select(self._mask)

    def reset(self, mask=None):
        """Reset all or selected initial worlds; empty selection does not publish."""
        batch, tree = self._batch, self._tree
        with batch._lock:
            self._check_open()
            self._select(mask)
            if not self._mask.any():
                return batch._state_epoch
            changed = []
            for owner, attribute, initial, previous, compare, selected in self._controls:
                current = getattr(owner, attribute)
                if (
                    current.shape != initial.shape
                    or current.dtype != np.float32
                    or not current.flags.c_contiguous
                    or not current.flags.writeable
                ):
                    raise RuntimeError("reset control owner changed layout; rebuild plan")
                np.equal(current, initial, out=compare)
                np.logical_not(compare, out=compare)
                np.logical_and(compare, selected, out=compare)
                changed.append(bool(compare.any()))
                np.copyto(previous, current, where=selected)
            metadata = 1 - self._metadata_bank
            if tree:
                for key, row in tree.rows.items():
                    selected = self._tree_mask[row]
                    self._validity[metadata][key] = False if selected else tree._valid[key]
                    self._codes[metadata][key] = "RESTORED" if selected else tree.status_codes[key]
            if self._floating is not None:
                self._floating.prepare_metadata(metadata)
            src, dst = batch._active, 1 - batch._active
            try:
                with batch._device():
                    batch._upload(self._selected_ticks, self._ticks_host, "reset_plan")
                    if batch._mode == "cuda":
                        self._uploads += 1
                        self._uploaded_bytes += self._ticks_host.nbytes
                    self._launch(src, src, dst)
                    batch._sync()
                    self._syncs += 1
                try:
                    for change, (owner, attribute, initial, _, _, selected) in zip(
                        changed, self._controls
                    ):
                        if change:
                            np.copyto(getattr(owner, attribute), initial, where=selected)
                except Exception:
                    try:
                        for change, (owner, attribute, _, previous, _, selected) in zip(
                            changed, self._controls
                        ):
                            if change:
                                np.copyto(getattr(owner, attribute), previous, where=selected)
                    except Exception as error:
                        batch._control_error = error
                    raise
            except Exception:
                self._failures += 1
                raise
            for change, (owner, _, _, _, _, _) in zip(changed, self._controls):
                owner._dirty = owner._dirty or change
            if tree:
                tree._valid = self._validity[metadata]
                tree.status_codes = self._code_views[metadata]
                tree._pending_info = None
            if self._floating is not None:
                self._floating.publish(metadata)
            self._metadata_bank = metadata
            batch._active = dst
            batch._state_epoch += 1
            self._resets += 1
            return batch._state_epoch

    def stats(self):
        with self._batch._lock:
            return dict(
                closed=self._closed,
                closing=self._closing,
                resets=self._resets,
                failed_resets=self._failures,
                device_allocations=self._allocations,
                device_owned_bytes=self._device_bytes,
                host_workspace_bytes=0
                if self._closed
                else sum(a.nbytes for a in self._host_arrays()),
                uploads=self._uploads,
                uploaded_bytes=self._uploaded_bytes,
                explicit_completions=self._syncs,
            )

    def close(self):
        with self._batch._lock:
            if self._closed:
                return
            self._closing = True
            self._batch._ensure_cleanup_ready()
            close_dependencies(self)
            self._resources.close()
            self._batch._reset_plans.discard(self)
            self._arrays.clear()
            self._controls.clear()
            self._initial = self._selected_ticks = None
            self._tree_initial = self._tree_fk = self._tree_energy = None
            self._scalar_kernels = self._tree_kernels = None
            self._mask = self._ticks_host = self._initial_ticks = None
            self._tree_rows = self._tree_mask = None
            self._validity = self._codes = self._code_views = None
            self._tree = None
            self._floating = None
            self._closed, self._closing = True, False

    def __enter__(self):
        with self._batch._lock:
            self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
