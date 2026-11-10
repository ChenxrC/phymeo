"""Explicit optional MuJoCo-Warp CUDA mode; no CPU integration fallback.

API domain: mujoco 3.3.7 / mujoco-warp 0.0.2 / warp-lang 1.17.0. The parent
qualified dependency pair is separate from adapter validation. v0.0.2 rejects
some native features and raises GPU solver tolerance to at least 1e-6. We keep
that declaration and use its public put_model/put_data/step/contact_force and
get_data_into. Capacity is checked BEFORE get_data_into, which otherwise clips.

The v0.0.2 concatenated get_state/set_state helper's EQ_ACTIVE offset and its
unsupported USERDATA/PLUGIN fields are avoided: public wp.copy transfers each
actual integration array between two owned resident buffers. No third-party
source is patched. Models requiring userdata/plugins/flex are rejected here.

https://github.com/google-deepmind/mujoco_warp/blob/v0.0.2/mujoco_warp/_src/io.py
https://github.com/google-deepmind/mujoco_warp/blob/v0.0.2/mujoco_warp/_src/support.py
"""

import copy
from dataclasses import dataclass, fields, replace
from importlib import import_module
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..training.resident_stepper import PhysicsStepFailure
from .mujoco_runtime import MuJoCoControl, MuJoCoRuntime, MuJoCoStepReceipt, _frozen

GPU_CONTACT_LAW = "mujoco-warp-native-constraint-v0.0.2"
_INTEGRATION_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "act",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "eq_active",
    "mocap_pos",
    "mocap_quat",
)


def _capacity(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _dependencies():
    try:
        mw = import_module("mujoco_warp")
        wp = import_module("warp")
    except ImportError as error:
        raise ImportError(
            "explicit CUDA mode requires optional MuJoCo-Warp and warp-lang"
        ) from error
    return mw, wp


def _raw_array(value):
    value = np.asarray(value)
    return _frozen(value, dtype=value.dtype)


@dataclass(frozen=True)
class MuJoCoWarpContact:
    """Actual v0.0.2 device Contact fields, not fabricated mjContact fields."""

    index: int
    world: int
    world_slot: int
    geometry_ids: tuple
    dimension: int
    distance: float
    position: np.ndarray
    frame: np.ndarray
    include_margin: float
    friction: np.ndarray
    solref: np.ndarray
    solref_friction: np.ndarray
    solimp: np.ndarray
    constraint_addresses: np.ndarray
    contact_type: int
    geometry_collision_id: int
    wrench_contact_frame: np.ndarray


class MuJoCoWarpRuntime(MuJoCoRuntime):
    """Owned GPU integration with atomic publication of the entire batch.

    Integration remains resident in two GPU buffers. CPU MjData mirrors only
    accepted readback and supplies detached endpoint observation recomputation;
    it never executes mj_step. last_step.contacts and its raw fields are the
    actual GPU solver readings. snapshot.contacts explicitly identifies its
    endpoint CPU forward recomputation, not a GPU force/impulse claim.

    A capacity, CUDA, extraction or validation failure leaves every previously
    published world unchanged, returns a known all-False PhysicsStepFailure,
    and does not swap the active buffer. Failed candidate work may have run on
    the GPU; it is never an accepted physical publication. No per-world partial
    acceptance is claimed by this batched adapter.
    """

    def __init__(
        self,
        compilation,
        *,
        world_keys=(0,),
        initial_states=None,
        mode="cuda",
        device="cuda:0",
        contact_capacity=64,
        constraint_capacity=256,
    ):
        if mode != "cuda":
            raise ValueError("MuJoCoWarpRuntime requires explicit cuda mode")
        self._contact_capacity = _capacity(contact_capacity, "contact_capacity")
        self._constraint_capacity = _capacity(constraint_capacity, "constraint_capacity")
        mw, wp = _dependencies()
        super().__init__(
            compilation, world_keys=world_keys, initial_states=initial_states, mode="cpu"
        )
        self._initialize_gpu(mw, wp, device)

    def _initialize_gpu(self, mw, wp, device):
        versions = (self._mj.__version__, mw.__version__, wp.__version__)
        if versions != ("3.3.7", "0.0.2", "1.17.0"):
            raise RuntimeError(
                "this CUDA adapter requires the explicit MuJoCo 3.3.7 / "
                "MuJoCo-Warp 0.0.2 / Warp 1.17.0 API pair"
            )
        if self._model.nuserdata or self._model.nplugin or self._model.nflex:
            raise NotImplementedError(
                "this GPU adapter does not support userdata, plugins or flex state; no CPU fallback"
            )
        self._mw, self._wp = mw, wp
        wp.init()
        self._device = wp.get_device(device)
        if not self._device.is_cuda:
            raise ValueError("MuJoCo CUDA mode requires an actual CUDA device; no CPU fallback")
        self._world_slots = MappingProxyType(
            {key: slot for slot, key in enumerate(self._world_keys)}
        )
        with wp.ScopedDevice(self._device):
            self._gpu_model = mw.put_model(self._model)
            active, work, mirror = self._new_buffers(self._data)
        self._gpu_data, self._gpu_work = active, work
        self._data = mirror
        info = dict(self._capabilities)
        info.update(
            backend="mujoco-warp-cuda",
            mode="cuda",
            gpu=True,
            device=str(self._device),
            mujoco_warp_version=mw.__version__,
            warp_version=wp.__version__,
            state_precision="fp32-device-integration",
            contact_law=GPU_CONTACT_LAW,
            publication="whole-batch-atomic",
            contact_capacity=self._contact_capacity,
            total_contact_capacity=len(self._world_keys) * self._contact_capacity,
            constraint_capacity=self._constraint_capacity,
            constraint_rows_strictly_below_capacity=True,
            solver_tolerance=float(self._gpu_model.opt.tolerance.numpy()[0]),
            endpoint_observer="detached-cpu-mj-forward",
            native_model=False,
        )
        self._capabilities = MappingProxyType(info)

    @property
    def world_slots(self):
        return self._world_slots

    def _make_data(self, native):
        return self._mw.put_data(
            self._model,
            native,
            nworld=len(self._world_keys),
            nconmax=self._contact_capacity,
            njmax=self._constraint_capacity,
        )

    def _upload_integration(self, target, native):
        for name in _INTEGRATION_FIELDS:
            destination = getattr(target, name)
            if not destination.size:
                continue
            if name == "time":
                source = np.asarray([native[key].time for key in self._world_keys])
            else:
                source = np.stack([getattr(native[key], name) for key in self._world_keys])
            self._wp.copy(
                destination,
                self._wp.array(source, dtype=destination.dtype, device=self._device),
            )

    def _copy_integration(self, destination, source):
        for name in _INTEGRATION_FIELDS:
            target = getattr(destination, name)
            if target.size:
                self._wp.copy(target, getattr(source, name))

    def _check_capacity(self, data):
        count = int(data.nacon.numpy()[0])
        constraints = np.asarray(data.nefc.numpy())
        if count < 0 or count > data.naconmax:
            raise RuntimeError("MuJoCo-Warp contact capacity exceeded before readback")
        if constraints.shape != (len(self._world_keys),) or np.any(constraints < 0):
            raise RuntimeError("MuJoCo-Warp returned invalid per-world constraint counts")
        # v0.0.2 connect/weld kernels can return before writing a final row
        # group when it exactly fills njmax. Reserve one row conservatively.
        if np.any(constraints >= data.njmax):
            raise RuntimeError("MuJoCo-Warp constraint capacity reached before readback")
        return count, constraints.astype(np.int64)

    def _readback(self, data, template):
        # get_data_into clips overflow, so this check must precede every call.
        self._check_capacity(data)
        result = {}
        for slot, key in enumerate(self._world_keys):
            observed = copy.copy(template[key])
            self._mw.get_data_into(observed, self._model, data, slot)
            self._finite_state(observed)
            if any(
                not np.isfinite(value).all()
                for value in (observed.ctrl, observed.qfrc_applied, observed.xfrc_applied)
            ):
                raise RuntimeError("MuJoCo-Warp returned nonfinite stored control channels")
            result[key] = observed
        return result

    def _new_buffers(self, native):
        first = native[self._world_keys[0]]
        active = self._make_data(first)
        work = self._make_data(first)
        self._upload_integration(active, native)
        self._upload_integration(work, native)
        self._wp.synchronize_device(self._device)
        return active, work, self._readback(active, native)

    def _validated_control(self, packet):
        result = super()._validated_control(packet)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            if any(
                not np.isfinite(np.asarray(value, dtype=np.float32)).all()
                for value in (result.ctrl, result.qfrc_applied, result.xfrc_applied)
            ):
                raise ValueError("CUDA control channels must be finite when represented as FP32")
        return result

    def _upload_controls(self, data):
        for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
            destination = getattr(data, name)
            if not destination.size:
                continue
            source = np.stack([getattr(self._controls[key], name) for key in self._world_keys])
            self._wp.copy(
                destination,
                self._wp.array(source, dtype=destination.dtype, device=self._device),
            )

    def _solver_contacts(self, data, count, constraints):
        # All generated sites, including zero-force sites, survive this read.
        raw = {
            field.name: getattr(data.contact, field.name).numpy()[:count]
            for field in fields(data.contact)
        }
        world_ids = np.asarray(raw["worldid"])
        if np.any(world_ids < 0) or np.any(world_ids >= len(self._world_keys)):
            raise RuntimeError("MuJoCo-Warp contact carries an invalid world slot")
        wrenches = np.zeros((count, 6), dtype=np.float32)
        if count:
            indices = self._wp.array(np.arange(count, dtype=np.int32), device=self._device)
            forces = self._wp.zeros(count, dtype=self._wp.spatial_vector, device=self._device)
            self._mw.contact_force(self._gpu_model, data, indices, False, forces)
            self._wp.synchronize_device(self._device)
            wrenches = forces.numpy()
        raw_constraints = {
            field.name: (getattr(data.efc, field.name).numpy(), getattr(field.type, "shape", ()))
            for field in fields(data.efc)
        }
        contacts, contact_fields, constraint_fields = {}, {}, {}
        for slot, key in enumerate(self._world_keys):
            selected = np.flatnonzero(world_ids == slot)
            contacts[key] = tuple(
                MuJoCoWarpContact(
                    index=int(index),
                    world=key,
                    world_slot=slot,
                    geometry_ids=tuple(int(value) for value in raw["geom"][index]),
                    dimension=int(raw["dim"][index]),
                    distance=float(raw["dist"][index]),
                    position=_frozen(raw["pos"][index]),
                    frame=_frozen(raw["frame"][index]),
                    include_margin=float(raw["includemargin"][index]),
                    friction=_frozen(raw["friction"][index]),
                    solref=_frozen(raw["solref"][index]),
                    solref_friction=_frozen(raw["solreffriction"][index]),
                    solimp=_frozen(raw["solimp"][index]),
                    constraint_addresses=_raw_array(raw["efc_address"][index]),
                    contact_type=int(raw["type"][index]),
                    geometry_collision_id=int(raw["geomcollisionid"][index]),
                    wrench_contact_frame=_frozen(wrenches[index]),
                )
                for index in selected
            )
            contact_fields[key] = MappingProxyType(
                {name: _raw_array(value[selected]) for name, value in raw.items()}
            )
            rows = {}
            for name, (value, shape) in raw_constraints.items():
                # Solver scalars/vectors use nworld/nv dimensions, not nefc.
                if len(shape) > 1 and shape[1] in ("njmax", "njmax_pad"):
                    rows[name] = _raw_array(value[slot, : constraints[slot]])
                else:
                    rows[name] = _raw_array(value[slot])
            constraint_fields[key] = MappingProxyType(rows)
        return contacts, contact_fields, constraint_fields

    def step(self, h):
        with self._lock:
            self._check_open()
            if not np.isfinite(h) or float(h) != self._timestep:
                raise ValueError("step h must equal the original compiled MuJoCo timestep")
            accepted = dict.fromkeys(self._world_keys, False)
            try:
                with self._wp.ScopedDevice(self._device):
                    self._copy_integration(self._gpu_work, self._gpu_data)
                    self._upload_controls(self._gpu_work)
                    self._mw.step(self._gpu_model, self._gpu_work)
                    self._wp.synchronize_device(self._device)
                    count, constraints = self._check_capacity(self._gpu_work)
                    observed = self._readback(self._gpu_work, self._data)
                    contacts, raw_contacts, raw_constraints = self._solver_contacts(
                        self._gpu_work, count, constraints
                    )
                receipts = {}
                for key in self._world_keys:
                    original, candidate = self._data[key], observed[key]
                    expected_time = np.float32(
                        np.float32(original.time) + np.float32(self._timestep)
                    )
                    if candidate.time != float(expected_time) or candidate.time <= original.time:
                        raise RuntimeError("MuJoCo-Warp did not advance exactly one FP32 timestep")
                    receipts[key] = MuJoCoStepReceipt(
                        world=key,
                        tick=self._ticks[key] + 1,
                        input_time=float(original.time),
                        output_time=float(candidate.time),
                        h=self._timestep,
                        control=self._controls[key],
                        contacts=contacts[key],
                        qacc=_frozen(candidate.qacc),
                        qfrc_actuator=_frozen(candidate.qfrc_actuator),
                        qfrc_constraint=_frozen(candidate.qfrc_constraint),
                        warning_counts=(),
                        backend="mujoco-warp-cuda",
                        contact_phase="last-mjwarp-step-constraint-evaluation",
                        raw_contact_fields=raw_contacts[key],
                        raw_constraint_fields=raw_constraints[key],
                        stored_control=MuJoCoControl(
                            candidate.ctrl,
                            qfrc_applied=candidate.qfrc_applied,
                            xfrc_applied=candidate.xfrc_applied,
                        ),
                    )
            except Exception as error:
                raise PhysicsStepFailure(
                    "MuJoCo-Warp candidate batch failed before publication",
                    accepted_worlds=accepted,
                ) from error
            self._gpu_data, self._gpu_work = self._gpu_work, self._gpu_data
            self._data = observed
            self._receipts = receipts
            self._ticks = {key: tick + 1 for key, tick in self._ticks.items()}
            return MappingProxyType(dict.fromkeys(self._world_keys, True))

    def snapshot(self, keys=None):
        states = super().snapshot(keys)
        return MappingProxyType(
            {
                key: replace(
                    state,
                    contact_law=GPU_CONTACT_LAW,
                    contact_phase="endpoint-cpu-forward-recomputed",
                )
                for key, state in states.items()
            }
        )

    def reset(self, keys=None):
        with self._lock:
            self._check_open()
            selected = self._select(keys)
            if not selected:
                return
            proposed = dict(self._data)
            for key in selected:
                proposed[key] = self._initial_data(self._initial[key])
            controls, ticks, receipts = (
                dict(self._controls),
                dict(self._ticks),
                dict(self._receipts),
            )
            for key in selected:
                controls[key] = self._validated_control(
                    type(self._controls[key])(np.zeros(self._model.nu))
                )
                ticks[key], receipts[key] = 0, None
            with self._wp.ScopedDevice(self._device):
                active, work, mirror = self._new_buffers(proposed)
            self._gpu_data, self._gpu_work, self._data = active, work, mirror
            self._controls, self._ticks, self._receipts = controls, ticks, receipts

    def clone(self, world_keys=None):
        with self._lock:
            self._check_open()
            selected = self._select(world_keys)
            cloned = object.__new__(type(self))
            cloned._contact_capacity = self._contact_capacity
            cloned._constraint_capacity = self._constraint_capacity
            cloned._initialize(
                self._compilation,
                copy.copy(self._model),
                selected,
                {key: self._initial[key] for key in selected},
                self._mj,
            )
            for key in selected:
                cloned._data[key] = copy.copy(self._data[key])
            cloned._initialize_gpu(self._mw, self._wp, self._device)
            cloned._controls = {key: self._controls[key] for key in selected}
            cloned._ticks = {key: self._ticks[key] for key in selected}
            cloned._receipts = {key: self._receipts[key] for key in selected}
            return cloned

    def close(self):
        with self._lock:
            if self._closed:
                return
            # Every accepted operation already synchronizes before publication.
            self._gpu_data = None
            self._gpu_work = None
            self._gpu_model = None
            super().close()
