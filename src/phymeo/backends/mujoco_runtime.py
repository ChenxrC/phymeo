"""Optional CPU MuJoCo execution of the original compiled native model.

No MuJoCo import occurs at module import. Native qpos/qvel, actuator controls
and the MuJoCo constraint contact law are deliberately distinct from TYX or
HOST scene coordinates and contact laws. Each runtime owns a model copy and
one independent MjData per logical world; the compilation record is borrowed.

MuJoCo 3.3.7 documents MjModel/MjData copy semantics and mj_contactForce:
https://mujoco.readthedocs.io/en/3.3.7/python.html
https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-contactforce

The separate explicit CUDA adapter uses mujoco-warp v0.0.2 declares mujoco>=3.3.7 and
warp-lang>=1.9.1. Its put_model/make_data/step/get_data_into are real APIs,
but get_data_into truncates overflowing contacts/constraints. A future GPU
adapter must check the device counts before readback, preserve its narrower
feature set and FP32/tolerance semantics, and separately qualify ownership,
reset and publication. Its lazy mode="cuda" dispatch has no CPU integration fallback.
https://github.com/google-deepmind/mujoco_warp/blob/v0.0.2/mujoco_warp/_src/io.py
https://github.com/google-deepmind/mujoco_warp/blob/v0.0.2/pyproject.toml
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from numbers import Integral
from threading import RLock
from types import MappingProxyType

import numpy as np

from ..training.resident_stepper import PhysicsStepFailure


COORDINATE_SYSTEM = "mujoco-native-qpos-qvel-v1"
CONTACT_LAW = "mujoco-native-constraint-v1"


def _mujoco():
    try:
        return import_module("mujoco")
    except ImportError as error:
        raise ImportError("the optional CPU MuJoCo runtime requires the mujoco package") from error


def _frozen(value, *, finite=True, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    if finite and not np.all(np.isfinite(array)):
        raise ValueError("MuJoCo values must be finite")
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=array.dtype).reshape(array.shape)


def _optional(value):
    return None if value is None else _frozen(value)


def _world_key(key):
    if isinstance(key, (bool, np.bool_)) or not isinstance(key, Integral):
        raise TypeError("MuJoCo world keys must be nonnegative i32 integers")
    key = int(key)
    if key < 0 or key > np.iinfo(np.int32).max:
        raise ValueError("MuJoCo world keys must be nonnegative i32 integers")
    return key


def _world_keys(keys, *, nonempty=False):
    result = tuple(_world_key(key) for key in keys)
    if len(set(result)) != len(result):
        raise ValueError("duplicate MuJoCo world key")
    if nonempty and not result:
        raise ValueError("a MuJoCo runtime requires at least one world")
    return result


@dataclass(frozen=True, init=False)
class MuJoCoControl:
    """Held native actuator values and optional native applied forces.

    ctrl follows the compiled actuator order. qfrc_applied is a native nv
    generalized-force vector; xfrc_applied is (nbody, 6), force then torque
    in world axes at the body COM, following MuJoCo's public convention.
    Omitted applied-force channels are zero, never carried from an old packet.
    """

    ctrl: np.ndarray
    qfrc_applied: np.ndarray | None
    xfrc_applied: np.ndarray | None

    def __init__(self, ctrl, *, qfrc_applied=None, xfrc_applied=None):
        object.__setattr__(self, "ctrl", _frozen(ctrl))
        object.__setattr__(self, "qfrc_applied", _optional(qfrc_applied))
        object.__setattr__(self, "xfrc_applied", _optional(xfrc_applied))


@dataclass(frozen=True, init=False)
class MuJoCoInitialState:
    """Explicit reset state in native coordinates, not a per-step state write.

    None uses the source model's qpos0, zero qvel/activation. Source keyframes
    are retained by MuJoCo but are never selected implicitly by this runtime.
    """

    q: np.ndarray | None
    v: np.ndarray | None
    act: np.ndarray | None
    time: float

    def __init__(self, q=None, v=None, *, act=None, time=0.0):
        if not np.isfinite(time) or time < 0:
            raise ValueError("initial MuJoCo time must be finite and nonnegative")
        object.__setattr__(self, "q", _optional(q))
        object.__setattr__(self, "v", _optional(v))
        object.__setattr__(self, "act", _optional(act))
        object.__setattr__(self, "time", float(time))


@dataclass(frozen=True)
class MuJoCoContact:
    """Detached complete mjContact fields and native contact-frame wrench.

    frame rows are normal/tangent axes; the normal points from geometry_ids[0]
    toward geometry_ids[1]. The six wrench values are force then torque from
    mj_contactForce, not an impulse or an inferred application point.
    Flex/element/vertex IDs remain present even for rigid-only test models.
    """

    index: int
    geometry_ids: tuple
    flex_ids: tuple
    element_ids: tuple
    vertex_ids: tuple
    dimension: int
    distance: float
    position: np.ndarray
    frame: np.ndarray
    include_margin: float
    friction: np.ndarray
    solref: np.ndarray
    solref_friction: np.ndarray
    solimp: np.ndarray
    regularized_mu: float
    cone_hessian: np.ndarray
    excluded: int
    constraint_address: int
    wrench_contact_frame: np.ndarray


@dataclass(frozen=True)
class MuJoCoStepReceipt:
    world: int
    tick: int
    input_time: float
    output_time: float
    h: float
    control: MuJoCoControl
    contacts: tuple
    qacc: np.ndarray
    qfrc_actuator: np.ndarray
    qfrc_constraint: np.ndarray
    warning_counts: tuple
    contact_phase: str = "last-mj-step-constraint-evaluation"
    contact_time: None = None
    raw_contact_fields: Mapping | None = None
    raw_constraint_fields: Mapping | None = None
    backend: str = "mujoco-cpu"


@dataclass(frozen=True)
class MuJoCoState:
    world: int
    q: np.ndarray
    v: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    time: float
    tick: int
    body_positions: np.ndarray
    body_quaternions: np.ndarray
    geometry_positions: np.ndarray
    geometry_matrices: np.ndarray
    qacc: np.ndarray
    qfrc_actuator: np.ndarray
    qfrc_passive: np.ndarray
    qfrc_constraint: np.ndarray
    qfrc_applied: np.ndarray
    xfrc_applied: np.ndarray
    sensordata: np.ndarray
    energy: np.ndarray
    contacts: tuple
    contact_time: float
    held_control: MuJoCoControl
    last_step: MuJoCoStepReceipt | None
    coordinate_system: str = COORDINATE_SYSTEM
    contact_law: str = CONTACT_LAW
    contact_phase: str = "endpoint-forward-recomputed"


def _contacts(mj, model, data):
    result = []
    for index in range(data.ncon):
        contact = data.contact[index]
        wrench = np.empty(6, dtype=np.float64)
        mj.mj_contactForce(model, data, index, wrench)
        result.append(
            MuJoCoContact(
                index=index,
                geometry_ids=tuple(int(value) for value in contact.geom),
                flex_ids=tuple(int(value) for value in contact.flex),
                element_ids=tuple(int(value) for value in contact.elem),
                vertex_ids=tuple(int(value) for value in contact.vert),
                dimension=int(contact.dim),
                distance=float(contact.dist),
                position=_frozen(contact.pos),
                frame=_frozen(contact.frame.reshape(3, 3)),
                include_margin=float(contact.includemargin),
                friction=_frozen(contact.friction),
                solref=_frozen(contact.solref),
                solref_friction=_frozen(contact.solreffriction),
                solimp=_frozen(contact.solimp),
                regularized_mu=float(contact.mu),
                cone_hessian=_frozen(contact.H, finite=False),
                excluded=int(contact.exclude),
                constraint_address=int(contact.efc_address),
                wrench_contact_frame=_frozen(wrench),
            )
        )
    return tuple(result)


class MuJoCoRuntime:
    """Own independent native CPU states behind the vector-env batch protocol.

    One step(h) means exactly one mj_step at the source model's timestep;
    another h is rejected before any world advances. Controls are held until
    replaced. Worlds advance in declared sparse-key order. Each step uses an
    unpublished MjData copy; a native failure discards only that candidate.
    Previously published worlds stay accepted and require a full reset before
    another batch step. No published q/v is repaired or rolled back.

    snapshot forward-evaluates a detached data copy at the accepted endpoint.
    Its contacts and body poses therefore share the state time, but are NOT
    the force integral of the preceding step. last_step stores the original
    post-mj_step constraint readings separately. Integrator-dependent stage
    time is intentionally unspecified (None), with the input/output interval
    retained. Reading snapshots never modifies the next step's warm start.
    """

    def __new__(cls, compilation, *, mode="cpu", **options):
        if cls is MuJoCoRuntime and mode == "cuda":
            from ._mujoco_warp_runtime import MuJoCoWarpRuntime

            return object.__new__(MuJoCoWarpRuntime)
        return object.__new__(cls)

    def __init__(self, compilation, *, world_keys=(0,), initial_states=None, mode="cpu"):
        if mode != "cpu":
            raise NotImplementedError("unknown MuJoCo GPU mode; use cpu or explicit cuda, no fallback")
        mj = _mujoco()
        factory = getattr(compilation, "fresh_native_model", None)
        if not callable(factory):
            raise TypeError("MuJoCoRuntime requires a source-bound MuJoCo compilation record")
        model = factory()
        if not isinstance(model, mj.MjModel):
            raise TypeError("fresh_native_model must return a native MuJoCo MjModel")
        self._initialize(compilation, model, world_keys, initial_states, mj)

    @classmethod
    def from_source(cls, path, *, world_keys=(0,), initial_states=None, mode="cpu", **options):
        from ..builders.mujoco_compiler import compile_mujoco

        record = compile_mujoco(path, require_asset=False)
        return cls(record, world_keys=world_keys, initial_states=initial_states, mode=mode, **options)

    @classmethod
    def from_xml(
        cls, xml, *, assets=None, includes=None, world_keys=(0,), initial_states=None, mode="cpu", **options
    ):
        from ..builders.mujoco_compiler import compile_mujoco_xml

        record = compile_mujoco_xml(xml, assets=assets, includes=includes, require_asset=False)
        return cls(record, world_keys=world_keys, initial_states=initial_states, mode=mode, **options)

    def _initialize(self, compilation, model, keys, initial_states, mj):
        self._lock = RLock()
        self._closed = False
        self._poisoned = False
        self._compilation = compilation
        self._mj = mj
        self._model = model
        self._world_keys = _world_keys(keys, nonempty=True)
        self._timestep = float(model.opt.timestep)
        if not np.isfinite(self._timestep) or self._timestep <= 0:
            raise ValueError("compiled MuJoCo timestep must be finite and positive")
        self._actuator_names = tuple(
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)
        )
        self._capabilities = MappingProxyType(
            {
                "backend": "mujoco-cpu",
                "mode": "cpu",
                "mujoco_version": str(mj.__version__),
                "coordinate_system": COORDINATE_SYSTEM,
                "contact_law": CONTACT_LAW,
                "native_model": True,
                "gpu": False,
                "nq": int(model.nq),
                "nv": int(model.nv),
                "nu": int(model.nu),
                "nbody": int(model.nbody),
                "ngeom": int(model.ngeom),
            }
        )
        if initial_states is None:
            initial_states = {key: MuJoCoInitialState() for key in self._world_keys}
        self._exact_mapping(initial_states, "initial states")
        initial = {}
        data = {}
        for key in self._world_keys:
            packet = initial_states[key]
            if not isinstance(packet, MuJoCoInitialState):
                raise TypeError("initial state values must be MuJoCoInitialState")
            candidate = self._initial_data(packet)
            initial[key] = MuJoCoInitialState(
                candidate.qpos, candidate.qvel, act=candidate.act, time=candidate.time
            )
            data[key] = candidate
        self._initial = initial
        self._data = data
        self._controls = {key: self._validated_control(MuJoCoControl(np.zeros(model.nu)))
                          for key in self._world_keys}
        self._ticks = dict.fromkeys(self._world_keys, 0)
        self._receipts = dict.fromkeys(self._world_keys)

    @property
    def world_keys(self):
        return self._world_keys

    @property
    def timestep(self):
        return self._timestep

    @property
    def actuator_names(self):
        return self._actuator_names

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def compilation(self):
        """Borrowed source/coordinate record, never the runtime's owned model."""
        return self._compilation

    def _check_open(self):
        if self._closed:
            raise RuntimeError("MuJoCo runtime is closed")

    def _select(self, keys):
        selected = self._world_keys if keys is None else _world_keys(keys)
        if not set(selected).issubset(self._world_keys):
            raise KeyError("selected MuJoCo world does not belong to this runtime")
        return selected

    def _exact_mapping(self, value, label):
        if not isinstance(value, Mapping):
            raise TypeError(f"MuJoCo {label} must be a world-key mapping")
        if set(value) != set(self._world_keys):
            raise ValueError(f"MuJoCo {label} must cover exactly all world keys")
        for key in value:
            _world_key(key)

    def _initial_data(self, packet):
        model = self._model
        data = self._mj.MjData(model)
        for name, value, shape in (
            ("qpos", packet.q, (model.nq,)),
            ("qvel", packet.v, (model.nv,)),
            ("act", packet.act, (model.na,)),
        ):
            if value is not None:
                if value.shape != shape:
                    raise ValueError(f"native initial {name} must have shape {shape}")
                getattr(data, name)[:] = value
        # A reset may select a valid native quaternion, never repair a bad one.
        for kind, address in zip(model.jnt_type, model.jnt_qposadr, strict=True):
            if kind == self._mj.mjtJoint.mjJNT_FREE:
                quaternion = data.qpos[address + 3 : address + 7]
            elif kind == self._mj.mjtJoint.mjJNT_BALL:
                quaternion = data.qpos[address : address + 4]
            else:
                continue
            if abs(float(quaternion @ quaternion) - 1.0) > 64 * np.finfo(float).eps:
                raise ValueError("native initial quaternion must have unit norm")
        data.time = packet.time
        self._mj.mj_forward(model, data)
        self._finite_state(data)
        return data

    def _validated_control(self, packet):
        model = self._model
        if isinstance(packet, Mapping):
            names = self._actuator_names
            if any(name is None for name in names) or len(set(names)) != len(names):
                raise ValueError("named controls require distinct names for every native actuator")
            if set(packet) != set(names):
                raise ValueError("named controls must cover exactly every native actuator")
            packet = MuJoCoControl([packet[name] for name in names])
        if not isinstance(packet, MuJoCoControl):
            raise TypeError("control values must be MuJoCoControl or a complete actuator-name mapping")
        qforce = np.zeros(model.nv) if packet.qfrc_applied is None else packet.qfrc_applied
        xforce = np.zeros((model.nbody, 6)) if packet.xfrc_applied is None else packet.xfrc_applied
        for name, value, shape in (
            ("ctrl", packet.ctrl, (model.nu,)),
            ("qfrc_applied", qforce, (model.nv,)),
            ("xfrc_applied", xforce, (model.nbody, 6)),
        ):
            if value.shape != shape:
                raise ValueError(f"native control {name} must have shape {shape}")
        return MuJoCoControl(packet.ctrl, qfrc_applied=qforce, xfrc_applied=xforce)

    def set_controls(self, mapping):
        with self._lock:
            self._check_open()
            self._exact_mapping(mapping, "controls")
            prepared = {key: self._validated_control(mapping[key]) for key in self._world_keys}
            self._controls = prepared

    @staticmethod
    def _finite_state(data):
        if not np.isfinite(data.time) or any(
            not np.all(np.isfinite(value)) for value in (data.qpos, data.qvel, data.act, data.qacc)
        ):
            raise RuntimeError("MuJoCo candidate contains a nonfinite native state")

    @staticmethod
    def _warnings(data):
        return tuple(int(item.number) for item in data.warning)

    def step(self, h):
        with self._lock:
            self._check_open()
            if self._poisoned:
                raise RuntimeError("MuJoCo partial publication requires a full reset before stepping")
            if not np.isfinite(h) or float(h) != self._timestep:
                raise ValueError("step h must equal the original compiled MuJoCo timestep")
            accepted = dict.fromkeys(self._world_keys, False)
            for key in self._world_keys:
                try:
                    original = self._data[key]
                    candidate = copy.copy(original)
                    control = self._controls[key]
                    candidate.ctrl[:] = control.ctrl
                    candidate.qfrc_applied[:] = control.qfrc_applied
                    candidate.xfrc_applied[:] = control.xfrc_applied
                    input_time = float(original.time)
                    before_warnings = self._warnings(candidate)
                    self._mj.mj_step(self._model, candidate)
                    self._finite_state(candidate)
                    if (float(candidate.time) != input_time + self._timestep
                            or candidate.time <= input_time):
                        raise RuntimeError("MuJoCo did not advance exactly one source timestep")
                    after_warnings = self._warnings(candidate)
                    if any(after > before for after, before in zip(
                        after_warnings, before_warnings, strict=True
                    )):
                        raise RuntimeError("MuJoCo raised a native warning during the unpublished step")
                    receipt = MuJoCoStepReceipt(
                        world=key, tick=self._ticks[key] + 1,
                        input_time=input_time, output_time=float(candidate.time), h=self._timestep,
                        control=control, contacts=_contacts(self._mj, self._model, candidate),
                        qacc=_frozen(candidate.qacc),
                        qfrc_actuator=_frozen(candidate.qfrc_actuator),
                        qfrc_constraint=_frozen(candidate.qfrc_constraint),
                        warning_counts=after_warnings,
                    )
                except Exception as error:
                    self._poisoned = any(accepted.values())
                    raise PhysicsStepFailure(
                        f"MuJoCo world {key} failed before its candidate publication",
                        accepted_worlds=accepted,
                    ) from error
                self._data[key] = candidate
                self._ticks[key] += 1
                self._receipts[key] = receipt
                accepted[key] = True
            return MappingProxyType(accepted)

    def snapshot(self, keys=None):
        with self._lock:
            self._check_open()
            result = {}
            for key in self._select(keys):
                committed = self._data[key]
                endpoint = copy.copy(committed)
                self._mj.mj_forward(self._model, endpoint)
                self._finite_state(endpoint)
                result[key] = MuJoCoState(
                    world=key, q=_frozen(committed.qpos), v=_frozen(committed.qvel),
                    act=_frozen(committed.act), ctrl=_frozen(committed.ctrl),
                    time=float(committed.time), tick=self._ticks[key],
                    body_positions=_frozen(endpoint.xpos), body_quaternions=_frozen(endpoint.xquat),
                    geometry_positions=_frozen(endpoint.geom_xpos),
                    geometry_matrices=_frozen(endpoint.geom_xmat.reshape(-1, 3, 3)),
                    qacc=_frozen(endpoint.qacc), qfrc_actuator=_frozen(endpoint.qfrc_actuator),
                    qfrc_passive=_frozen(endpoint.qfrc_passive),
                    qfrc_constraint=_frozen(endpoint.qfrc_constraint),
                    qfrc_applied=_frozen(committed.qfrc_applied),
                    xfrc_applied=_frozen(committed.xfrc_applied), sensordata=_frozen(endpoint.sensordata),
                    energy=_frozen(endpoint.energy), contacts=_contacts(self._mj, self._model, endpoint),
                    contact_time=float(committed.time), held_control=self._controls[key],
                    last_step=self._receipts[key],
                )
            return MappingProxyType(result)

    def reset(self, keys=None):
        with self._lock:
            self._check_open()
            selected = self._select(keys)
            if self._poisoned and set(selected) != set(self._world_keys):
                raise ValueError("partial publication requires resetting all MuJoCo worlds")
            prepared = {key: self._initial_data(self._initial[key]) for key in selected}
            controls = {key: self._validated_control(MuJoCoControl(np.zeros(self._model.nu)))
                        for key in selected}
            for key in selected:
                self._data[key] = prepared[key]
                self._controls[key] = controls[key]
                self._ticks[key] = 0
                self._receipts[key] = None
            if set(selected) == set(self._world_keys):
                self._poisoned = False

    def clone(self, world_keys=None):
        """Copy accepted integration state/controls; reset still means source initial state."""
        with self._lock:
            self._check_open()
            if self._poisoned:
                raise RuntimeError("reset the partially published MuJoCo runtime before cloning")
            selected = self._select(world_keys)
            model = copy.copy(self._model)
            clone = object.__new__(type(self))
            clone._initialize(
                self._compilation, model, selected,
                {key: self._initial[key] for key in selected}, self._mj,
            )
            signature = self._mj.mjtState.mjSTATE_INTEGRATION
            for key in selected:
                state = np.empty(self._mj.mj_stateSize(self._model, signature), dtype=np.float64)
                self._mj.mj_getState(self._model, self._data[key], state, signature)
                self._mj.mj_setState(clone._model, clone._data[key], state, signature)
                clone._controls[key] = self._controls[key]
                clone._ticks[key] = self._ticks[key]
                clone._receipts[key] = self._receipts[key]
            return clone

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._data.clear()
            self._initial.clear()
            self._controls.clear()
            self._receipts.clear()
            self._model = None
            self._compilation = None

    def __enter__(self):
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
