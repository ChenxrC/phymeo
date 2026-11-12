"""Owned PhySolve runtime with explicit candidate execution and publication.

Two persistent device banks make publication transactional. Synchronous helpers
use the same begin/enqueue/complete/accept contract. Completion reads a compact
device validation result; this is not Graph or P2PE C ABI qualification. CPU MuJoCo prepares
initial data. Native coordinates are preserved, including WXYZ free-root qpos.
"""

from contextlib import contextmanager
from numbers import Integral
from threading import RLock
from types import MappingProxyType

import numpy as np

from ..builders.mujoco_compiler import MuJoCoCompilation
from ..builders.mujoco_scene import CompiledMuJoCoWorld
from .physolve_initial import INTEGRATION_FIELDS, NativeInitialState
from .physolve_payload import PhySolvePayload

OBSERVATION_FIELDS = (
    "qpos",
    "qvel",
    "qacc",
    "time",
    "ctrl",
    "qfrc_applied",
    "qfrc_actuator",
    "qfrc_constraint",
    "xpos",
    "xquat",
)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class PhySolveInputError(RuntimeError):
    """Only input conversion failed; the unpublished candidate can be discarded."""


class PhySolveRuntime:
    """Own model/banks/handles/stream; borrow the immutable compilation source.

    Contacts and xpos are solver evaluation-time results, not a separate
    endpoint mj_forward recomputation. reset restores compile-time initial
    state (no keyframe selected), then performs native forward.
    """

    body_observation_contract_version = 1
    contact_observation_contract_version = 1
    contact_reference_contract_version = 1
    body_measurement_contract_version = 1
    site_measurement_contract_version = 1
    prepared_ray_contract_version = 1
    prepared_ray_capacity_rows = True
    selected_update_contract_version = 1
    device_reset_contract_version = 1
    free_body_update_contract_version = 1
    free_body_component_contract_version = 1
    stream_binding_contract_version = 1
    step_graph_contract_version = 1

    def __init__(
        self,
        compilation,
        *,
        payload,
        nworld=1,
        nconmax=64,
        njmax=256,
        device="cuda:0",
        initial_state=None,
        scalar_controls=(),
        body_observations=False,
        contact_observations=False,
        body_measurements=False,
    ):
        if type(body_observations) is not bool:
            raise TypeError("body_observations must be bool")
        if type(contact_observations) is not bool:
            raise TypeError("contact_observations must be bool")
        if type(body_measurements) is not bool:
            raise TypeError('body_measurements must be bool')
        world = compilation if isinstance(compilation, CompiledMuJoCoWorld) else None
        if world is not None:
            if initial_state is not None or world.binding_version != 1:
                raise ValueError("compiled world owns its versioned initial state")
            compilation = world.compilation
        if not isinstance(compilation, MuJoCoCompilation):
            raise TypeError("a Phymeo MuJoCoCompilation is required")
        if not isinstance(payload, PhySolvePayload):
            raise TypeError("an explicit frozen PhySolvePayload is required")
        self.nworld = _positive(nworld, "nworld")
        self.nconmax = _positive(nconmax, "nconmax")
        self.njmax = _positive(njmax, "njmax")
        import mujoco
        import physolve_tyx as frontend
        import tyx as tx

        # Initialize/validate before loading the native library or allocating GPU owners.
        if world is None:
            cpu = compilation.fresh_native_model()
            initial = mujoco.MjData(cpu)
            initial.qpos[:] = compilation.native_initial_qpos
            initial.qvel[:] = compilation.native_initial_qvel
        else:
            cpu, initial = world.fresh_native_data()
        if initial_state is not None and not isinstance(initial_state, NativeInitialState):
            raise TypeError("initial_state requires NativeInitialState")
        initial_state = initial_state or NativeInitialState.from_data(compilation, initial)
        initial_state.apply(compilation, initial)

        self._lock = RLock()
        self._tx, self._frontend = tx, frontend
        self._native = payload.load()
        self._closed, self._closing, self._active = False, False, 0
        self._banks, self._handles = [], []
        self._stream = self._model = None
        self._owned_stream = None
        self._stream_retained = []
        self._stream_rebindings = 0
        self._step_graphs = None
        self._graph_captures = 0
        self._graph_replays = 0
        self._reset = None
        self._controls_binding = None
        self._body_observations = None
        self._contact_observations = None
        self._body_measurements = None
        self._site_measurements = []
        self._ray_queries = []
        self._free_body_updates = None
        self._validation = self._storage = None
        self._storage_generation = 1
        self._pending = None
        self._execution_epoch = 0
        self._compilation = compilation
        self._cpu_model = cpu
        cpu = self._cpu_model
        if cpu.nuserdata or cpu.nplugin or cpu.nflex or cpu.nsensor:
            raise NotImplementedError(
                "PhySolve profile excludes userdata, plugins, flex and sensors"
            )
        if cpu.opt.integrator not in (
            mujoco.mjtIntegrator.mjINT_EULER, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        ):
            raise NotImplementedError("this direct qualification profile requires Euler or implicitfast")
        if cpu.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON:
            raise NotImplementedError("PhySolve requires the Newton constraint solve algorithm")
        # This first profile deliberately fixes dense Jacobians; sparse capacity
        # accounting must be qualified before allowing a sparse request.
        if cpu.opt.jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
            raise NotImplementedError("sparse Jacobians are not qualified in this profile")
        cpu.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        selected = tx.get_device(device)
        if not selected.is_cuda or selected.arch != 120:
            raise RuntimeError("frozen payload is qualified only on CUDA sm_120")
        self._device = selected
        try:
            with tx.use_device(selected):
                self._stream = tx.CudaStream(device=selected.ordinal)
                self._owned_stream = self._stream
                with tx.use_stream(self._stream):
                    mujoco.mj_forward(cpu, initial)
                    if initial.ncon > self.nconmax or initial.nefc > self.njmax:
                        raise ValueError("initial model exceeds contact/constraint capacity")
                    self._model = frontend.put_model(cpu)
                    self._model.opt.graph_conditional = True
                    for _ in range(2):
                        data = frontend.put_data(
                            cpu, initial, nworld=self.nworld, nconmax=self.nconmax, njmax=self.njmax
                        )
                        self._banks.append(data)
                        # Each bank owns its frontend scratch before Host prepare/capture.
                        frontend.prepare_workspace(self._model, data)
                        handle = self._native._create_runtime(self._model, data, "tyx", "tyx")
                        self._handles.append(handle)
                        self._native._prepare_runtime(handle)
                        self._native._forward_runtime(handle)
                        self._validate(data)
                    from ._physolve_reset import PreparedReset

                    self._reset = PreparedReset(
                        self._banks, self.nworld, selected, OBSERVATION_FIELDS
                    )
                    self._reset.prepare(self._stream)
                    if body_observations:
                        from ._physolve_bodies import PreparedBodyObservations

                        self._body_observations = PreparedBodyObservations(
                            self._model, self._banks, selected
                        )
                        self._body_observations.prepare(self._stream, self._reset.all_mask)
                    from ._physolve_binding import bind_storage
                    from ._physolve_validation import PreparedValidation

                    self._validation = PreparedValidation(
                        self._banks, selected,
                        extra_banks=None if self._body_observations is None else self._body_observations.banks,
                    )
                    self._validation.prepare(self._stream)
                    if contact_observations:
                        from ._physolve_contacts import PreparedContacts

                        self._contact_observations = PreparedContacts(
                            self._model, self._banks, selected, self._validation.flags
                        )
                        self._contact_observations.prepare(self._stream, self._reset.all_mask)
                        self._validation.read(self._stream)
                        self._validation.readbacks = 0
                    if scalar_controls:
                        from ._physolve_controls import PreparedControls
                        self._controls_binding = PreparedControls(
                            scalar_controls, self._banks, cpu, self.nworld, device
                        )
                        self._controls_binding.prepare(self._stream)
                    if body_measurements:
                        from ._physolve_measurements import PreparedMeasurements
                        self._body_measurements = PreparedMeasurements(
                            self._model, self._banks, selected, self._validation.flags,
                        )
                        self._body_measurements.prepare(self._stream, self._reset.all_mask)
                        self._validation.read(self._stream)
                        self._validation.readbacks = 0
                    self._storage = bind_storage(
                        self, self._banks, self._stream, selected, self._storage_generation,
                        self._controls_binding,
                        self._body_observations,
                        self._contact_observations,
                        self._validation.flags,
                        self._body_measurements,
                    )
                    self._complete()
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup:
                error.runtime = self  # Retain owner for an explicit cleanup retry.
                raise error from cleanup
            raise

    @contextmanager
    def _scope(self):
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("PhySolve runtime is closing or closed")
            with self._tx.use_device(self._device), self._tx.use_stream(self._stream):
                yield

    def prepare_site_measurements(self, binding):
        """Cold-create an owned site projection plan; callers only borrow it."""
        from ._physolve_sites import PreparedSiteMeasurements

        with self._scope():
            if self._pending is not None:
                raise RuntimeError('site preparation requires idle execution')
            plan = PreparedSiteMeasurements.__new__(PreparedSiteMeasurements)
            plan.resources = []
            plan.storage = self.storage_binding()
            self._site_measurements.append(plan)
            try:
                plan.__init__(self, binding)
            except Exception:
                # Retain even partial uploads until normal runtime cleanup completes.
                if not plan.resources:
                    self._site_measurements.pop()
                raise
            return plan

    def prepare_ray_query(self, scene, *, max_rays):
        """Own a prepared device query over this compilation's endpoint geometry."""
        from ..builders.native_rays import NativeRayScene
        from .prepared_rays import PreparedRayQuery

        with self._scope():
            if self._pending is not None:
                raise RuntimeError('ray preparation requires idle execution')
            if (not isinstance(scene, NativeRayScene) or scene.contract_version != 1
                    or scene.compilation is not self._compilation):
                raise ValueError('ray scene belongs to another compilation or contract')
            plan = PreparedRayQuery.__new__(PreparedRayQuery)
            plan.resources, plan.meshes = [], []
            plan.storage = self.storage_binding()
            self._ray_queries.append(plan)
            try:
                plan.__init__(plan.storage, scene.surfaces, scene.bodies, max_rays,
                              self.nworld, lambda: self.observation_bank)
            except Exception:
                if not plan.resources and not plan.meshes:
                    self._ray_queries.pop()
                raise
            return plan

    def _complete(self):
        self._synchronize_stream(self._stream)

    def _synchronize_stream(self, stream):
        for handle in self._handles:
            if handle is not None:
                self._native._synchronize_runtime_stream(
                    handle, int(self._device.context), int(stream.handle)
                )
                return
        if stream is not None:
            stream.synchronize()

    def bind_stream(self, handle=0, *, owner=None):
        """Bind a borrowed same-context stream, or restore the owned stream (0).

        Cold and synchronous: callers must finish/accept pending execution first
        and retain external handles through unbind/close. Storage and prepared
        field plans stay valid and must use storage.stream on every submission.
        Preparation failure restores old bindings; failed recovery permits close
        retry only. Returns whether the canonical stream changed.
        """
        from ._physolve_stream import bind_stream
        return bind_stream(self, handle, owner)

    @property
    def step_graph_ready(self):
        return self._step_graphs is not None and self._step_graphs.ready

    def _release_step_graphs(self):
        if self._step_graphs is not None:
            self._step_graphs.close()
            self._step_graphs = None

    def prepare_step_graph(self, *, steps=4):
        """Capture both banks without executing physics; fixed four-step profile.

        No warmup rollout, physical state clone, or implicit direct fallback.
        Stream rebind retires these Graphs and requires explicit preparation.
        """
        if _positive(steps, 'steps') != 4:
            raise ValueError('fixed step Graph v1 requires four substeps')
        with self._scope():
            self._require_idle()
            if self.step_graph_ready:
                return False
            if self._stream.is_capturing:
                raise RuntimeError('cannot prepare step Graph during another capture')
            self._complete()
            self._release_step_graphs()
            from ._physolve_graph import PreparedStepGraphs
            self._step_graphs = PreparedStepGraphs(self, steps)
            try:
                self._step_graphs.prepare()
                self._graph_captures += 2
            except Exception as error:
                try:
                    self._release_step_graphs()
                except Exception as cleanup:
                    self._closing = True
                    error.runtime = self
                    raise error from cleanup
                raise
            return True

    def _validate(self, data):
        if self._validation is not None:
            self._validation.read(self._stream)
            return
        # Cold construction validation before persistent device checks exist.
        self._complete()
        if (
            int(data.nacon.numpy()[0]) > data.naconmax
            or int(data.ncollision.numpy()[0]) > data.naconmax
            or np.any(data.nefc.numpy() > data.njmax)
        ):
            raise RuntimeError("PhySolve contact/constraint capacity exceeded; not published")
        for name in ("qpos", "qvel", "qacc", "time"):
            if not np.isfinite(getattr(data, name).numpy()).all():
                raise RuntimeError(f"non-finite PhySolve {name}; not published")

    def _candidate(self):
        index = 1 - self._active
        data, previous = self._banks[index], self._banks[self._active]
        for name in INTEGRATION_FIELDS:
            getattr(data, name).assign(getattr(previous, name), stream=self._stream)
        return index, data

    def _controls(self, controls):
        result = {}
        shapes = {
            "ctrl": (self.nworld, self._cpu_model.nu),
            "qfrc_applied": (self.nworld, self._cpu_model.nv),
        }
        for name, value in controls.items():
            if value is None:
                continue
            expected = shapes[name]
            array = np.asarray(value)
            if array.shape != expected or array.dtype.kind not in "fiu":
                raise ValueError(f"{name} requires numeric shape {expected}")
            with np.errstate(over="ignore", invalid="ignore"):
                array = np.array(array, dtype=np.float32, order="C", copy=True)
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite in float32")
            result[name] = array
        return result

    def step(self, *, ctrl=None, qfrc_applied=None):
        """Advance one compiled timestep; controls persist until changed/reset."""
        with self._scope():
            if ctrl is not None and self._controls_binding is not None:
                raise ValueError("managed scalar controls require target fields, not raw ctrl")
            controls = self._controls(dict(ctrl=ctrl, qfrc_applied=qfrc_applied))
            with self._synchronous_execution() as ticket:
                for name, value in controls.items():
                    getattr(self._banks[ticket[1]], name).assign(value, stream=self._stream)
                self.enqueue_step(ticket)

    def forward(self):
        """Refresh derived values without advancing time or changing qpos/qvel."""
        with self._scope():
            with self._synchronous_execution() as ticket:
                self.enqueue_forward(ticket)

    def reset(self, mask=None):
        """Reset selected worlds; keep unselected integration and snapshot values."""
        with self._scope():
            self._require_idle()
            if mask is not None:
                mask = np.asarray(mask)
                if mask.dtype != np.bool_ or mask.shape != (self.nworld,):
                    raise ValueError("reset mask must be bool[nworld]")
                if not mask.any():
                    return
            with self._synchronous_execution() as ticket:
                self.enqueue_reset(ticket, mask)

    @contextmanager
    def _synchronous_execution(self):
        # Refuse someone else's pending transaction before entering cleanup scope.
        self._require_idle()
        try:
            ticket = self.begin_execution()
            yield ticket
            self.complete_execution(ticket)
            self.accept_execution(ticket)
        except Exception as primary:
            ticket = self.pending_execution
            if ticket is not None:
                try:
                    self.discard_execution(ticket)
                except Exception as cleanup:
                    raise primary from cleanup
            raise

    @property
    def storage_generation(self):
        return self._storage_generation if not self._closing else -1

    @property
    def active_bank(self):
        return self._active

    observation_execution_contract_version = 1

    @property
    def observation_bank(self):
        """Ordered device reads may consume an enqueued candidate before acceptance.

        Consumers must gate dependent device writes on storage.execution_error.
        This does not publish the candidate or remove complete/accept obligations.
        """
        with self._scope():
            if self._pending is None:
                return self._active
            if self._pending['phase'] not in {'enqueued', 'completed'}:
                raise RuntimeError('observations require an enqueued or accepted execution')
            return self._pending['ticket'][1]

    @property
    def pending_execution(self):
        """Recovery ticket, including a failure during candidate preparation."""
        return self._pending["ticket"] if self._pending else None

    def execution_costs(self):
        """Explicit validation transfers; native Host internal work is additional."""
        with self._scope():
            bodies = self._body_observations
            contacts = self._contact_observations
            return {"validation_readbacks": self._validation.readbacks,
                    "validation_d2h_bytes": 4 * self._validation.readbacks,
                    "endpoint_kinematics_passes": 0 if bodies is None else bodies.kinematics_passes,
                    "body_observation_publications": 0 if bodies is None else bodies.publications,
                    "body_observation_device_bytes": 0 if bodies is None else bodies.allocated_bytes,
                    "contact_measurement_device_bytes": 0 if contacts is None else contacts.allocated_bytes,
                    "contact_measurement_extractions": 0 if contacts is None else contacts.extractions,
                    "contact_measurement_merges": 0 if contacts is None else contacts.merges,
                    "contact_measurement_copies": 0 if contacts is None else contacts.copies,
                    "body_measurement_evaluations": 0 if self._body_measurements is None else self._body_measurements.evaluations,
                    "body_measurement_copies": 0 if self._body_measurements is None else self._body_measurements.copies,
                    "body_measurement_device_bytes": 0 if self._body_measurements is None else self._body_measurements.allocated_bytes,
                    "site_measurement_device_bytes": sum(plan.allocated_bytes for plan in self._site_measurements),
                    "ray_queries": [plan.costs() for plan in self._ray_queries],
                    "free_body_update_device_bytes": 0 if self._free_body_updates is None else self._free_body_updates.allocated_bytes,
                    "stream_rebindings": self._stream_rebindings,
                    "step_graph_captures": self._graph_captures,
                    "step_graph_ready": self.step_graph_ready,
                    "step_graph_replays": self._graph_replays,
                    "native_direct_conditional_sync": True}

    @property
    def writable_bank(self):
        pending = self._pending
        return pending["ticket"][1] if pending and pending["phase"] == "prepared" else None

    def storage_binding(self, *, contract_version=1):
        """Borrow prepared fields/stream; no native model, Data or Host escapes."""
        with self._scope():
            if contract_version != 1:
                raise ValueError("unsupported runtime storage contract")
            return self._storage

    def _require_idle(self):
        if self._pending is not None:
            raise RuntimeError("an execution is pending; accept or discard it first")

    def _require_ticket(self, ticket, phase=None):
        if self._pending is None or self._pending["ticket"] is not ticket:
            raise RuntimeError("stale or foreign execution ticket")
        if phase is not None and self._pending["phase"] != phase:
            raise RuntimeError(f"execution requires {phase} phase")
        return ticket[1]

    def begin_execution(self):
        """Copy integration once; caller may now enqueue candidate field writes."""
        with self._scope():
            self._require_idle()
            self._execution_epoch += 1
            ticket = (self._execution_epoch, 1 - self._active)
            self._pending = {"ticket": ticket, "phase": "preparing"}
            # On failure retain the candidate until discard/close confirms completion.
            self._validation.clear(self._stream)
            self._candidate()
            if self._controls_binding is not None:
                self._controls_binding.copy(self._active, ticket[1], self._stream)
            self._pending["phase"] = "prepared"
            return ticket

    def enqueue_step(self, ticket, *, steps=1, execution_mode='direct'):
        """Advance one candidate for all substeps; no publication or host readback here.

        Native direct conditional loops may synchronize internally; this API
        separates publication and does not promise a nonblocking Host solver.
        """
        steps = _positive(steps, "steps")
        with self._scope():
            index = self._require_ticket(ticket, "prepared")
            if execution_mode not in ('direct', 'cuda_graph'):
                raise ValueError('unknown step execution mode')
            if execution_mode == 'cuda_graph' and (not self.step_graph_ready or steps != 4):
                raise RuntimeError('fixed four-step Graph must be prepared before replay')
            self._pending["phase"] = "enqueued"
            try:
                if execution_mode == 'cuda_graph':
                    self._step_graphs.replay(index)
                else:
                    self._enqueue_step_work(index, steps)
            except Exception:
                self._pending["phase"] = "failed"
                raise

    def _enqueue_step_work(self, index, steps, *, record=True):
        for substep in range(steps):
            if self._body_measurements is not None and substep == steps - 1:
                self._body_measurements.capture(index, self._stream)
            if self._contact_observations is not None and substep == steps - 1:
                self._contact_observations.capture_time(index, self._stream)
            self._native._step_runtime(self._handles[index])
            self._validation.enqueue(index, self._stream, include_extra=False)
        if self._body_observations is not None:
            self._body_observations.update(index, self._stream, integrated=True, record=record)
            self._validation.enqueue_extra(index, self._stream)
        if self._contact_observations is not None:
            self._contact_observations.update(index, self._stream, integrated=True, record=record)
        if self._body_measurements is not None:
            self._body_measurements.update(index, self._stream, integrated=True, record=record)

    def enqueue_forward(self, ticket):
        with self._scope():
            index = self._require_ticket(ticket, "prepared")
            self._pending["phase"] = "enqueued"
            try:
                self._native._forward_runtime(self._handles[index])
                if self._body_observations is not None:
                    self._body_observations.update(index, self._stream, integrated=False)
                self._validation.enqueue(index, self._stream)
                if self._contact_observations is not None:
                    self._contact_observations.update(index, self._stream)
                if self._body_measurements is not None:
                    self._body_measurements.update(index, self._stream)
            except Exception:
                self._pending["phase"] = "failed"
                raise

    def enqueue_update(self, ticket, *, refresh_state=False, compose_controls=False,
                       selected_worlds=None):
        """Accept prepared field edits without advancing time.

        Command-only updates retain existing solve-time observations using D2D
        copies. State edits perform one native forward. No CPU physics or full
        host snapshot is involved; failed candidates remain unpublished.

        Callers writing target_position/target_velocity must request
        compose_controls=True; qpos/qvel edits require refresh_state=True.
        These flags describe the prepared write plan, not automatic dirty tracking.
        """
        with self._scope():
            index = self._require_ticket(ticket, "prepared")
            if selected_worlds is not None:
                import tyx as tx
                if (not isinstance(selected_worlds, tx.Array) or
                    selected_worlds.dtype != tx.bool1 or
                    tuple(selected_worlds.shape) != (self.nworld,) or
                    tuple(selected_worlds.strides) != (1,) or
                    str(selected_worlds.device) != str(self._storage.device)):
                    raise ValueError("selected_worlds requires canonical-device bool[nworld]")
                # Keep caller-owned selection alive through complete/accept/discard.
                self._pending['selected_worlds'] = selected_worlds
            self._pending["phase"] = "enqueued"
            try:
                if compose_controls:
                    if self._controls_binding is None:
                        raise ValueError("this model has no managed scalar controls")
                    self._controls_binding.compose(index, self._stream)
                if refresh_state:
                    self._native._forward_runtime(self._handles[index])
                    if self._body_observations is not None:
                        self._body_observations.update(index, self._stream, integrated=False)
                else:
                    # Capacity counts belong to the preserved solve-time result,
                    # not a previous discarded candidate in this bank.
                    for name in (*OBSERVATION_FIELDS, "nacon", "ncollision", "nefc"):
                        if name not in INTEGRATION_FIELDS:
                            getattr(self._banks[index], name).assign(
                                getattr(self._banks[self._active], name), stream=self._stream
                            )
                    if self._body_observations is not None:
                        self._body_observations.copy(self._active, index, self._stream)
                self._validation.enqueue(index, self._stream)
                if self._contact_observations is not None:
                    if refresh_state:
                        self._contact_observations.update(index, self._stream, mask=selected_worlds)
                    else:
                        self._contact_observations.copy(self._active, index, self._stream)
                if refresh_state and selected_worlds is not None:
                    self._reset.preserve_observations(index, self._stream, selected_worlds)
                    if self._body_observations is not None:
                        self._body_observations.preserve(index, selected_worlds, self._stream)
                if self._body_measurements is not None:
                    if refresh_state:
                        self._body_measurements.update(index, self._stream, mask=selected_worlds)
                    else:
                        self._body_measurements.copy(self._active, index, self._stream)
            except Exception:
                self._pending["phase"] = "failed"
                raise

    def prepare_free_body_updates(self):
        """Cold FREE-root bindings: (name, body, q address, DOF address, local COM).

        No constrained child is admitted. Resources belong to this runtime and
        are retained on failed preparation until close completes successfully.
        """
        with self._scope():
            self._require_idle()
            if self._free_body_updates is None:
                from ._physolve_free import PreparedFreeBodies
                self._free_body_updates = PreparedFreeBodies(
                    self._cpu_model, self._banks, self._validation.flags,
                    self.nworld, self._device, self._storage,
                )
                self._free_body_updates.prepare(self._stream)
            if not self._free_body_updates.ready:
                raise RuntimeError('free-body preparation failed; close the retained runtime')
            return self._free_body_updates.bindings

    def enqueue_free_body_update(self, ticket, values, *, selected_bodies,
                                 components=('position', 'rotation', 'linear_velocity', 'angular_velocity'),
                                 component_flags=None, selected_worlds=None, compose_controls=False):
        """Borrow device [N,F,13] origin XYZ/XYZW/world-COM twist and bool[N,F].

        Unspecified components retain world pose/COM velocity, including when
        rotation changes. Inputs must remain unchanged until transaction end.
        Device validation precedes all candidate writes; one native forward
        follows. Completion rejects invalid values before publication.
        Additive per-body mode uses components=None and device int32[F] bits
        position=1, rotation=2, COM linear=4, world angular=8. Selected bodies
        require bits 1..15. Additional selected worlds are unioned for mixed
        state/control edits; compose_controls applies once in the same forward.
        """
        with self._scope():
            index = self._require_ticket(ticket, 'prepared')
            plan = self._free_body_updates
            if plan is None or not plan.ready:
                raise RuntimeError('prepare_free_body_updates is required before execution')
            mapping = dict(position=1, rotation=2, linear_velocity=4, angular_velocity=8)
            if component_flags is not None and components is not None:
                raise ValueError('component_flags requires components=None')
            if component_flags is None and (not isinstance(components, (tuple, list)) or not components or
                    any(not isinstance(c, str) or c not in mapping for c in components) or
                    len(set(components)) != len(components)):
                raise ValueError('free-body components must be unique declared field names')
            plan.check(values, selected_bodies, component_flags, selected_worlds)
            self._pending['free_body_inputs'] = (values, selected_bodies, component_flags, selected_worlds)
            try:
                plan.enqueue(index, values.flatten(), selected_bodies.flatten(),
                             -1 if component_flags is not None else sum(mapping[c] for c in components),
                             self._stream, component_flags, selected_worlds)
                self.enqueue_update(ticket, refresh_state=True, selected_worlds=plan.worlds,
                                    compose_controls=compose_controls)
            except Exception:
                self._pending['phase'] = 'failed'
                raise

    def enqueue_reset(self, ticket, mask=None, *, selected_worlds=None):
        with self._scope():
            index = self._require_ticket(ticket, "prepared")
            if selected_worlds is not None:
                import tyx as tx
                if mask is not None:
                    raise ValueError('host and device reset masks are mutually exclusive')
                if (not isinstance(selected_worlds, tx.Array) or
                    selected_worlds.dtype != tx.bool1 or
                    tuple(selected_worlds.shape) != (self.nworld,) or
                    tuple(selected_worlds.strides) != (1,) or
                    str(selected_worlds.device) != str(self._storage.device)):
                    raise ValueError('selected_worlds requires canonical-device bool[nworld]')
                self._pending['selected_worlds'] = selected_worlds
            if mask is not None:
                mask = np.asarray(mask)
                if mask.dtype != np.bool_ or mask.shape != (self.nworld,):
                    raise ValueError("reset mask must be bool[nworld]")
            self._pending["phase"] = "enqueued"
            try:
                selector = self._reset.restore(index, mask, self._stream, device_mask=selected_worlds)
                partial = selected_worlds is not None or (mask is not None and not mask.all())
                if self._controls_binding is not None:
                    self._controls_binding.reset(
                        index, selector,
                        self._stream,
                    )
                self._native._forward_runtime(self._handles[index])
                if self._body_observations is not None:
                    self._body_observations.update(index, self._stream, integrated=False)
                # Error bits latch before unselected observations overwrite qacc.
                self._validation.enqueue(index, self._stream)
                if self._contact_observations is not None:
                    self._contact_observations.update(
                        index, self._stream,
                        mask=selector if partial else None,
                    )
                if partial:
                    self._reset.preserve_observations(index, self._stream, selector)
                    if self._body_observations is not None:
                        self._body_observations.preserve(index, selector, self._stream)
                if self._body_measurements is not None:
                    self._body_measurements.update(index, self._stream, mask=selector if partial else None)
            except Exception:
                self._pending["phase"] = "failed"
                raise

    def complete_execution(self, ticket):
        """Wait for the canonical stream and check 4 bytes; do not publish."""
        with self._scope():
            index = self._require_ticket(ticket, "enqueued")
            try:
                self._validate(self._banks[index])
            except Exception:
                self._pending["phase"] = "failed"
                raise
            self._pending["phase"] = "completed"

    def accept_execution(self, ticket):
        """Publish only this completed candidate, once; no GPU work."""
        with self._scope():
            index = self._require_ticket(ticket, "completed")
            self._active = index
            self._pending = None

    def discard_execution(self, ticket):
        """Keep the accepted bank; failed completion retains all owners for retry."""
        with self._scope():
            self._require_ticket(ticket)
            self._complete()
            self._validation.clear(self._stream)
            self._pending = None

    def snapshot(self):
        """Independent read-only host arrays, explicitly synchronous D2H."""
        with self._scope():
            self._complete()
            data = self._banks[self._active]
            fields = {}
            for name in OBSERVATION_FIELDS:
                array = getattr(data, name).numpy(stream=self._stream)
                fields[name] = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(
                    array.shape
                )
            return MappingProxyType(fields)

    def close(self):
        """Complete before release; failed phase retains owners for retry."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            self._complete()
            for stream in self._stream_retained:
                self._synchronize_stream(stream)
            self._release_step_graphs()
            while self._ray_queries:
                self._ray_queries[-1].close()
                self._ray_queries.pop()
            while self._site_measurements:
                self._site_measurements[-1].close()
                self._site_measurements.pop()
            for index, handle in enumerate(self._handles):
                if handle is not None:
                    self._native._close_runtime(handle, True)
                    self._handles[index] = None
            while self._stream_retained:
                stream = self._stream_retained[-1]
                stream.destroy()
                self._stream_retained.pop()
                if stream is self._stream:
                    self._stream = None
                if stream is self._owned_stream:
                    self._owned_stream = None
            if self._stream is not None:
                self._stream.destroy()
                if self._stream is self._owned_stream:
                    self._owned_stream = None
                self._stream = None
            if self._owned_stream is not None:
                self._owned_stream.destroy()
                self._owned_stream = None
            self._banks.clear()
            if self._storage is not None:
                self._storage._release()
            self._storage = self._validation = self._pending = None
            self._storage_generation += 1
            self._reset = None
            self._controls_binding = None
            self._body_observations = None
            self._contact_observations = None
            self._body_measurements = None
            self._free_body_updates = None
            self._model = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
