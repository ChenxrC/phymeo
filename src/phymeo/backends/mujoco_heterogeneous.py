"""Independent native-model buckets with one whole heterogeneous publication.

Ordinary physics reuses each resident double bank and the original full receipt.
Boundary reset/restore may allocate replacements. This is not one fused kernel,
cross-world contact, zero-readback execution, or a GPU performance guarantee.
"""

import hashlib
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from threading import RLock
from types import MappingProxyType
from weakref import WeakKeyDictionary

import numpy as np

from ..builders.mujoco_compiler import MuJoCoCompilation
from ..training.resident_stepper import PhysicsStepFailure
from .mujoco_heterogeneous_checkpoint import MuJoCoTyxHeterogeneousCheckpoint, _capture
from .mujoco_runtime import (
    MuJoCoInitialState,
    MuJoCoRuntime,
    _frozen,
    _mujoco,
    _world_key,
    _world_keys,
)
from .mujoco_tyx_state import MuJoCoTyxIntegrationState

MAX_MODELS, MAX_WORLDS = 32, 32
_CONSTRUCTION_QUARANTINE = []


def _key(value):
    if type(value) is not str or not value or len(value) > 128:
        raise ValueError("model key must be a nonempty string of at most 128 characters")
    return value


def _capacity(value, name):
    if type(value) is not int or value < 0 or value > 2147483647:
        raise ValueError(name + " must be a nonnegative builtin i32 integer")
    return value


@dataclass(frozen=True)
class MuJoCoTyxModelSpec:
    key: str
    compilation: MuJoCoCompilation
    contact_capacity: int = 64
    constraint_capacity: int = 256

    def __post_init__(self):
        _key(self.key)
        if type(self.compilation) is not MuJoCoCompilation:
            raise TypeError("model spec requires a source-bound MuJoCoCompilation")
        _capacity(self.contact_capacity, "contact_capacity")
        _capacity(self.constraint_capacity, "constraint_capacity")


@dataclass(frozen=True)
class MuJoCoTyxWorldSpec:
    world: int
    model_key: str
    initial_state: MuJoCoInitialState | None = None

    def __post_init__(self):
        object.__setattr__(self, "world", _world_key(self.world))
        _key(self.model_key)
        state = self.initial_state
        if state is None:
            state = MuJoCoInitialState()
        if type(state) is not MuJoCoInitialState:
            raise TypeError("world spec requires MuJoCoInitialState")
        object.__setattr__(
            self,
            "initial_state",
            MuJoCoInitialState(state.q, state.v, act=state.act, time=state.time),
        )


@dataclass(frozen=True)
class MuJoCoTyxWorldLayout:
    world: int
    model_key: str
    bucket: int
    row: int
    nq: int
    nv: int
    nu: int
    na: int
    nbody: int
    model_sha256: str
    archive_sha256: str
    primary_sha256: str
    domain: str


@dataclass(frozen=True)
class MuJoCoTyxHeterogeneousStepReceipt:
    schema: str
    revision_before: int
    revision_after: int
    world_keys: tuple[int, ...]
    bucket_revisions: tuple[tuple[int, int], ...]
    receipts: Mapping
    states: Mapping
    accepted: Mapping


@dataclass(frozen=True)
class MuJoCoTyxHeterogeneousBoundaryReceipt:
    schema: str
    operation: str
    revision_before: int
    revision_after: int
    world_keys: tuple[int, ...]
    buckets: tuple[int, ...]
    cleanup_errors: tuple[str, ...] = ()


def _signature(spec, mode):
    source = spec.compilation.source
    return (
        source.primary_xml,
        source.archive_bytes,
        source.model_binary,
        source.mujoco_version,
        mode,
        spec.contact_capacity,
        spec.constraint_capacity,
    )


def _model_preflight(spec, mj):
    from .mujoco_tyx._model import compile_model

    model = spec.compilation.fresh_native_model()
    plan = compile_model(mj, model)
    if getattr(plan, "constraint_contact", False) and (
        spec.contact_capacity > 128 or spec.constraint_capacity > 512
    ):
        raise ValueError("T4 contact/constraint capacities cannot exceed 128/512")
    h = float(model.opt.timestep)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("heterogeneous timestep must be finite and positive")
    return h, tuple(int(getattr(model, field)) for field in ("nq", "nv", "nu", "na", "nbody")), plan


class MuJoCoTyxHeterogeneousBatch:
    """Own up to 32 exact-source buckets/worlds, with explicit expansion budgets."""

    def __init__(self, models, worlds, *, mode="cuda", cache_root=None, resource_budget=None):
        from .mujoco_resource_budget import MuJoCoTyxResourceBudget
        from .mujoco_tyx._allocation_manifest import allocation_manifest

        if resource_budget is not None and type(resource_budget) is not MuJoCoTyxResourceBudget:
            raise TypeError("resource_budget requires MuJoCoTyxResourceBudget")
        if mode not in ("interpret", "cuda"):
            raise ValueError("heterogeneous mode must be interpret or cuda")
        models, worlds = tuple(models), tuple(worlds)
        if not 1 <= len(models) <= MAX_MODELS or not 1 <= len(worlds) <= MAX_WORLDS:
            raise ValueError("heterogeneous batch requires 1..32 models and 1..32 worlds")
        if (len(models) > 8 or len(worlds) > 16) and resource_budget is None:
            raise ValueError("expanded heterogeneous batches require explicit resource_budget")
        if any(type(value) is not MuJoCoTyxModelSpec for value in models):
            raise TypeError("models must contain MuJoCoTyxModelSpec")
        if any(type(value) is not MuJoCoTyxWorldSpec for value in worlds):
            raise TypeError("worlds must contain MuJoCoTyxWorldSpec")
        if len({spec.key for spec in models}) != len(models):
            raise ValueError("duplicate heterogeneous model key")
        world_keys = _world_keys((world.world for world in worlds), nonempty=True)
        by_key = {spec.key: spec for spec in models}
        if {world.model_key for world in worlds} != set(by_key):
            raise ValueError("world declarations must use every model and no unknown model")
        mj = _mujoco()
        metadata = {spec.key: _model_preflight(spec, mj) for spec in models}
        h = metadata[models[0].key][0]
        if any(value[0] != h for value in metadata.values()):
            raise ValueError("heterogeneous models require exactly the same timestep")
        signatures, representatives, model_bucket = [], [], {}
        for spec in models:
            signature = _signature(spec, mode)
            bucket = next((i for i, old in enumerate(signatures) if old == signature), None)
            if bucket is None:
                bucket = len(signatures)
                signatures.append(signature)
                representatives.append(spec)
            model_bucket[spec.key] = bucket
        members = tuple(
            tuple(world for world in worlds if model_bucket[world.model_key] == bucket)
            for bucket in range(len(signatures))
        )
        manifests = tuple(
            allocation_manifest(
                metadata[spec.key][2],
                len(rows),
                metadata[spec.key][1][4],
                (spec.contact_capacity, spec.constraint_capacity),
            )
            for spec, rows in zip(representatives, members, strict=True)
        )
        self._resource_budget, self._allocation_manifests = resource_budget, manifests
        self._lock = RLock() if resource_budget is None else resource_budget._lock
        self._busy, self._closed = False, False
        self._revision, self._last_step = 0, None
        self._mode, self._cache_root, self._h = mode, cache_root, h
        self._world_keys, self._worlds, self._models = world_keys, worlds, models
        self._buckets, self._retired, self._cleanup_errors = (), [], ()
        self._workspaces = set()
        self._counter_errors = ()
        self._observed, self._device_totals = WeakKeyDictionary(), {}
        self._stats = dict(
            completed_steps=0,
            accepted_world_steps=0,
            failed_batches=0,
            control_submissions=0,
            control_refusals=0,
            boundary_operations=0,
            boundary_refusals=0,
            boundary_clones=0,
            bucket_physics_prepares=0,
        )
        built = []
        try:
            admission = (
                resource_budget.admit(sum(m.payload_bytes for m in manifests))
                if resource_budget is not None and mode == "cuda"
                else nullcontext()
            )
            with admission:
                for spec, rows in zip(representatives, members, strict=True):
                    owner = MuJoCoRuntime(
                        spec.compilation,
                        world_keys=tuple(row.world for row in rows),
                        initial_states={row.world: row.initial_state for row in rows},
                        mode=mode,
                        implementation="tyx",
                        contact_capacity=spec.contact_capacity,
                        constraint_capacity=spec.constraint_capacity,
                        cache_root=cache_root,
                        resource_budget=resource_budget,
                    )
                    built.append(owner)
                    self._observe_owner(owner)
            self._buckets = tuple(built)
            layout = {}
            for world in worlds:
                bucket = model_bucket[world.model_key]
                owner = self._buckets[bucket]
                spec = by_key[world.model_key]
                source = spec.compilation.source
                layout[world.world] = MuJoCoTyxWorldLayout(
                    world.world,
                    world.model_key,
                    bucket,
                    tuple(row.world for row in members[bucket]).index(world.world),
                    *metadata[world.model_key][1],
                    hashlib.sha256(source.model_binary).hexdigest(),
                    hashlib.sha256(source.archive_bytes).hexdigest(),
                    hashlib.sha256(source.primary_xml).hexdigest(),
                    owner.capabilities["domain"],
                )
            self._layout = MappingProxyType(layout)
            self._capabilities = MappingProxyType(
                dict(
                    backend="mujoco-tyx-heterogeneous-" + mode,
                    mode=mode,
                    implementation="tyx",
                    bucket_count=len(built),
                    world_count=len(worlds),
                    max_models=MAX_MODELS,
                    max_worlds=MAX_WORLDS,
                    publication="whole-heterogeneous-batch-atomic",
                    scheduling="exact-source-homogeneous-buckets",
                    fused_kernel=False,
                    host_diagnostics=True,
                    device_native_state=True,
                    device_native_controls=True,
                    device_endpoint_bodies=True,
                    device_contact_tokens=False,
                    cross_world_contacts=False,
                    scale_qualified=False,
                    explicit_resource_budget=resource_budget is not None,
                    models=MappingProxyType(
                        {
                            spec.key: self._buckets[model_bucket[spec.key]].capabilities
                            for spec in models
                        }
                    ),
                )
            )
            self._binding = MappingProxyType(
                dict(
                    schema="mujoco-tyx-heterogeneous-checkpoint-v1",
                    mode=mode,
                    timestep=h,
                    world_keys=world_keys,
                    worlds=tuple(
                        (key, layout[key].model_key, layout[key].bucket, layout[key].row)
                        for key in world_keys
                    ),
                    buckets=tuple(
                        MappingProxyType(
                            dict(
                                world_keys=owner.world_keys,
                                model_sha256=layout[owner.world_keys[0]].model_sha256,
                                archive_sha256=layout[owner.world_keys[0]].archive_sha256,
                                primary_sha256=layout[owner.world_keys[0]].primary_sha256,
                                mujoco_version=representatives[i].compilation.source.mujoco_version,
                                capacities=(
                                    representatives[i].contact_capacity,
                                    representatives[i].constraint_capacity,
                                ),
                                dimensions=metadata[representatives[i].key][1],
                                domain=owner.capabilities["domain"],
                            )
                        )
                        for i, owner in enumerate(self._buckets)
                    ),
                )
            )
        except BaseException as primary:
            self._buckets = ()
            failures = self._close_owners(built)
            self._closed = True
            if failures:
                _CONSTRUCTION_QUARANTINE.append(self)
                raise BaseExceptionGroup(
                    "heterogeneous construction and cleanup failed; owner retained",
                    [primary, *failures],
                ) from None
            raise

    def _check(self):
        if self._closed:
            raise RuntimeError("heterogeneous batch is closed")
        if self._busy:
            raise RuntimeError("heterogeneous batch transaction is already active")

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._check()
            self._busy = True
            try:
                with ExitStack() as stack:
                    for owner in self._buckets:
                        stack.enter_context(owner._lock)
                    yield
            finally:
                self._busy = False

    def _observe_owner(self, owner):
        current = dict(owner.counters)
        previous = self._observed.get(owner, {})
        for key, value in current.items():
            if type(value) is int:
                old = previous.get(key, 0)
                if value < old:
                    raise RuntimeError("original runtime operational counter decreased")
                self._device_totals[key] = self._device_totals.get(key, 0) + value - old
        self._observed[owner] = current
        return MappingProxyType(current)

    def _close_owners(self, owners):
        failures = []
        for owner in owners:
            # Counter observation must never prevent the real close attempt.
            if owner not in self._retired:
                try:
                    self._observe_owner(owner)
                except BaseException as error:
                    failures.append(error)
                    self._counter_errors += (repr(error),)
            try:
                owner.close()
            except BaseException as error:
                failures.append(error)
                if owner not in self._retired:
                    self._retired.append(owner)
            else:
                if owner in self._retired:
                    self._retired.remove(owner)
        if failures:
            self._cleanup_errors += tuple(repr(error) for error in failures)
        return failures

    def _select(self, keys):
        selected = self._world_keys if keys is None else _world_keys(keys)
        if not set(selected).issubset(self._world_keys):
            raise ValueError("selected world does not belong to this heterogeneous batch")
        return selected

    def _before_commit(self, operation, pending):
        """Internal fault seam; no user callback is accepted by the public API."""

    @property
    def world_keys(self):
        return self._world_keys

    @property
    def timestep(self):
        return self._h

    @property
    def layout(self):
        return self._layout

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def checkpoint_revision(self):
        with self._lock:
            self._check()
            return self._revision

    @property
    def last_step(self):
        with self._lock:
            self._check()
            return self._last_step

    @property
    def cleanup_errors(self):
        with self._lock:
            return self._cleanup_errors

    @property
    def counters(self):
        with self._lock:
            if self._busy:
                raise RuntimeError("heterogeneous batch transaction is active")
            snapshots = (
                () if self._closed else tuple(self._observe_owner(owner) for owner in self._buckets)
            )
            return MappingProxyType(
                {
                    **self._device_totals,
                    **self._stats,
                    "bucket_counters": snapshots,
                    "observed_device_counters": MappingProxyType(dict(self._device_totals)),
                    "retained_cleanup_owners": len(self._retired),
                    "counter_observation_errors": self._counter_errors,
                    "counter_scope": "public-runtime-observed; boundary constructors may replace internal owners",
                }
            )

    def resource_snapshot(self):
        """Budget spans live, candidate and ambiguously retained CUDA arrays."""
        with self._lock:
            if self._busy:
                raise RuntimeError("heterogeneous batch transaction is active")
            return MappingProxyType(
                dict(
                    schema="mujoco-tyx-heterogeneous-resources-v1",
                    closed=self._closed,
                    manifests=self._allocation_manifests,
                    manifest_payload_bytes=sum(m.payload_bytes for m in self._allocation_manifests),
                    budget=None
                    if self._resource_budget is None
                    else self._resource_budget.snapshot(),
                    buckets=tuple(owner.resource_snapshot() for owner in self._buckets),
                    retained_owner_count=len(self._retired),
                    device_workspaces=tuple(dict(w.counters) for w in self._workspaces),
                    completeness="Solver-bank CUDA payload only; HOST/native/context/JIT/external controls/workspaces unknown",
                )
            )

    def prepare_device_workspace(self, *, cache_root=None, include_bodies=False):
        """Own a padded native-state workspace; no physics or CPU forward."""
        from .mujoco_heterogeneous_io import MuJoCoTyxHeterogeneousWorkspace

        return MuJoCoTyxHeterogeneousWorkspace(
            self, cache_root=cache_root, include_bodies=include_bodies
        )

    def set_controls_from_device(
        self,
        ctrl,
        *,
        qfrc_applied=None,
        xfrc_applied=None,
        world_keys,
        expected_revision,
        producer_stream=None,
    ):
        """Copy/audit valid native columns, ignoring padding, then publish all rows.

        Controls are native values, without policy scaling. Omitted forces are
        fresh zeros. Physics stepping remains a separate atomic transaction.
        """
        from .mujoco_heterogeneous_io import submit_controls

        return submit_controls(
            self, (ctrl, qfrc_applied, xfrc_applied), world_keys, expected_revision, producer_stream
        )

    def reset_states(self, states, *, expected_revision):
        """Reset selected native states atomically; original templates stay intact."""
        with self._transaction():
            if type(expected_revision) is not int or expected_revision != self._revision:
                raise ValueError("stale heterogeneous reset revision")
            if not isinstance(states, Mapping):
                raise TypeError("reset states require a world-key mapping")
            selected = self._select(states)
            prepared = {}
            for key in selected:
                packet = states[key]
                if type(packet) is not MuJoCoInitialState:
                    raise TypeError("reset states require MuJoCoInitialState")
                if packet.time != 0.0:
                    raise ValueError("explicit episode reset time must be zero")
                prepared[key] = MuJoCoInitialState(packet.q, packet.v, act=packet.act)
            return self._boundary("reset-states", selected, prepared)

    def set_controls(self, mapping):
        with self._transaction():
            if not isinstance(mapping, Mapping) or set(mapping) != set(self._world_keys):
                raise ValueError("controls must cover exactly the heterogeneous world keys")
            _world_keys(mapping)
            drafts = []
            try:
                for owner in self._buckets:
                    drafts.append(
                        owner._prepare_controls_locked(
                            {key: mapping[key] for key in owner.world_keys}
                        )
                    )
                self._before_commit("controls", tuple(drafts))
                for owner, draft in zip(self._buckets, drafts, strict=True):
                    owner._validate_controls_draft_locked(draft)
                revision = self._revision + 1
                counters = {
                    **self._stats,
                    "control_submissions": self._stats["control_submissions"] + 1,
                }
            except BaseException:
                for owner in self._buckets:
                    owner._discard_controls_draft_locked()
                self._stats["control_refusals"] += 1
                raise
            for owner, draft in zip(self._buckets, drafts, strict=True):
                owner._commit_controls_draft_locked(draft)
            self._revision, self._stats = revision, counters
            for owner in self._buckets:
                owner._finish_controls_cleanup_locked()
                try:
                    self._observe_owner(owner)
                except BaseException as error:
                    # Controls already committed. Keep a separate diagnostic;
                    # do not report an accepted action as a failed transaction.
                    self._counter_errors += (repr(error),)

    @staticmethod
    def _draft_states(owner, draft):
        result = {}
        for key in owner.world_keys:
            data, row = draft.data[key], draft.receipts[key]
            result[key] = MuJoCoTyxIntegrationState(
                world=key,
                q=_frozen(data.qpos),
                v=_frozen(data.qvel),
                act=_frozen(data.act),
                ctrl=_frozen(data.ctrl),
                time=float(data.time),
                tick=draft.ticks[key],
                qfrc_applied=_frozen(data.qfrc_applied),
                xfrc_applied=_frozen(data.xfrc_applied),
                held_control=row.control,
                last_step=row,
            )
        return result

    def step(self, h):
        with self._transaction():
            if not np.isfinite(h) or float(h) != self._h:
                raise ValueError("step h must equal every compiled timestep")
            drafts = []
            try:
                for owner in self._buckets:
                    self._stats["bucket_physics_prepares"] += 1
                    drafts.append(owner._prepare_step_locked(h))
                    self._observe_owner(owner)
                receipts, states = {}, {}
                for owner, draft in zip(self._buckets, drafts, strict=True):
                    receipts.update(draft.receipts)
                    states.update(self._draft_states(owner, draft))
                accepted = MappingProxyType(dict.fromkeys(self._world_keys, True))
                publication = MuJoCoTyxHeterogeneousStepReceipt(
                    "mujoco-tyx-heterogeneous-step-v1",
                    self._revision,
                    self._revision + 1,
                    self._world_keys,
                    tuple((draft.revision, draft.revision + 1) for draft in drafts),
                    MappingProxyType({key: receipts[key] for key in self._world_keys}),
                    MappingProxyType({key: states[key] for key in self._world_keys}),
                    accepted,
                )
                self._before_commit("step", publication)
                for owner, draft in zip(self._buckets, drafts, strict=True):
                    owner._validate_step_draft_locked(draft)
                counters = {
                    **self._stats,
                    "completed_steps": self._stats["completed_steps"] + 1,
                    "accepted_world_steps": self._stats["accepted_world_steps"]
                    + len(self._world_keys),
                }
            except BaseException as error:
                failures = []
                for owner in self._buckets:
                    try:
                        owner._discard_step_draft_locked()
                    except BaseException as cleanup:
                        failures.append(cleanup)
                    try:
                        self._observe_owner(owner)
                    except BaseException as observation:
                        self._counter_errors += (repr(observation),)
                self._stats["failed_batches"] += 1
                if failures:
                    error = BaseExceptionGroup(
                        "heterogeneous preparation and discard failed", [error, *failures]
                    )
                if not isinstance(error, Exception):
                    raise error
                raise PhysicsStepFailure(
                    "heterogeneous candidate failed before any bucket publication",
                    accepted_worlds=dict.fromkeys(self._world_keys, False),
                ) from error
            for owner, draft in zip(self._buckets, drafts, strict=True):
                owner._commit_step_draft_locked(draft)
            self._last_step, self._revision, self._stats = (
                publication,
                publication.revision_after,
                counters,
            )
            return accepted

    def _snapshot(self, keys, full):
        with self._lock:
            self._check()
            selected = self._select(keys)
            gathered = {}
            for owner in self._buckets:
                subset = tuple(key for key in owner.world_keys if key in selected)
                if subset:
                    gathered.update(
                        owner.snapshot(subset) if full else owner.integration_snapshot(subset)
                    )
                    self._observe_owner(owner)
            return MappingProxyType({key: gathered[key] for key in selected})

    def snapshot(self, keys=None):
        return self._snapshot(keys, True)

    def integration_snapshot(self, keys=None):
        return self._snapshot(keys, False)

    def _publication_metadata(self):
        row = self._last_step
        if row is None:
            return None
        return dict(
            schema=row.schema,
            revision_before=row.revision_before,
            revision_after=row.revision_after,
            world_keys=row.world_keys,
            bucket_revisions=row.bucket_revisions,
        )

    def checkpoint(self):
        with self._transaction():
            leaves = tuple(owner.checkpoint() for owner in self._buckets)
            result = _capture(self._binding, self._revision, leaves, self._publication_metadata())
            for owner in self._buckets:
                self._observe_owner(owner)
            return result

    def _clone_bucket(self, owner):
        self._stats["boundary_clones"] += 1
        return owner.clone()

    def _boundary(self, operation, selected, leaves=None, publication=None):
        indices = tuple(
            i
            for i, owner in enumerate(self._buckets)
            if operation == "restore" or set(owner.world_keys).intersection(selected)
        )
        prepared, peak = 0, 0
        for i in indices:
            size = self._allocation_manifests[i].payload_bytes
            peak = max(peak, prepared + 2 * size)
            prepared += size
        admission = (
            self._resource_budget.admit(max(peak, prepared))
            if self._resource_budget is not None and self._mode == "cuda"
            else nullcontext()
        )
        admitted = False
        try:
            with admission:
                admitted = True
                return self._boundary_admitted(operation, selected, leaves, publication)
        except BaseException:
            if not admitted:
                self._stats["boundary_refusals"] += 1
            raise

    def _boundary_admitted(self, operation, selected, leaves=None, publication=None):
        replacements, pending = {}, []
        indices = tuple(
            i
            for i, owner in enumerate(self._buckets)
            if operation == "restore" or set(owner.world_keys).intersection(selected)
        )
        if not indices:
            return MuJoCoTyxHeterogeneousBoundaryReceipt(
                "mujoco-tyx-heterogeneous-boundary-v1",
                operation,
                self._revision,
                self._revision,
                (),
                (),
            )
        try:
            for i in indices:
                owner = self._buckets[i]
                original = leaves[i] if operation == "restore" else owner.checkpoint()
                candidate = self._clone_bucket(owner)
                pending.append(candidate)
                self._observe_owner(candidate)
                candidate.restore(original, expected_revision=candidate.checkpoint_revision)
                if operation == "reset":
                    candidate.reset(tuple(key for key in owner.world_keys if key in selected))
                elif operation == "reset-states":
                    candidate.reset_states(
                        {key: leaves[key] for key in owner.world_keys if key in selected},
                        expected_revision=candidate.checkpoint_revision,
                    )

                self._observe_owner(candidate)
                replacements[i] = candidate
            replacement_buckets = tuple(
                replacements.get(i, owner) for i, owner in enumerate(self._buckets)
            )
            restored_publication = None
            if publication is not None:
                states = {}
                for owner in replacement_buckets:
                    states.update(owner.integration_snapshot())
                if any(states[key].last_step is None for key in self._world_keys):
                    raise ValueError("checkpoint publication lacks a complete leaf receipt")
                # Held controls may have changed after the saved last step. The
                # historical aggregate retains the control from its actual step.
                states = {
                    key: replace(value, held_control=value.last_step.control)
                    for key, value in states.items()
                }
                restored_publication = MuJoCoTyxHeterogeneousStepReceipt(
                    publication["schema"],
                    publication["revision_before"],
                    publication["revision_after"],
                    self._world_keys,
                    tuple(tuple(row) for row in publication["bucket_revisions"]),
                    MappingProxyType({key: states[key].last_step for key in self._world_keys}),
                    MappingProxyType({key: states[key] for key in self._world_keys}),
                    MappingProxyType(dict.fromkeys(self._world_keys, True)),
                )
            result = MuJoCoTyxHeterogeneousBoundaryReceipt(
                "mujoco-tyx-heterogeneous-boundary-v1",
                operation,
                self._revision,
                self._revision + 1,
                tuple(selected),
                indices,
            )
            retired = tuple(self._buckets[i] for i in indices)
            counters = {
                **self._stats,
                "boundary_operations": self._stats["boundary_operations"] + 1,
            }
            self._before_commit(operation, result)
        except BaseException as primary:
            self._stats["boundary_refusals"] += 1
            failures = self._close_owners(pending)
            if failures:
                raise BaseExceptionGroup(
                    "heterogeneous boundary preparation and cleanup failed", [primary, *failures]
                ) from None
            raise
        self._buckets, self._last_step = replacement_buckets, restored_publication
        self._revision, self._stats = result.revision_after, counters
        failures = self._close_owners(retired)
        object.__setattr__(result, "cleanup_errors", tuple(repr(error) for error in failures))
        return result

    def reset(self, keys=None):
        with self._transaction():
            return self._boundary("reset", self._select(keys))

    def restore(self, checkpoint, *, expected_revision):
        with self._transaction():
            if type(expected_revision) is not int or expected_revision != self._revision:
                raise ValueError("stale heterogeneous destination revision")
            if type(checkpoint) is not MuJoCoTyxHeterogeneousCheckpoint:
                raise TypeError("restore requires MuJoCoTyxHeterogeneousCheckpoint")
            leaves, publication = checkpoint._read(self._binding)
            if len(leaves) != len(self._buckets):
                raise ValueError("heterogeneous checkpoint bucket binding")
            return self._boundary("restore", self._world_keys, leaves, publication)

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError("cannot close an active heterogeneous transaction")
            if self._closed and not self._retired and not self._workspaces:
                return
            owners = tuple(dict.fromkeys((*self._retired, *self._buckets)))
            failures = []
            for workspace in tuple(self._workspaces):
                try:
                    workspace.close()
                except BaseException as error:
                    failures.append(error)
            failures.extend(self._close_owners(owners))
            self._buckets, self._closed = (), True
            if failures:
                raise BaseExceptionGroup("heterogeneous cleanup retains failed owners", failures)

    def __enter__(self):
        with self._lock:
            self._check()
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if exc_value is not None:
                raise BaseExceptionGroup(
                    "heterogeneous body and cleanup failed", [exc_value, cleanup]
                ) from None
            raise
