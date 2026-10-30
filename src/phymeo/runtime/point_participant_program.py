"""Public particle participant stages with one robot-free terminal batch.

Factories are borrowed; each workspace owns its participant instances. No
configuration or velocity equation is implemented by this adapter.
"""

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields
from types import MappingProxyType

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.scene import BackendContext, HistoryField, SceneState, StageMetrics, fingerprint
from .program import ProgramCapabilities, own_diagnostics
from .registry import StateFamily, StateRegistry, copy_history
from .terminal_impulse import _batch_requests, _own_requests


def _capabilities(value):
    return {
        item.name: tuple(sorted(entry)) if isinstance(entry, frozenset) else entry
        for item in fields(value)
        for entry in (getattr(value, item.name),)
    }


class PointParticipantProgram:
    """Bind stateless particle factories without introducing a robot domain."""

    def __init__(self, scene, participants, *, registry=None, history_fields=()):
        scene.assert_unchanged()
        if not isinstance(participants, Mapping):
            raise TypeError("participants must be a domain mapping")
        bindings = {binding.name: binding for binding in scene.domains}
        if not bindings or set(participants) != set(bindings):
            raise ValueError("participants must match every nonempty scene domain")
        if registry is not None and not isinstance(registry, StateRegistry):
            raise TypeError("registry must be a StateRegistry")
        schema = tuple(history_fields)
        if any(not isinstance(entry, HistoryField) for entry in schema):
            raise TypeError("history schema requires HistoryField values")
        if len({entry.name for entry in schema}) != len(schema):
            raise ValueError("duplicate history field")
        owned_registry = StateRegistry(defaults=False) if registry is None else registry.copy()
        families, ordered, caps_all = {}, {}, []
        for name, binding in bindings.items():
            factory = participants[name]
            family, caps = factory.family, factory.capabilities
            if not isinstance(family, StateFamily) or not isinstance(caps, ProgramCapabilities):
                raise TypeError("participant requires typed family and capabilities")
            if binding.family != "particles" or family.kind != binding.family:
                raise ValueError("point participant adapter supports particles only")
            if factory.point_role != "node" or fingerprint(factory.model) != fingerprint(
                binding.model
            ):
                raise ValueError("participant model and node role must match its domain")
            if "float64" not in caps.dtypes or family.kind not in caps.families:
                raise ValueError("participant registered precision/family unsupported")
            if caps.history != "none" or caps.provider_kind != "mathematical":
                raise ValueError("adapter requires stateless mathematical participants")
            if caps_all and caps.backend != caps_all[0].backend:
                raise ValueError("participants require one backend")
            if family.kind in families and families[family.kind] != family:
                raise ValueError("participants disagree on the state family")
            if registry is not None and owned_registry.get(family.kind) != family:
                raise ValueError("registry and participant family disagree")
            if registry is None and family.kind not in families:
                owned_registry.register(family)
            families[family.kind], ordered[name] = family, factory
            caps_all.append(caps)
        common = set(caps_all[0].features)
        for caps in caps_all[1:]:
            common.intersection_update(caps.features)
        self.scene, self.participants = scene, MappingProxyType(ordered)
        self.registry, self.history_fields = owned_registry, schema
        self.capabilities = ProgramCapabilities(
            caps_all[0].backend,
            {"float64"},
            {"particles"},
            common | {"staged_participants", "terminal_point_impulses"},
            history="explicit" if schema else "none",
        )
        self._signature = fingerprint(self.problem_signature())

    def problem_signature(self):
        return dict(
            law="public-particle-participant-terminal-v1",
            scene=self.scene.identity,
            participants=tuple(
                (
                    name,
                    factory.point_role,
                    factory.problem_signature(),
                    _capabilities(factory.capabilities),
                )
                for name, factory in self.participants.items()
            ),
            history=tuple(
                (entry.name, entry.shape, entry.dtype.str) for entry in self.history_fields
            ),
            capabilities=_capabilities(self.capabilities),
        )

    def create_workspace(self, scene, state, inputs, h, settings, context):
        self.scene.assert_unchanged()
        if (
            scene.identity != self.scene.identity
            or fingerprint(self.problem_signature()) != self._signature
        ):
            raise ValueError("bound particle program changed")
        for binding in self.scene.domains:
            factory = self.participants[binding.name]
            if factory.family != self.registry.get(binding.family) or fingerprint(
                factory.model
            ) != fingerprint(binding.model):
                raise ValueError("bound participant family/model changed")
        if not isinstance(context, BackendContext):
            raise TypeError("expected BackendContext")
        if context.backend != self.capabilities.backend or context.dtype != np.dtype(np.float64):
            raise ValueError("particle participant backend/precision unsupported")
        original, original_inputs = self.registry.snapshot(
            scene, state, inputs, context, self.history_fields
        )
        return _PointWorkspace(self, original, original_inputs, h, settings, context)

    def close(self):
        """Borrowed factories remain usable by other replay adapters."""


class _PointWorkspace:
    def __init__(self, owner, state, inputs, h, settings, context):
        self.owner, self.context = owner, context
        self._history = copy_history(state.history, owner.history_fields)
        self._closed = self._failed = self._predicted = self._prepared = self._terminal = False
        self._participants = {}
        with ExitStack() as resources:
            for name, factory in owner.participants.items():
                child = factory.create_workspace(
                    name, state.domains[name], inputs.domains[name], h, settings, context
                )
                resources.callback(child.close)
                self._participants[name] = child
            self._resources = resources.pop_all()

    def _open(self):
        if self._closed:
            raise RuntimeError("point participant workspace is closed")
        if self._failed:
            raise RuntimeError("point participant candidate is poisoned")

    def _reject(self, message):
        self._failed = True
        raise RuntimeError(message)

    def _call(self, function, *args):
        self._open()
        try:
            return function(*args)
        except BaseException:
            self._failed = True
            raise

    def _all(self, operation):
        for child in self._participants.values():
            self._call(getattr(child, operation))

    def predict(self):
        self._open()
        if self._predicted:
            self._reject("prediction must execute once")
        self._all("predict")
        self._predicted = True

    def _position_ready(self):
        self._open()
        if not self._predicted or self._prepared:
            self._reject("position stage requires prediction before prepare_velocity")

    def position_sweep(self):
        self._position_ready()
        self._all("position_sweep")

    def _metrics(self, phase):
        values = []
        residuals = {}
        for name, child in self._participants.items():
            value = self._call(getattr(child, phase + "_metrics"))
            if not isinstance(value, StageMetrics):
                self._failed = True
                raise SolverFailure("participant supplied invalid stage metrics")
            values.append(value)
            residuals.update({name + "/" + key: entry for key, entry in value.residuals.items()})
        return StageMetrics(
            residuals,
            geometric_error=max(value.geometric_error for value in values),
            penetration=max(value.penetration for value in values),
        )

    def position_metrics(self):
        self._position_ready()
        return self._metrics("position")

    def prepare_velocity(self):
        self._position_ready()
        self._all("prepare_velocity")
        self._prepared = True

    def _velocity_ready(self):
        self._open()
        if not self._prepared:
            self._reject("velocity stage requires prepare_velocity")

    def velocity_sweep(self):
        self._velocity_ready()
        if self._terminal:
            self._reject("velocity sweep cannot follow terminal application")
        self._all("velocity_sweep")

    def velocity_metrics(self):
        self._velocity_ready()
        return self._metrics("velocity")

    def apply_point_velocity_impulses(self, requests):
        self._velocity_ready()
        if self._terminal:
            self._reject("terminal point impulses may execute once")
        try:
            owned = _own_requests(requests)
            for request in owned:
                if request.domain not in self._participants:
                    raise ValueError("terminal domain is not a particle participant")
                self._participants[request.domain].point_port(request.indices, request.local_points)
            batches = _batch_requests(owned)
            self._terminal = True
            for request in batches:
                if len(request.indices):
                    self._participants[request.domain].apply_velocity_impulses(
                        request.indices, request.local_points, request.impulses
                    )
        except BaseException:
            self._failed = True
            raise

    def result(self):
        self._velocity_ready()

        def snapshot():
            value = SceneState(
                {name: child.result() for name, child in self._participants.items()}, self._history
            )
            return self.owner.registry.clone_state(
                self.owner.scene, value, self.context, self.owner.history_fields
            )

        return self._call(snapshot)

    def diagnostics(self):
        self._velocity_ready()

        def snapshot():
            return own_diagnostics(
                dict(
                    participants={
                        name: child.diagnostics() for name, child in self._participants.items()
                    },
                    point_participant=dict(terminal_applied=self._terminal),
                )
            )

        return self._call(snapshot)

    def close(self):
        if not self._closed:
            self._closed = True
            self._resources.close()
