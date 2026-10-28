"""Re-evaluate held input loads over one frozen interval without accumulating trials."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..core._arrays import float_array
from ..core.capabilities import SolverFailure
from ..core.scene import BackendContext, SceneInputs, StepSettings
from ..core.validation import finite_real, validate_dt
from .program import SceneStepResult, own_diagnostics
from .registry import StateRegistry
from .scene_stepper import SceneStepper


def _same_array(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)


def _check_snapshot_copy(scene, registry, state, inputs, copied_state, copied_inputs):
    for binding in scene.domains:
        family = registry.get(binding.family)
        for kind, before, after, names in (
            ("state", state, copied_state, family.state_fields),
            ("input", inputs, copied_inputs, family.input_fields),
        ):
            for name in names:
                if not _same_array(
                    getattr(before.domains[binding.name], name),
                    getattr(after.domains[binding.name], name),
                ):
                    raise ValueError(
                        f"registered {kind} snapshot copy changed {binding.name}.{name}"
                    )


@dataclass(frozen=True)
class ReplayTrial:
    """Trial data; inner acceptance does not certify an outer contact law.

    evaluate() returns owned data. Direct dataclass construction borrows arguments.

    integrated_loads contains the actual represented input difference * duration,
    in native input units times seconds. A rejected result commits no load/state.
    """

    result: SceneStepResult
    integrated_loads: Mapping[str, Mapping[str, np.ndarray]]
    effective_inputs: SceneInputs
    duration: float


class SceneLoadReplay:
    """Repeat a program from the same complete original state/history snapshot.

    Explicit channels declare which registered inputs are rates of integrated
    physical loads. Their held addition is integral / duration. This class does
    not solve contact, choose geometry or perform a second velocity impulse.
    """

    def __init__(
        self,
        scene,
        program,
        state,
        inputs,
        dt,
        *,
        channels,
        settings=None,
        context=None,
        registry=None,
        verifier=None,
    ):
        context = BackendContext() if context is None else context
        if not isinstance(context, BackendContext) or context.dtype != np.dtype(np.float64):
            raise TypeError("load replay requires a float64 BackendContext")
        settings = StepSettings(substeps=1) if settings is None else settings
        if not isinstance(settings, StepSettings):
            raise TypeError("expected StepSettings")
        self._duration = float(validate_dt(finite_real(dt, "duration"), context.dtype))
        if registry is not None and not isinstance(registry, StateRegistry):
            raise TypeError("expected StateRegistry")
        registry = StateRegistry() if registry is None else registry.copy()
        schema = tuple(program.history_fields)
        initial, original_inputs = registry.snapshot(scene, state, inputs, context, schema)
        _check_snapshot_copy(scene, registry, state, inputs, initial, original_inputs)
        if not isinstance(channels, Mapping):
            raise TypeError("channels must map domains to registered input fields")
        bindings = {domain.name: domain for domain in scene.domains}
        if not set(channels) <= set(bindings):
            raise ValueError("channel domain is not registered in this scene")
        owned_channels = {}
        for domain, names in channels.items():
            if isinstance(names, (str, bytes)):
                raise TypeError("channel fields must be a collection of names")
            names = tuple(names)
            registered = registry.get(bindings[domain].family).input_fields
            if len(set(names)) != len(names) or not set(names) <= set(registered):
                raise ValueError("channels must be unique registered input fields")
            owned_channels[domain] = names
        self._scene, self._context, self._registry, self._schema = scene, context, registry, schema
        self._initial, self._original_inputs = initial, original_inputs
        self._channels = MappingProxyType(owned_channels)
        self._runtime = SceneStepper(scene, program, settings, context, registry, verifier)
        self._closed = False

    @property
    def duration(self):
        return self._duration

    def snapshot(self):
        return self._registry.clone_state(self._scene, self._initial, self._context, self._schema)

    def _compose(self, loads):
        if not isinstance(loads, Mapping):
            raise TypeError("integrated loads must be a domain mapping")
        if not set(loads) <= set(self._channels):
            raise ValueError("integrated load domain has no declared channels")
        for domain, values in loads.items():
            if not isinstance(values, Mapping):
                raise TypeError("domain integrated loads must be a field mapping")
            if not set(values) <= set(self._channels[domain]):
                raise ValueError("integrated load field is not a declared channel")
        inputs, represented = {}, {}
        for binding in self._scene.domains:
            domain = binding.name
            family = self._registry.get(binding.family)
            original = self._original_inputs.domains[domain]
            fields = {name: getattr(original, name).copy() for name in family.input_fields}
            requests = {}
            for name in self._channels.get(domain, ()):
                before = fields[name]
                requested = float_array(
                    loads.get(domain, {}).get(name, np.zeros_like(before)),
                    "integrated load",
                    ndim=before.ndim,
                )
                if requested.dtype != np.dtype(np.float64) or requested.shape != before.shape:
                    raise ValueError("integrated load shape/float64 precision must match input")
                with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                    fields[name] = before + requested / self.duration
                requests[name] = requested
            actual = family.inputs_type(**fields)
            if domain in self._channels:
                represented[domain] = {}
            for name in family.input_fields:
                before = getattr(original, name)
                after = np.asarray(getattr(actual, name))
                if (
                    after.dtype != before.dtype
                    or after.shape != before.shape
                    or not np.isfinite(after).all()
                ):
                    raise SolverFailure("constructed input changed shape, precision or finiteness")
                if name not in requests:
                    if not _same_array(after, before):
                        raise SolverFailure("constructed input changed an undeclared channel")
                    continue
                requested = requests[name]
                with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                    delivered = (after - before) * self.duration
                    error = np.abs(delivered - requested)
                    budget = 1e-10 * np.abs(requested)
                if not np.isfinite(delivered).all() or np.any(error > budget):
                    raise SolverFailure("integrated load is not representable in the actual input")
                represented[domain][name] = delivered
            inputs[domain] = actual
        effective = SceneInputs(inputs)
        copied_state, copied_inputs = self._registry.snapshot(
            self._scene, self._initial, effective, self._context, self._schema
        )
        _check_snapshot_copy(
            self._scene, self._registry, self._initial, effective, copied_state, copied_inputs
        )
        return copied_inputs, own_diagnostics(represented)

    def evaluate(self, integrated_loads):
        if self._closed:
            raise RuntimeError("load replay is closed")
        try:
            effective, represented = self._compose(integrated_loads)
        except (FloatingPointError, OverflowError) as error:
            raise SolverFailure(f"integrated load is not representable: {error}") from error
        result = self._runtime.step(self._initial, effective, self.duration)
        return ReplayTrial(result, represented, effective, self.duration)

    def close(self):
        if not self._closed:
            self._runtime.close()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("load replay is closed")
        return self

    def __exit__(self, *_):
        self.close()
