"""World-keyed TYX pages for the cloth-link forces of ordinary midpoint scene steps."""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Integral
from queue import SimpleQueue
from threading import RLock
from types import MappingProxyType

import numpy as np

from ..core.robot import RobotInputs
from ..core.scene import SceneInputs, StepSettings
from ..core.state import StepInputs
from ..runtime.program import SceneStepResult
from ..solvers.compliant_cloth_contact import CompliantClothContactProgram
from ._world_contact_sites import ClothLinkWorld
from ._world_step_layout import world_key
from .tyx_normal_layer import TyxNormalLayer

_PHASES = ("entry", "midpoint", "endpoint")
_MAX_WORLDS = 64
_MAX_SITES = 1 << 20
_STOP = object()
_COUNTERS = ("completed_launches", "sites", "uploads", "downloads", "uploaded_bytes", "downloaded_bytes")


@dataclass(frozen=True)
class MaterialFaceRange:
    """Half-open range in a phase page stream; node and face IDs remain world-local."""

    world: int
    robot: str
    link: str
    face: str
    nodes: tuple[int, int, int]
    start: int
    stop: int


@dataclass(frozen=True)
class MultifacePhase:
    """Offsets in the concatenated row sequence, before its fixed 128-site device paging."""

    substep: int
    phase: str
    sites: int
    world_offsets: tuple
    faces: tuple[MaterialFaceRange, ...]


@dataclass
class _Request:
    phase: str
    rows: np.ndarray
    faces: tuple


class _WorldCall:
    """Suspend this ordinary synchronous step only inside its own material evaluation."""

    def __init__(self):
        self.events, self.replies = SimpleQueue(), SimpleQueue()

    def submit(self, phase, rows, faces):
        self.events.put(_Request(phase, rows, faces))
        answer = self.replies.get()
        if answer is _STOP:
            raise RuntimeError("grouped material step cancelled")
        return answer

    def run(self, link, stepper, state, inputs, h):
        try:
            token = link._phase_dispatch.set(self.submit)
            try:
                result = stepper.step(state, inputs, h)
            finally:
                link._phase_dispatch.reset(token)
            self.events.put(result)
        except BaseException as error:
            self.events.put(error)

    def next_event(self):
        event = self.events.get()
        if isinstance(event, BaseException):
            raise event
        return event


class TyxMultifaceContactBatch:
    """Advance finite cloth/link worlds with one normal-law page stream per shared phase.

    Supply an iterable of (nonnegative world ID, ClothLinkWorld) and common StepSettings.
    Each original scene stepper owns its geometry, mechanics, path checks and complete report.
    HOST runs one world at a time up to the next material phase. Only this caller thread
    evaluates the combined TYX rows. A physical rejection retains that world's checkpoint.
    Capacity, device and configuration exceptions leave all world checkpoints and the last
    published phase map intact. A subsequent call starts from those same checkpoints.
    """

    def __init__(self, worlds, mode="cuda", *, settings=None, cache_root=None):
        if mode not in ("cuda", "interpret"):
            raise ValueError("multiface batch requires cuda or interpret")
        self._mode, self._cache = mode, cache_root
        self._settings = StepSettings() if settings is None else settings
        self._lock = RLock()
        self._closed = False
        self._layer = None
        self._retired = dict.fromkeys(_COUNTERS, 0)
        self._last_phases = ()
        self._links, self._steppers, self._initial = {}, {}, {}
        for raw_key, spec in worlds:
            key = world_key(raw_key)
            if key in self._steppers:
                raise ValueError("duplicate world key")
            if len(self._steppers) >= _MAX_WORLDS:
                raise ValueError(f"multiface batch supports at most {_MAX_WORLDS} worlds")
            if not isinstance(spec, ClothLinkWorld) or not isinstance(spec.program, CompliantClothContactProgram):
                raise TypeError("expected a ClothLinkWorld with a compliant cloth program")
            program = spec.program
            if program.cloth_links is None or program.backed_pads or program.cloth_self is not None:
                raise ValueError("expected a midpoint cloth-link program without backed pads or self-contact")
            self._links[key] = program.cloth_links
            self._steppers[key] = stepper = program.stepper(self._settings)
            unloaded = {name: RobotInputs.zeros(r.model.dof_count) for name, r in program.robots.items()}
            unloaded[program.cloth_domain] = StepInputs.zeros(program.cloth.particle_count)
            self._initial[key], _ = stepper._snapshot(spec.initial, SceneInputs(unloaded))
        if not self._steppers:
            raise ValueError("at least one material world is required")
        self._states = self._initial.copy()
        # A suspended Python stack per world retains the original geometry and scattering
        # locals. The scheduler resumes only one at a time, including on the first phase.
        self._pool = ThreadPoolExecutor(max_workers=len(self._states), thread_name_prefix="phymeo-material")

    @property
    def last_phases(self):
        with self._lock:
            return self._last_phases

    def _open(self):
        if self._closed:
            raise RuntimeError("multiface batch is closed")

    def _keys(self, keys):
        selected = tuple(self._states) if keys is None else tuple(map(world_key, keys))
        if len(set(selected)) != len(selected):
            raise ValueError("duplicate selected world")
        for key in selected:
            if key not in self._states:
                raise KeyError(key)
        return selected

    def snapshot(self, keys=None):
        with self._lock:
            self._open()
            return {k: self._steppers[k]._clone(self._states[k]) for k in self._keys(keys)}

    def reset(self, keys=None):
        """Restore the selected original checkpoints without changing their neighbours."""
        with self._lock:
            self._open()
            for k in self._keys(keys):
                self._states[k] = self._initial[k]

    def _retire_layer(self):
        if self._layer is not None:
            counters = self._layer.stats()
            for k in _COUNTERS:
                self._retired[k] += counters[k]
            self._layer.close()
            self._layer = None

    def _evaluate(self, requests, index, capacity):
        substep, phase_id = divmod(index, len(_PHASES))
        phase = _PHASES[phase_id]
        if substep >= self._settings.substeps or any(req.phase != phase for req in requests.values()):
            raise RuntimeError(f"unexpected world material phase, expected {phase} in substep {substep}")
        count = sum(len(req.rows) for req in requests.values())
        if count > capacity:
            raise ValueError(f"site capacity {capacity} below {count} at substep {substep} {phase}")
        start, offsets, faces = 0, [], []
        for key, req in requests.items():
            stop = start + len(req.rows)
            offsets.append((key, start, stop))
            for robot, link, face, nodes, size in req.faces:
                faces.append(MaterialFaceRange(key, robot, link, face, nodes, start, start + size))
                start += size
            if start != stop:
                raise RuntimeError("cloth-link material addresses do not cover the submitted rows")
        trace = MultifacePhase(substep, phase, count, tuple(offsets), tuple(faces))
        if count == 0:
            return trace, np.empty((0, 5), dtype=np.float32)
        if self._layer is None:
            self._layer = TyxNormalLayer(self._mode, cache_root=self._cache)
        values = np.concatenate([req.rows for req in requests.values()])
        try:
            answer = self._layer.evaluate(values)
            if answer.shape != (count, 5) or not np.isfinite(answer).all() or np.any(answer[:, 1] < 0):
                raise RuntimeError("invalid grouped normal layer response")
            return trace, answer
        except BaseException:
            self._retire_layer()
            raise

    def step(self, h, inputs_by_world, *, site_capacity=_MAX_SITES):
        """Return full per-world SceneStepResults; capacity bounds material rows per phase."""
        with self._lock:
            self._open()
            if not isinstance(inputs_by_world, Mapping) or set(self._keys(inputs_by_world)) != set(self._states):
                raise ValueError("inputs must cover every registered world")
            if isinstance(site_capacity, (bool, np.bool_)) or not isinstance(site_capacity, Integral) or not 0 <= site_capacity <= _MAX_SITES:
                raise ValueError(f"site capacity must be an integer in [0, {_MAX_SITES}]")
            capacity = int(site_capacity)
            calls = {k: _WorldCall() for k in self._states}
            jobs, events, results, traces = [], {}, {}, []
            try:
                for k, call in calls.items():
                    jobs.append(self._pool.submit(call.run, self._links[k], self._steppers[k], self._states[k], inputs_by_world[k], h))
                    events[k] = call.next_event()
                while events:
                    requests = {}
                    for k, event in events.items():
                        if isinstance(event, SceneStepResult):
                            if event.report.accepted and len(traces) != len(_PHASES) * self._settings.substeps:
                                raise RuntimeError("world accepted with missing material phases")
                            results[k] = event
                        else:
                            requests[k] = event
                    if not requests:
                        break
                    trace, answer = self._evaluate(requests, len(traces), capacity)
                    traces.append(trace)
                    events = {}
                    for k, start, stop in trace.world_offsets:
                        calls[k].replies.put(answer[start:stop])
                        events[k] = calls[k].next_event()
                # Clone all proposals before replacing any live world. Callers own their reports.
                states = {k: self._steppers[k]._clone(results[k].state) for k in self._states}
                publication = MappingProxyType({k: results[k] for k in self._states})
                self._states, self._last_phases = states, tuple(traces)
                return publication
            finally:
                for call in calls.values():
                    call.replies.put(_STOP)
                for job in jobs:
                    job.result()

    def stats(self):
        """Actual normal-layer counters, including work performed before a later rejection."""
        with self._lock:
            current = self._layer.stats() if self._layer is not None else {}
            return dict(backend=self._mode, **{k: self._retired[k] + current.get(k, 0) for k in _COUNTERS})

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._pool.shutdown(wait=True)
                self._retire_layer()
                for stepper in self._steppers.values():
                    stepper.close()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args):
        self.close()
