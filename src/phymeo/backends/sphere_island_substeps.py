"""Private staging and immutable traces for bounded sphere-island macrosteps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from ._world_step_layout import as_fp32

if TYPE_CHECKING:
    from .tyx_sphere_island import SphereIslandResult, SphereIslandState


@dataclass(frozen=True)
class SphereIslandStage:
    """Relative seconds from the input checkpoint. Witnesses describe the interval's end.

    An initial_impulse has zero duration and leaves positions fixed. Other rows are discrete
    substep samples. An interval containing a first impact need not end at its continuous TOI.
    """

    kind: str
    start_s: float
    end_s: float
    report: SphereIslandResult


@dataclass(frozen=True)
class SphereIslandAdvance:
    """A world advances exactly one public tick on success.

    Committed stages are empty on rejection. failure holds a diagnostic from a discarded
    trial, which can have a different pose from the state's unchanged public checkpoint.
    """

    accepted: bool
    reason: str | None
    duration_s: float
    requested_substeps: int
    completed_advances: int
    state: SphereIslandState
    stages: tuple[SphereIslandStage, ...]
    failure: SphereIslandStage | None


def _closing_at_start(spec, state):
    """Only prime exact closed/closing material in a nonpenetrating configuration."""
    closing = False
    for body, geom in spec.bodies:
        if spec.plane_height is not None:
            b = state.bodies[body]
            gap = (b.position[2] - spec.plane_height) - geom.radius
            if gap < 0:
                return False
            closing |= gap == 0 and b.velocity[2] < 0
    for (a, ga), (b, gb) in combinations(spec.bodies, 2):
        x, y = state.bodies[a], state.bodies[b]
        delta = np.subtract(y.position, x.position)
        radius = ga.radius + gb.radius
        signed_square = delta @ delta - radius * radius
        if signed_square < 0:
            return False
        closing |= signed_square == 0 and delta @ np.subtract(y.velocity, x.velocity) < 0
    return closing


def advance(batch, h, substeps, keys, forces, contact_capacity, iterations):
    """Called with the batch lock held. Trials touch its device scratch but never its checkpoints."""
    if (
        isinstance(substeps, (bool, np.bool_)) or not isinstance(substeps, Integral)
        or not 1 <= substeps <= 64 or int(substeps) & (int(substeps) - 1)
    ):
        raise ValueError("substeps must be a power of two in [1,64]")
    count = int(substeps)
    h, initial, external, capacity = batch._inputs(h, forces, contact_capacity, iterations)
    dt = float(as_fp32(h / count, "island substep duration", ()))
    if not dt or dt * count != h:
        raise ValueError("substeps must exactly cover the supported positive FP32 interval")
    selected = batch._keys(keys)
    if any(k not in selected for packet in (forces, contact_capacity) if packet is not None for k in packet):
        raise ValueError("force or capacity names an unselected world")
    if not selected:
        return MappingProxyType({})
    starting = {k: batch._states[k] for k in selected}
    active = dict(starting)
    history = {key: [] for key in active}
    outcome = {}

    def stage(kind, start_s, end_s, selected, applied):
        trial = batch._trial(selected, initial, applied, capacity, end_s - start_s, iterations)
        for key, row in trial.items():
            original = starting[key]
            if row.accepted and kind == "initial_impulse":
                if any(
                    body.position != original.bodies[local].position
                    for local, body in row.state.bodies.items()
                ):
                    row = replace(row, accepted=False, reason="initial_impulse_changed_pose",
                                  state=original, contacts=())
            if not row.accepted:
                # Earlier candidate stages and their impulses never become public commitments.
                completed = sum(s.kind == "advance" for s in history[key])
                outcome[key] = SphereIslandAdvance(False, row.reason, h, count, completed, original, (),
                                                    SphereIslandStage(kind, start_s, end_s, row))
                del active[key]
                history[key].clear()
                continue
            new = replace(row.state, tick=original.tick + int(end_s == h))
            active[key] = new
            history[key].append(SphereIslandStage(kind, start_s, end_s, replace(row, state=new)))
            bs, _ = batch._ranges[key]
            for i, body in enumerate(new.bodies.values(), bs.start):
                initial[i] = (*body.position, *body.velocity)

    closing = {key: s for key, s in active.items() if _closing_at_start(batch._worlds[key], s)}
    if closing:
        stage("initial_impulse", 0.0, 0.0, closing, np.zeros_like(external))
    for i in range(count):
        if not active:
            break
        stage("advance", i * dt, (i + 1) * dt, active, external)
    for key, state in active.items():
        outcome[key] = SphereIslandAdvance(True, None, h, count, count, state, tuple(history[key]), None)
    return batch._publish({key: outcome[key] for key in starting})
