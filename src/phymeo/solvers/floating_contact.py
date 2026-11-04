"""Atomic loaded floating-tree velocity prediction, instantaneous contact, and one pose drift."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..constraints.contact_impulse import ContactImpulseResult
from ..core.capabilities import SolverFailure
from ..core.result import StepReport
from ..core.rigid import RigidBodyModel, RigidBodyState
from ..core.validation import validate_dt
from ..coupling.floating_port import FloatingRobotPointPort
from ..coupling.ports import RigidPointPort
from ..coupling.witness_impulses import _unpack_row, solve_witness_impulses
from ..dynamics.floating_robot import FloatingRobotState
from ..dynamics.robot_dynamics import _finite
from .floating_tree import _drift_floating_state, floating_dynamics_terms


@dataclass(frozen=True)
class FloatingContactStep:
    """Accepted floating states, force-only predictions at input q and the impulse solver audit.

    Rejected numerical attempts retain the original dynamic states with no predictions or
    impulse result. Fixed supports can be inspected in the accepted impulse result.
    """

    states: Mapping[str, FloatingRobotState]
    report: StepReport
    predicted: Mapping[str, FloatingRobotState]
    impulses: ContactImpulseResult | None


def _require_complete_support(rows):
    segments = {}
    for row in rows:
        pair, _, index = _unpack_row(row)
        curved = getattr(pair, "curved", None)
        if curved is None or curved.locus == "point":
            span = getattr(getattr(pair, "witness", None), "axis_parameter_span", None)
            if span is not None and span[0] != span[1]:
                raise SolverFailure("extended capsule support requires a complete contact manifold")
            continue
        if (
            curved.locus != "segment"
            or isinstance(index, bool)
            or not isinstance(index, Integral)
            or index not in (0, 1)
        ):
            raise SolverFailure(
                "a floating step requires both endpoints of a supported segment manifold"
            )
        original, found = segments.setdefault((pair.first, pair.second), (pair, set()))
        if pair is not original or index in found:
            raise SolverFailure("duplicate or inconsistent sampled segment manifold")
        found.add(index)
    if any(len(found) != 2 for _, found in segments.values()):
        raise SolverFailure("incomplete segment manifold")


def step_floating_contacts(
    systems,
    attachments,
    rows,
    mu_static,
    mu_dynamic,
    dt,
    *,
    fixed_bodies=None,
    local_vertices=None,
    local_normals=None,
    velocity_tolerance=1e-10,
    response_tolerance=1e-12,
    position_tolerance=1e-9,
    max_iterations=200,
):
    """Advance dynamic domains using contacts already determined at this step's input pose.

    systems maps each domain to (FloatingRobotModel, FloatingRobotState, FloatingRobotInputs).
    fixed_bodies maps additional domains to (RigidBodyModel, RigidBodyState) with zero inverse
    mass and zero linear/angular speed. attachments maps either kind of domain to (body indices,
    local material points). A domain with no attached sites may be omitted. rows and friction
    arrays follow solve_witness_impulses: each collision-key binding selects a domain/site row,
    and local_vertices supplies body-local shape points for active row keys only.
    Half-spaces require their oriented body-local vectors in local_normals.
    For a cylinder resting along a line, supply both indexed endpoints of the same
    registered pair before drift. A positive-area hull-hull face requires every indexed
    corner of its single contact polygon. Other extended curved supports require a manifold solver.

    Forces, gravity and drives advance velocities at input q. All graph rows then act on those
    velocities at that same q. Only the resulting velocity drifts each dynamic domain. Pass
    empty rows after separation. New arrivals inside a drift require prior event localization.
    Invalid bindings raise before any caller-owned state is changed.
    """
    if not isinstance(systems, Mapping) or not systems:
        raise ValueError("at least one floating system is required")
    if not isinstance(attachments, Mapping):
        raise TypeError("attachments must map domains to body rows and local points")
    fixed = {} if fixed_bodies is None else fixed_bodies
    if not isinstance(fixed, Mapping):
        raise TypeError("fixed_bodies must be a mapping")
    if set(fixed) & set(systems) or set(attachments) - (set(systems) | set(fixed)):
        raise ValueError("fixed and moving domains must be disjoint and cover all attachments")
    participants = dict(systems)
    originals = MappingProxyType({name: state for name, (_, state, _) in participants.items()})
    h = validate_dt(dt, np.float64)
    empty = (np.empty(0, dtype=np.int64), np.empty((0, 3)))
    poses = {}
    ports = {}
    try:
        rows = tuple(rows)
        _require_complete_support(rows)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for name, (model, state, inputs) in participants.items():
                terms = floating_dynamics_terms(model, state, inputs)
                predicted = FloatingRobotState(state.q, _finite(state.v + h * terms.acceleration))
                poses[name] = predicted
                ids, points = attachments.get(name, empty)
                ports[name] = FloatingRobotPointPort(model, predicted, ids, points, domain=name)
            for name, (model, state) in fixed.items():
                if not isinstance(model, RigidBodyModel) or not isinstance(state, RigidBodyState):
                    raise TypeError("each fixed support requires a rigid body model and state")
                if (
                    np.any(model.inverse_mass)
                    or np.any(state.velocity)
                    or np.any(state.angular_velocity)
                ):
                    raise ValueError(
                        f"fixed support {name!r} requires zero inverse mass and zero velocity"
                    )
                ids, points = attachments.get(name, empty)
                ports[name] = RigidPointPort(model, state, ids, points, domain=name)
            impulses = solve_witness_impulses(
                ports,
                rows,
                mu_static,
                mu_dynamic,
                local_vertices=local_vertices,
                local_normals=local_normals,
                velocity_tolerance=velocity_tolerance,
                response_tolerance=response_tolerance,
                position_tolerance=position_tolerance,
                max_iterations=max_iterations,
            )
            committed = {
                name: _drift_floating_state(model, state, impulses.states[name].v, h)
                for name, (model, state, _) in participants.items()
            }
    except (FloatingPointError, OverflowError, SolverFailure) as exc:
        return FloatingContactStep(
            originals, StepReport(False, reason=str(exc)), MappingProxyType({}), None
        )
    return FloatingContactStep(
        MappingProxyType(committed),
        StepReport(
            True,
            iterations=impulses.iterations,
            max_residual=impulses.residual_mps / velocity_tolerance,
        ),
        MappingProxyType(poses),
        impulses,
    )
