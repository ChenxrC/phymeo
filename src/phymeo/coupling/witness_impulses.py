"""Bind actual published surface witnesses to the existing multi-domain Coulomb solver."""

from collections.abc import Mapping

import numpy as np

from ..constraints.contact_impulse import solve_contact_impulses
from ..core.validation import finite_real
from ..geometry.collision_pipeline import CollisionPair
from .domain_graph import MultiDomainPointOperator


def solve_witness_impulses(
    ports,
    rows,
    mu_static,
    mu_dynamic,
    *,
    sample="pose",
    velocity_tolerance=1e-10,
    response_tolerance=1e-12,
    position_tolerance=1e-9,
    max_iterations=200,
):
    """Solve coincident material rows at one physical instant, with zero restitution.

    Each row is a (CollisionPair, {shape_key: (port_domain, local_row)}) tuple.
    A static row must be touching. First-contact rows must share one reported witness
    time and the ports must represent that sampled configuration. world_impulses in
    the returned ContactImpulseResult act on the second collision shape in each row.
    """
    if sample not in ("pose", "first_contact"):
        raise ValueError("select pose or first_contact witnesses")
    tol = finite_real(position_tolerance, "position_tolerance")
    if tol < 0:
        raise ValueError("position tolerance must be non-negative")
    first, second, witnesses = [], [], []
    event_time = None
    for pair, bindings in rows:
        if not isinstance(pair, CollisionPair) or not pair.candidate:
            raise ValueError("an active published collision pair is required")
        if not isinstance(bindings, Mapping) or set(bindings) != {pair.first, pair.second}:
            raise ValueError("witness bindings must cover both registered shape keys")
        if sample == "first_contact":
            sweep = pair.sweep
            if (
                pair.status != "hit"
                or sweep is None
                or sweep.first_t is None
                or sweep.witness_time is None
            ):
                raise ValueError("a first-contact witness with a physical sample time is required")
            now = sweep.witness_time
            if not (sweep.first_t[0] <= now <= sweep.first_t[1]) or (
                event_time is not None and event_time != now
            ):
                raise ValueError("contact graph witnesses must refer to one sampled instant")
            event_time = now
            witness = sweep.witness
        else:
            if pair.status != "touching" or pair.sweep is not None:
                raise ValueError("a static touching witness is required")
            witness = pair.witness
        if (
            witness is None
            or witness.normal is None
            or witness.first_point is None
            or witness.second_point is None
        ):
            raise ValueError("a resolved normal and both material world points are required")
        for value in (witness.first_point, witness.second_point, witness.normal):
            observed = np.asarray(value, dtype=np.float64)
            if observed.shape != (3,) or not np.isfinite(observed).all():
                raise ValueError(
                    "collision witness world vectors must be finite and three-dimensional"
                )
        if not np.isfinite(witness.gap_m) or abs(witness.gap_m) > tol:
            raise ValueError("signed witness gap exceeds the position tolerance")
        first.append(bindings[pair.first])
        second.append(bindings[pair.second])
        witnesses.append(witness)
    # Public collision normals point from first to second. The existing graph
    # expects a normal from its right endpoint toward its left endpoint.
    graph = MultiDomainPointOperator(ports, second, first)
    positions = {name: port.positions() for name, port in ports.items()}
    for witness, a, b in zip(witnesses, first, second, strict=True):
        for endpoint, point in ((a, witness.first_point), (b, witness.second_point)):
            domain, row = endpoint
            if np.linalg.norm(positions[domain][row] - point) > tol:
                raise ValueError("material port is not bound to its collision witness")
    normals = np.array([w.normal for w in witnesses], dtype=np.float64).reshape(-1, 3)
    return solve_contact_impulses(
        graph,
        normals,
        mu_static,
        mu_dynamic,
        velocity_tolerance=velocity_tolerance,
        response_tolerance=response_tolerance,
        position_tolerance=tol,
        max_iterations=max_iterations,
    )
