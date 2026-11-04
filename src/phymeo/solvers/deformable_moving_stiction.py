"""Two-sided compliant stiction for free particles and finite, plane-parallel material faces."""

from dataclasses import dataclass, replace
from math import atan2, hypot

import numpy as np

from ..core.capabilities import SolverFailure
from .deformable_sliding import _tuple
from .deformable_stiction import MaterialStickContact, MaterialStickSlipStepper, _in_plane, _Trial


@dataclass(frozen=True)
class MovingMaterialStickContact(MaterialStickContact):
    """Signed work uses actual displacements; support rows follow material.second_weights."""

    effective_inverse_mass_per_kg: float
    vertex_tangent_work_j: float
    support_node_tangent_work_j: tuple
    support_turn_rad: float


@dataclass(frozen=True)
class _MovingTrial(_Trial):
    rows: tuple
    coefficients: np.ndarray
    effective_inverse_mass: float
    nodal_velocity: np.ndarray
    velocity_scale: np.ndarray
    h: float


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _minimum_oriented_area(a, b, da, db, sign):
    """Minimum signed doubled area of two linearly moving vectors over [0,1]."""
    c = sign * _cross2(a, b)
    linear = sign * (_cross2(da, b) + _cross2(a, db))
    quadratic = sign * _cross2(da, db)
    low = min(c, c + linear + quadratic)
    if quadratic > 0 and -2 * quadratic < linear < 0:
        low = min(low, c - linear * linear / (4 * quadratic))
    return low


def _face_transport(initial, final, n, eps):
    """Certify an interior path and return the proper in-plane polar rotation of the face."""
    face = initial[1:] - initial[1]
    moves = final - initial
    relative = moves - moves[1]
    extent = max(
        np.linalg.norm(face, axis=1).max(), np.linalg.norm(face + relative[1:], axis=1).max()
    )
    if np.max(np.abs(relative[1:] @ n)) > 32 * eps * extent:
        raise SolverFailure(
            "moving stiction requires a plane-parallel material face; support would tilt"
        )
    axis = face[1] / hypot(*face[1])
    basis = np.column_stack((axis, np.cross(n, axis)))
    points = (initial - initial[1]) @ basis
    shifts = relative @ basis
    a, b = points[2] - points[1], points[3] - points[1]
    da, db = shifts[2] - shifts[1], shifts[3] - shifts[1]
    sign = 1 if _cross2(a, b) > 0 else -1
    # The inputs are physical coordinates; do the planar predicates in FP64.
    floor = 128 * np.finfo(float).eps * extent**2
    if _minimum_oriented_area(a, b, da, db, sign) <= floor:
        raise SolverFailure("moving material face would degenerate during the affine step")
    for i, j in ((1, 2), (2, 3), (3, 1)):
        if (
            _minimum_oriented_area(
                points[j] - points[i],
                points[0] - points[i],
                shifts[j] - shifts[i],
                shifts[0] - shifts[i],
                sign,
            )
            <= floor
        ):
            raise SolverFailure(
                "loaded material projection reaches a finite face edge; unload before transferring history"
            )
    # Columns are material edge vectors. The affine map is independent of vertex numbering.
    first = np.column_stack((a, b))
    last = np.column_stack((a + da, b + db))
    gradient = np.linalg.solve(first.T, last.T).T
    x, y = float(np.trace(gradient)), float(gradient[1, 0] - gradient[0, 1])
    length = hypot(x, y)
    if not np.isfinite(length) or length == 0:
        raise SolverFailure("undefined material polar rotation")
    planar = np.array([[x, -y], [y, x]]) / length
    return basis @ planar @ basis.T + np.outer(n, n), atan2(y, x)


class MovingMaterialStickSlipStepper(MaterialStickSlipStepper):
    """Apply equal and opposite nodal loads, using the current VT material interpolation.

    Active supports may translate and deform within a plane of constant normal through
    each affine step. The elastic vector follows the face's in-plane polar rotation.
    Every dynamic node can belong to only one loaded station. Contact must release
    before changing faces; a loaded step cannot cross a finite edge or flip a face.
    """

    def _dynamic_rows(self, trial):
        return tuple(i for i in trial.rows if self._model.inverse_mass[i] > 0)

    def _trial(self, load, state, base_force, previous, h):
        feature = load.feature
        if feature.kind != "VT" or len(load.sites) != 1:
            raise SolverFailure(
                "moving stiction requires a unique vertex against a triangle interior"
            )
        site = load.sites[0]
        ((vertex, _),) = self._normal._site_rows(feature.first_nodes, site.material.first_weights)
        face = self._normal._site_rows(feature.second_nodes, site.material.second_weights)
        if len(face) != 3 or self._model.inverse_mass[vertex] <= 0:
            raise SolverFailure(
                "moving stiction requires a free vertex and a finite face interior, away from the edge"
            )
        rows = (vertex, *(i for i, _ in face))
        index = list(rows)
        c = np.array([1.0, *(-w for _, w in face)])
        inv = self._model.inverse_mass[index].astype(float)
        pinned = inv == 0
        v = state.velocity[index].astype(float)
        if np.any(v[pinned]):
            raise SolverFailure("fixed material nodes must be stationary")
        n = -np.asarray(feature.normal)
        n /= hypot(*n)
        key = (feature.kind, feature.first_nodes, feature.second_nodes)
        prior = previous.pop(key, None)
        if prior is None and any(k[1] == feature.first_nodes for k in previous):
            raise SolverFailure("moving contact changed face without a sampled release")
        if prior is not None and np.dot(prior.outward_normal, n) < 1 - 1e-12:
            raise SolverFailure("active material face changed side or tilted")
        old = np.zeros(3) if prior is None else np.asarray(prior.elastic_displacement_m)
        effective = float((c * c) @ inv)
        acceleration = self._model.gravity.astype(float) + inv[:, None] * base_force[index].astype(
            float
        )
        acceleration[pinned] = 0
        free_v = v + h * acceleration
        free = h * _in_plane(c @ free_v, n)
        force, sliding = self._return_map(old, free, effective, h, site.force_n, prior)
        accepted_v = free_v + h * (inv * c)[:, None] * force
        expected = h * _in_plane(c @ accepted_v, n)
        vscale = (
            np.linalg.norm(v, axis=1)
            + h * np.linalg.norm(acceleration, axis=1)
            + h * inv * abs(c) * hypot(*force)
        )
        if not np.isfinite(accepted_v).all() or not np.isfinite(expected).all():
            raise SolverFailure("moving material response exceeds precision")
        return _MovingTrial(
            feature,
            site,
            vertex,
            face,
            n,
            prior,
            old,
            force,
            expected,
            accepted_v[0],
            sliding,
            rows,
            c,
            effective,
            accepted_v,
            vscale,
            h,
        )

    def _accept(self, trial, initial, final, tick):
        index = list(trial.rows)
        before = initial.position[index].astype(float)
        after = final.position[index].astype(float)
        velocity = final.velocity[index].astype(float)
        eps = float(np.finfo(initial.position.dtype).eps)
        dx = after - before
        scale = (
            np.linalg.norm(before, axis=1)
            + np.linalg.norm(after, axis=1)
            + trial.h * trial.velocity_scale
        )
        verror = np.linalg.norm(velocity - trial.nodal_velocity, axis=1)
        xerror = np.linalg.norm(dx - trial.h * trial.nodal_velocity, axis=1)
        if np.any(
            verror > 48 * eps * (trial.velocity_scale + np.linalg.norm(velocity, axis=1))
        ) or np.any(xerror > 48 * eps * scale):
            raise SolverFailure(
                "particle solver did not realize both sides of the material response"
            )
        relative = trial.coefficients @ dx
        if trial.feature.distance_m + relative @ trial.normal <= 0:
            raise SolverFailure("loaded vertex would cross the moving support plane")
        rotation, turn = _face_transport(before, after, trial.normal, eps)
        delta = _in_plane(relative, trial.normal)
        tol = 48 * eps * float(abs(trial.coefficients) @ scale)
        if hypot(*(delta - trial.predicted_delta)) > tol:
            raise SolverFailure(
                "relative moving material displacement differs from the mass response"
            )
        contact = self._record(trial, delta, eps, tol, tick)
        history = replace(
            contact.history,
            elastic_displacement_m=_tuple(rotation @ contact.history.elastic_displacement_m),
        )
        work = trial.coefficients * (dx @ trial.force)
        if abs(float(work.sum()) - contact.tangent_work_j) > tol * hypot(*trial.force):
            raise SolverFailure("vertex and support work differ from relative material work")
        return MovingMaterialStickContact(
            **vars(replace(contact, history=history)),
            effective_inverse_mass_per_kg=trial.effective_inverse_mass,
            vertex_tangent_work_j=float(work[0]),
            support_node_tangent_work_j=_tuple(work[1:]),
            support_turn_rad=turn,
        )
