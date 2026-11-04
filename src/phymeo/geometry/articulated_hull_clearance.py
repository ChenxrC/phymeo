"""Positive clearance between a convex link tool and a thickened, affine material triangle."""

from fractions import Fraction

import numpy as np

from ..constraints.collision_shapes import CollisionMotion, RigidTurn
from ..core.capabilities import SolverFailure
from ._convex_support_path import _AffineSupport, _RigidSupport, _Window
from ._prism_pair_exact import sqrt_bounds
from ._round_sweep import _vector
from ._triangle_pair_sweep import _FacePath

_AXES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


class _CoatedFace(_AffineSupport):
    """The isotropic material half-thickness expands every directional support uniformly."""

    def __init__(self, shape, path, thickness):
        super().__init__(shape, path)
        self.thickness = Fraction(thickness)
        self.allowances = {}

    def interval_upper(self, direction, lo, hi):
        allowance = self.allowances.get(direction)
        if allowance is None:
            n = _vector(direction)
            allowance = self.thickness * sqrt_bounds(n @ n, 96)[1]
            self.allowances[direction] = allowance
        a, b = super().interval_upper(direction, lo, hi)
        return a + allowance, b + allowance


def link_hull_motion(model, tool, shape, q0, q1, start_fk, end_fk):
    """Build the exact joint path of this link for supported affine joint-coordinate drifts."""
    link = tool.link_index
    chain = []
    while link >= 0:
        chain.append(link)
        link = int(model.parents[link])
    chain.reverse()
    changed = [j for j in chain if q0[j] != q1[j]]
    revolute = [j for j in changed if model.joint_types[j] == "revolute"]
    if len(revolute) > 1:
        raise SolverFailure(
            f"{tool.key}: multiple active rotational ancestors lack a certified path"
        )
    if revolute and any(j in changed for j in chain[chain.index(revolute[0]) + 1 :]):
        raise SolverFailure(
            f"{tool.key}: moving prismatic joint below a turning ancestor is unresolved"
        )
    i = tool.link_index
    local = np.asarray(tool.local_vertices)
    x0 = local @ start_fk.rotation[i].T + start_fk.position[i]
    x1 = local @ end_fk.rotation[i].T + end_fk.position[i]
    if revolute:
        j = revolute[0]
        angle = float(q1[j] - q0[j])
        if not np.isfinite(angle) or abs(angle) > 2 * np.pi:
            raise SolverFailure(f"{tool.key}: joint arc exceeds one turn in a checked segment")
        axis = start_fk.axis[j]
        turn = RigidTurn(
            start_fk.position[j], end_fk.position[j], axis / np.linalg.norm(axis), angle
        )
    else:
        turn = RigidTurn(start_fk.position[i], end_fk.position[i], (1.0, 0.0, 0.0), 0.0)
    path = CollisionMotion(x0, turn=turn)
    support = (_RigidSupport if revolute else _AffineSupport)(shape, path)
    endpoint, rotation_error = support.pose(Fraction(1))
    # FK frame composition and its conversion to actual world vertices use FP64.
    # Count all ancestors, including stationary radials and telescoping offsets.
    reach = np.linalg.norm(local, axis=1).max() + sum(
        np.linalg.norm(model.joint_position[j]) for j in chain
    )
    reach += sum(max(abs(q0[j]), abs(q1[j])) for j in chain if model.joint_types[j] == "prismatic")
    slack = 256 * np.finfo(float).eps * (len(chain) + 1) ** 2 * max(1.0, reach)
    if np.max(np.linalg.norm(endpoint.start - x1, axis=1)) > slack + float(rotation_error):
        raise SolverFailure(
            f"{tool.key}: constructed joint arc misses the actual endpoint geometry"
        )
    return support, float(slack)


def require_cloth_clearance(tool, face, path, radius, slack):
    """Prove strict separation of the whole material coating or reject the physical segment."""
    if not _FacePath(face, path).valid:
        raise SolverFailure("degenerate cloth triangle on the tool clearance path")
    cloth = _CoatedFace(face, path, Fraction(radius) + Fraction(slack))
    window = _Window(tool, cloth)
    cells = [(Fraction(0), Fraction(1), 0)]
    for _ in range(256):
        if not cells:
            return
        lo, hi, depth = cells.pop()
        if window.clear_on(lo, hi, _AXES):
            continue
        mid = (lo + hi) / 2
        sample = window.sample(mid)
        normal = None if sample.witness is None else sample.witness.normal
        directions = () if normal is None else (normal,)
        if window.clear_on(lo, hi, directions):
            continue
        # A proposed normal is only a direction. The projection inequality,
        # including rotation uncertainty and physical coating, decides clearance.
        if depth >= 16 or not window.clear_on(mid, mid, (*directions, *_AXES)):
            raise SolverFailure(
                "convex tool/cloth core clearance unproved on the physical joint path"
            )
        cells.extend(((mid, hi, depth + 1), (lo, mid, depth + 1)))
    raise SolverFailure("convex tool/cloth clearance subdivision capacity exceeded")
