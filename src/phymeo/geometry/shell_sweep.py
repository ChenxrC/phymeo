"""Complete finite time partitions for the actual nonlinear director curve."""

from dataclasses import dataclass
from fractions import Fraction as F

from ..constraints.conforming_shell import ShellProofLimits, ShellUnresolved
from ._shell_intervals import Arithmetic
from .shell_directors import _binding, _geometry_receipt, _path_polynomials, _positions
from .shell_embedding import _cell_proof, _reference_proof


@dataclass(frozen=True)
class TimeCell:
    start: F
    end: F
    depth: int
    accepted: bool


@dataclass(frozen=True)
class ShellSweepCertificate:
    schema: str
    precision: str
    status: str
    reference_id: str
    binding: str
    policy: ShellProofLimits
    h: F
    reference: object
    cells: tuple
    tree: tuple
    endpoint_geometry: tuple
    operations: int


def certify_shell_sweep(reference, path, *, h, limits=ShellProofLimits()):
    path = _positions(reference, path, limits, path=True)
    arithmetic = Arithmetic(limits)
    duration = arithmetic.input_float(h)
    if duration <= 0:
        raise ShellUnresolved("invalid_input", "h must be finite positive FP64 seconds")
    reference_witness, reference_polys, reference_evaluated = _reference_proof(reference, arithmetic)
    polynomials = _path_polynomials(reference, path, arithmetic)
    pending, accepted, tree = [(F(0), F(1), 0)], [], []
    refinable = {"law_domain_unknown", "director_domain_unknown", "projected_support_unknown", "division_interval_contains_zero", "tetra_orientation_unknown", "embedding_not_certified", "degenerate_face", "vanishing_director", "inverted_or_degenerate_tetra"}
    while pending:
        left, right, depth = pending.pop()
        arithmetic.visit_cell(depth)
        try:
            witness = _cell_proof(reference, reference_witness, reference_polys, reference_evaluated, polynomials, left, right, arithmetic)
        except ShellUnresolved as error:
            if error.category not in refinable:
                raise
            tree.append(TimeCell(left, right, depth, False))
            if depth >= limits.max_time_depth:
                raise ShellUnresolved("time_depth_exhausted", error.category) from error
            # Both children are scheduled; a failed leaf can never discard its sibling.
            midpoint = arithmetic.qdiv(arithmetic.qadd(left, right), F(2))
            if arithmetic.created_cells + len(pending) + 2 > limits.max_time_cells:
                raise ShellUnresolved("time_cells_exhausted", "failed parents and all pending children") from error
            pending.append((midpoint, right, depth + 1))
            pending.append((left, midpoint, depth + 1))
        else:
            tree.append(TimeCell(left, right, depth, True))
            accepted.append(witness)
    if not accepted or accepted[0].start != 0 or accepted[-1].end != 1 or any(a.end != b.start for a, b in zip(accepted, accepted[1:])):
        raise ShellUnresolved("incomplete_time_cover", "producer partition")
    endpoint_geometry = tuple(_geometry_receipt(reference, path[index], arithmetic) for index in range(2))
    return ShellSweepCertificate("conforming-shell-sweep-v1", "ideal-from-stored-fp64", "strict_geometry_certified", reference.reference_id, _binding("sweep", reference, path, limits, h=duration), limits, duration, reference_witness, tuple(accepted), tuple(tree), endpoint_geometry, arithmetic.operations)