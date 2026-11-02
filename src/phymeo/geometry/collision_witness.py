"""World-space geometry and material addresses from a collision query."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialContactPoint:
    """A world point on both surfaces, addressed separately on each pair owner."""

    point: tuple[float, ...]
    first_weights: tuple[tuple[int, float], ...]
    second_weights: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class CollisionWitness:
    """One representative with the normal oriented from the first shape to the second."""

    gap_m: float
    normal: tuple[float, ...] | None
    first_point: tuple[float, ...] | None
    second_point: tuple[float, ...] | None
    first_weights: tuple[tuple[int, float], ...]
    second_weights: tuple[tuple[int, float], ...]
    source: str = "host-fp64"
    # Endpoints of the complete closest (first, second) axis-parameter interval
    # for two nonzero capsules. A unique minimum repeats the same pair.
    axis_parameter_span: tuple[tuple[float, float], tuple[float, float]] | None = None
    # World-frame tangent displacement from the moving anchor of an infinite plane.
    first_offset: tuple[float, ...] | None = None
    second_offset: tuple[float, ...] | None = None
    # For zero-thickness triangle pairs the convex hull of these points is the
    # sampled intersection. Record order follows material IDs, not boundary winding.
    intersection_points: tuple[MaterialContactPoint, ...] = ()


@dataclass(frozen=True)
class CollisionSweep:
    """Closed first-contact time bracket on [0,1]; pair.witness is the endpoint witness."""

    start_gap_m: float
    end_gap_m: float
    first_t: tuple[float, float] | None
    witness_time: float | None
    witness: CollisionWitness | None
