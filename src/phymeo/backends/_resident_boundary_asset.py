"""Registration-only asset identity admission; no FK, geometry or device work."""

from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionShape
from ._resident_affine_binding import ResidentAffineTreeBinding
from ._resident_affine_tree import AffineTreeWorldSpec
from ._resident_floating_affine_binding import ResidentFloatingAffineTreeBinding
from ._resident_floating_affine_tree import FloatingAffineTreeWorldSpec
from ._resident_floating_binding import ResidentFloatingTreeBinding
from ._resident_floating_tree import FloatingTreeWorldSpec
from ._resident_tree_binding import ResidentTreeBinding
from ._resident_tree_world import TreeWorldSpec


@dataclass(frozen=True, eq=False)
class BoundaryAssetAdmission:
    group_index: int
    local_points_m: np.ndarray
    provenance: MappingProxyType


_FAMILIES = {
    ResidentTreeBinding: ("ordinary", TreeWorldSpec, TreeWorldSpec),
    ResidentAffineTreeBinding: ("affine", TreeWorldSpec, AffineTreeWorldSpec),
    ResidentFloatingTreeBinding: ("floating", FloatingTreeWorldSpec, FloatingTreeWorldSpec),
    ResidentFloatingAffineTreeBinding: (
        "floating_affine",
        FloatingTreeWorldSpec,
        FloatingAffineTreeWorldSpec,
    ),
}


def admit_boundary_asset(shape, binding, *, physical_spec, owning_spec):
    """Admit a factory source against the exact owning world, never direct topology.

    Private-field forgery is outside the public immutable asset contract.
    """
    family = _FAMILIES.get(type(binding))
    if family is None or type(shape) is not CollisionShape:
        raise TypeError("boundary asset requires an exact four-family binding and CollisionShape")
    name, physical_type, owning_type = family
    if type(physical_spec) is not physical_type or type(owning_spec) is not owning_type:
        raise TypeError("boundary asset family differs from its physical/owning spec")
    physical_owner = owning_spec.physical if "affine" in name else owning_spec
    if physical_owner is not physical_spec:
        raise ValueError("boundary asset physical spec is not its owning world spec")
    original = binding._source_shape
    if type(original) is not CollisionShape or binding.topology is not None:
        raise ValueError(
            "boundary asset requires an original factory source, not a direct topology"
        )
    if shape.key != binding.key or shape.key.world != binding.world:
        raise ValueError("boundary asset key/world differs from its material binding")
    if shape.kind not in ("triangle", "mesh", "sphere", "capsule"):
        raise ValueError(
            "boundary asset supports only original triangle/mesh/sphere/capsule material"
        )
    if shape != replace(original, key=binding.key):
        raise ValueError(
            "boundary asset source shape IDs, facets, radius, revision or filters changed"
        )
    if binding._model_token != physical_spec.model_token:
        raise ValueError("boundary asset source model differs from the full physical model")
    graph = None
    if name == "affine":
        graph = binding._graph_token
        if graph is None or graph != owning_spec.graph_token:
            raise ValueError("boundary asset original affine graph differs from the owning graph")
    elif name == "floating_affine":
        graph = binding.graph_token
        if graph != owning_spec.graph_token:
            raise ValueError(
                "boundary asset original floating affine graph differs from the owning graph"
            )
    count = physical_spec.b if name.startswith("floating") else physical_spec.n
    group = binding.group_index
    if type(group) is not int or not 0 <= group < count:
        raise ValueError("boundary asset material group is outside the actual full body list")
    points = np.asarray(binding.local_points_m)
    if (
        points.dtype != np.float64
        or points.shape != (len(shape.vertex_ids), 3)
        or not np.isfinite(points).all()
    ):
        raise ValueError("boundary asset must retain all finite original FP64 material points")
    if shape.kind in ("triangle", "mesh") and shape.radius != 0:
        raise ValueError("boundary asset finite surface cannot acquire a radius")
    if shape.kind == "triangle" and len(points) != 3:
        raise ValueError("boundary asset triangle needs its three original points")
    if shape.kind == "mesh" and len(shape.triangles) < 4:
        raise ValueError("boundary asset mesh needs its original four or more faces")
    if shape.kind == "sphere" and len(points) != 1 or shape.kind == "capsule" and len(points) != 2:
        raise ValueError("boundary asset round point/axis membership differs")
    local = np.frombuffer(np.ascontiguousarray(points).tobytes(), dtype=np.float64).reshape(
        points.shape
    )
    proof = MappingProxyType(
        dict(
            schema="phymeo-boundary-asset-admission-v1",
            binding_family=name,
            source_shape=original,
            model_token=binding._model_token,
            graph_token=graph,
            physical_topology=physical_spec.topology,
            group_index=group,
            local_points_m=local,
        )
    )
    return BoundaryAssetAdmission(group, local, proof)
