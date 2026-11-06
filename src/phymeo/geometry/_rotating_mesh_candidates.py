"""Pointwise face candidates for a bounded mesh-round arc query.

The mesh stays in its initial, validated material chart. The caller supplies
one nominal finite-axis sample from the existing rotation enclosure. Its
rotation/rounding error and whole-path speed bound remain with the original
continuous query; this class never treats a sampled face set as a time bound.
"""

from ._closed_mesh import ClosedMesh
from ._mesh_round_candidates import MeshRoundFaceSelector, _axis_square, _point, _radius


class MeshRoundArcCandidates:
    """Adapt one owned material geometry to the existing face-BVH callback.

    Only a tuple of original facet indices is returned. Nearest distance,
    winding, farthest vertices, containment and actual witnesses are still
    evaluated by the established HOST geometry. Nothing is cached by sample
    time, endpoint identity or previous candidate list.
    """

    def __init__(self, mesh, selector):
        if not isinstance(mesh, ClosedMesh) or not isinstance(selector, MeshRoundFaceSelector):
            raise TypeError("arc candidates require a closed mesh and its face selector")
        points = tuple((node, _point(point)) for node, point in mesh.points.items())
        binding = selector.binding
        if (
            tuple(node for node, _ in points) != binding.vertex_ids
            or tuple(f.ids for f in mesh.facets) != binding.faces
            or any(f.shape.key != binding.key for f in mesh.facets)
            or dict(points) != selector._points
        ):
            raise ValueError("arc candidate material binding does not match prepared mesh")
        self._points = points
        self._selector = selector

    def sample_axis(self, ends, *, radius):
        """Keep all global nearest ties and every radius-near finite face.

        Each mesh vertex lies on the closed boundary. The minimum of its exact
        squared distances to this *finite* segment is therefore an upper bound
        for the nearest boundary distance. The existing selector checks that
        bound and encloses it outward in FP32. A sphere is the one-point case.
        Empty or malformed callback output propagates; there is no full-face
        fallback. Fresh stored endpoints after rounding require a fresh call.
        """
        try:
            count = len(ends)
        except TypeError as error:
            raise ValueError("arc sample requires one or two finite endpoints") from error
        if count not in (1, 2):
            raise ValueError("arc sample requires one or two finite endpoints")
        points = tuple(_point(point) for point in ends)
        if len(points) != count:
            raise ValueError("arc sample endpoint count changed")
        radius = _radius(radius)
        start, end = points[0], points[-1]
        squared = {node: _axis_square(point, start, end) for node, point in self._points}
        return self._selector.sample(start, end, radius=radius, vertex_distance2=squared)
