"""Queryable, scoped facts; this registry is not a second runtime dispatcher."""

from dataclasses import dataclass
from itertools import combinations_with_replacement

_STATUSES = frozenset(
    ("verified", "implemented_unverified", "missing", "unsupported_domain", "not_applicable")
)
_KINDS = ("sphere", "capsule", "box", "hull")
_OPERATIONS = ("static", "manifold", "translation_ccd", "rotation_ccd", "response")
_BASE = "source:8128440"
_CATALOG = "docs/benchmarks/p65-collision-public-catalog.md"
_SUPPORT = "docs/benchmarks/p65-convex-support-pairs.md"


@dataclass(frozen=True)
class CollisionCapability:
    first_kind: str
    second_kind: str
    operation: str
    owner_mode: str
    backend: str
    status: str
    route: str
    domain: str
    evidence: tuple[str, ...]

    def __post_init__(self):
        fields = (
            self.first_kind,
            self.second_kind,
            self.operation,
            self.owner_mode,
            self.backend,
            self.route,
            self.domain,
        )
        if any(not isinstance(x, str) or not x.strip() for x in fields):
            raise ValueError("capability fields must be nonempty strings")
        if self.status not in _STATUSES:
            raise ValueError("unknown capability status")
        evidence = tuple(self.evidence)
        if any(not isinstance(x, str) or not x for x in evidence):
            raise ValueError("invalid evidence reference")
        if self.status == "verified" and (
            not any(x.startswith("source:") for x in evidence)
            or not any(x.startswith("docs/") for x in evidence)
        ):
            raise ValueError("verified capability requires source and acceptance evidence")
        a, b = sorted((self.first_kind, self.second_kind))
        object.__setattr__(self, "first_kind", a)
        object.__setattr__(self, "second_kind", b)
        object.__setattr__(self, "evidence", evidence)


def _row(a, b, operation, status, route, domain, evidence=(), *, owner="geometry", backend="host"):
    return CollisionCapability(a, b, operation, owner, backend, status, route, domain, evidence)


def collision_capabilities():
    """Return immutable layer-specific records, including deliberately open scopes.

    A box is registered as a hull in public geometry. It remains a separate
    capability label for the OBB adapter; arbitrary hulls are never recast as boxes.
    Verified rows reuse the cited accepted baseline without rerunning it.
    """
    result = []
    for a, b in combinations_with_replacement(_KINDS, 2):
        rounds = a in ("sphere", "capsule") and b in ("sphere", "capsule")
        polyhedra = a in ("box", "hull") and b in ("box", "hull")
        route = (
            "host rational finite axes"
            if rounds
            else "host HullPairQuery"
            if polyhedra
            else "host convex support map"
        )
        evidence = (_BASE, _CATALOG) if rounds or polyhedra else ()
        for operation in _OPERATIONS:
            status = "implemented_unverified"
            domain = "bounded stored-FP64 convex geometry; unresolved results remain explicit"
            proof = evidence
            if operation == "static":
                status = "verified" if proof else "implemented_unverified"
            elif operation == "manifold":
                status = "verified" if rounds or polyhedra else "missing"
                domain = (
                    "closest finite axis set"
                    if rounds
                    else "complete support patches; all normal ties retained"
                    if polyhedra
                    else "one support witness does not prove a complete pressure patch"
                )
                proof = evidence if status == "verified" else ()
            elif operation == "translation_ccd":
                status = "verified" if rounds or polyhedra else "implemented_unverified"
                domain = "rigid translations only; finite time/iteration budget"
            elif operation == "rotation_ccd":
                proof = ()
                domain = "bounded point-query fallback exists; full family acceptance remains open"
            else:
                status, proof = "not_applicable", ()
                domain = "geometry does not integrate owners; query the explicit owner mode"
            result.append(_row(a, b, operation, status, route, domain, proof))
        for backend in ("hybrid-interpret", "hybrid-cuda"):
            result.append(
                _row(
                    a,
                    b,
                    "static",
                    "verified" if evidence else "implemented_unverified",
                    "TYX broad phase with HOST material narrow phase",
                    "accepted public catalog domains; not device-only narrow phase",
                    evidence,
                    backend=backend,
                )
            )
        current = ("source:25c39d8", "docs/benchmarks/p65-generic-collision-20261107.md")
        domain = (
            "instantaneous unique-normal shallow contact; finite axes and full hull vertices; "
            "ambiguous normals, rounded hulls and general deep axis containment unsupported"
        )
        result.append(
            _row(
                a,
                b,
                "manifold",
                "verified",
                "host contact_patch_sample",
                domain,
                current,
                owner="patch_adapter",
            )
        )
        result.append(
            _row(
                a,
                b,
                "response",
                "verified",
                "host RobotRobotContacts with common patch loads",
                domain,
                current,
                owner="robot_robot",
            )
        )
        if (a, b) != ("capsule", "capsule"):
            result.append(
                _row(
                    a,
                    b,
                    "response",
                    "verified",
                    "host RobotSceneContacts with pair-scoped legacy/convex assembly",
                    domain + "; robot S/C/B/H versus free S/B/H, capsule must be robot",
                    current,
                    owner="robot_rigid",
                )
            )
    for a, b, route, domain in (
        ("mesh", "sphere", "host MeshRoundQuery", "closed finite mesh and round surface"),
        ("mesh", "capsule", "host MeshRoundQuery", "closed finite mesh and finite capsule"),
        ("hull", "triangle", "host HullSurfaceQuery", "full finite material triangle"),
        (
            "triangle",
            "triangle",
            "host triangle_pair_query",
            "finite material surfaces and adjacency",
        ),
        (
            "tetrahedron",
            "tetrahedron",
            "host TetrahedralCellQuery",
            "positive cells and original owners",
        ),
    ):
        result.append(_row(a, b, "static", "verified", route, domain, (_BASE, _CATALOG)))
    for kind in ("cylinder", "cone", "ellipsoid"):
        result.append(
            _row(
                kind,
                "hull",
                "static",
                "verified",
                "host convex support map",
                "bounded stored-FP64 point witnesses; deep ambiguity may be unresolved",
                (_BASE, _SUPPORT),
            )
        )
        result.append(
            _row(
                kind,
                "hull",
                "manifold",
                "missing",
                "host convex support map",
                "point support does not establish a distributed pressure patch",
            )
        )
    return tuple(result)


def get_collision_capability(
    first_kind, second_kind, operation, *, owner_mode="geometry", backend="host"
):
    """Unknown backends/owners never inherit a different layer's verified status."""
    a, b = sorted((first_kind, second_kind))
    key = (a, b, operation, owner_mode, backend)
    for row in collision_capabilities():
        if (row.first_kind, row.second_kind, row.operation, row.owner_mode, row.backend) == key:
            return row
    return _row(
        a,
        b,
        operation,
        "missing",
        "unregistered",
        "no scoped capability record",
        owner=owner_mode,
        backend=backend,
    )
