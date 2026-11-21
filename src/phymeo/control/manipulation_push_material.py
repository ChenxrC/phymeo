"""Optional finite-material admission for native push route proposals.

This is a frozen geometry constraint, not a force/contact prediction. It keeps
proposal identities and ranking intact and has no robot/object-specific rules.
"""

import json
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .manipulation_diagnostic import _readonly, _require

DISTANCE_TOLERANCE_M = 1e-7
NORMAL_ALIGNMENT_MIN = 1 - 1e-6
BARYCENTRIC_TOLERANCE = 1e-10


@dataclass(frozen=True, eq=False)
class NativePushMaterialFaces:
    source_q: np.ndarray
    asset_sha256: str
    face_ids: tuple[str, ...]
    triangles_world: np.ndarray

    def __post_init__(self):
        q = np.asarray(self.source_q)
        ids = tuple(self.face_ids)
        triangles = np.asarray(self.triangles_world, dtype=np.float64)
        _require(
            q.ndim == 1
            and 0 < len(q) <= 4096
            and q.dtype in (np.dtype("float32"), np.dtype("float64"))
            and np.isfinite(q).all(),
            "finite material source q",
        )
        _require(
            type(self.asset_sha256) is str
            and len(self.asset_sha256) == 64
            and all(c in "0123456789abcdef" for c in self.asset_sha256),
            "material asset SHA256",
        )
        _require(
            0 < len(ids) <= 4096
            and len(set(ids)) == len(ids)
            and all(type(value) is str and value for value in ids),
            "material face identities",
        )
        _require(
            triangles.shape == (len(ids), 3, 3) and np.isfinite(triangles).all(),
            "finite material triangles",
        )
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        magnitudes = np.linalg.norm(normals, axis=1)
        _require(
            np.isfinite(magnitudes).all() and np.all(magnitudes > 0),
            "nondegenerate material triangles",
        )
        q = _readonly(q, dtype=q.dtype)
        triangles = _readonly(triangles)
        object.__setattr__(self, "source_q", q)
        object.__setattr__(self, "face_ids", ids)
        object.__setattr__(self, "triangles_world", triangles)
        object.__setattr__(self, "_normals", _readonly(normals / magnitudes[:, None]))
        payload = dict(
            schema="native-push-material-faces-v1",
            q_dtype=q.dtype.str,
            q_hex=q.tobytes().hex(),
            asset_sha256=self.asset_sha256,
            face_ids=ids,
            triangles_hex=triangles.tobytes().hex(),
            distance_tolerance_m=DISTANCE_TOLERANCE_M,
            normal_alignment_min=NORMAL_ALIGNMENT_MIN,
            barycentric_tolerance=BARYCENTRIC_TOLERANCE,
        )
        object.__setattr__(
            self,
            "binding",
            sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        )

    def validate_source(self, q):
        current = np.asarray(q)
        _require(
            current.dtype == self.source_q.dtype
            and current.shape == self.source_q.shape
            and current.tobytes() == self.source_q.tobytes(),
            "material admission source q changed",
        )

    def classify(self, point_world, outward_normal_world):
        point, normal = (
            np.asarray(point_world, dtype=float),
            np.asarray(outward_normal_world, dtype=float),
        )
        _require(
            point.shape == normal.shape == (3,)
            and np.isfinite(point).all()
            and np.isfinite(normal).all()
            and abs(np.linalg.norm(normal) - 1) <= 2e-6,
            "finite material candidate point and unit outward normal",
        )
        normal = normal / np.linalg.norm(normal)
        plane_any = inside_any = False
        matched, minimum_gap = [], float("inf")
        for face, triangle, n in zip(
            self.face_ids, self.triangles_world, self._normals, strict=True
        ):
            gap = float((point - triangle[0]) @ n)
            _require(np.isfinite(gap), "finite candidate plane gap")
            minimum_gap = min(minimum_gap, abs(gap))
            if abs(gap) > DISTANCE_TOLERANCE_M:
                continue
            plane_any = True
            # Drop the dominant normal axis. This avoids squaring the triangle's
            # condition number as a Gram-matrix barycentric solve would do.
            axes = [i for i in range(3) if i != int(np.argmax(np.abs(n)))]
            u, v = (triangle[1] - triangle[0])[axes], (triangle[2] - triangle[0])[axes]
            w = (point - gap * n - triangle[0])[axes]
            determinant = u[0] * v[1] - u[1] * v[0]
            _require(
                np.isfinite(determinant) and determinant != 0, "finite material projected triangle"
            )
            b = (w[0] * v[1] - w[1] * v[0]) / determinant
            c = (u[0] * w[1] - u[1] * w[0]) / determinant
            bary = np.array([1 - b - c, b, c])
            _require(np.isfinite(bary).all(), "finite candidate barycentric coordinates")
            if np.min(bary) < -BARYCENTRIC_TOLERANCE:
                continue
            inside_any = True
            if float(n @ normal) >= NORMAL_ALIGNMENT_MIN:
                matched.append(face)
        reason = (
            "declared_material_face"
            if matched
            else "material_normal_mismatch"
            if inside_any
            else "material_point_outside_finite_faces"
            if plane_any
            else "material_point_off_declared_faces"
        )
        return dict(
            eligible=bool(matched),
            reason=reason,
            face_ids=tuple(matched),
            minimum_abs_plane_gap_m=minimum_gap,
        )

    def audit(self, proposals):
        records, identities = [], set()
        for proposal in proposals:
            identity = (proposal.index, proposal.distance_ordinal)
            _require(identity not in identities, "unique material proposal index/length")
            identities.add(identity)
            records.append(
                dict(
                    index=proposal.index,
                    distance_ordinal=proposal.distance_ordinal,
                    input_binding=proposal.input_binding,
                    **self.classify(proposal.point_world, proposal.outward_normal_world),
                )
            )
        return dict(
            kind="material_admission",
            schema="native-push-material-faces-v1",
            binding=self.binding,
            asset_sha256=self.asset_sha256,
            q_dtype=self.source_q.dtype.str,
            q_sha256=sha256(self.source_q.tobytes()).hexdigest(),
            source_q=self.source_q.tolist(),
            face_ids=self.face_ids,
            triangles_world=self.triangles_world.tolist(),
            distance_tolerance_m=DISTANCE_TOLERANCE_M,
            normal_alignment_min=NORMAL_ALIGNMENT_MIN,
            barycentric_tolerance=BARYCENTRIC_TOLERANCE,
            proposals=tuple(records),
            qualification="Finite point/normal admission at frozen entry only; original ranking, IK, path and budgets still apply",
        )
