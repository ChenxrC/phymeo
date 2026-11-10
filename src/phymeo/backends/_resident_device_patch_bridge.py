"""Owned conversion of qualified device patches to strict entry material sites.

No collision search or force law lives here. Raw device words remain in query;
the effective FP64 material representation is constructed once and its deltas
are bounded using the original B1 local-scale engineering gates.
"""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..geometry.collision_witness import CollisionWitness
from ..geometry.contact_patch_adapter import ContactPatchSample, ContactSite
from ._device_convex_query import ConvexQueryResult, _freeze
from ._resident_contact_bridge import ContactShapeFrame

_BARY_SUM = 64 * 2**-23
_SURFACE = 2e-4
_GAP = 1e-4
_NORMAL = 2e-3


@dataclass(frozen=True, eq=False)
class DevicePatchConsumption:
    query: ConvexQueryResult
    samples: Mapping
    differences: Mapping
    patch_differences: Mapping

    def __post_init__(self):
        object.__setattr__(self, "samples", _freeze(self.samples))
        object.__setattr__(self, "differences", _freeze(self.differences))
        object.__setattr__(self, "patch_differences", _freeze(self.patch_differences))


def _exact(left, right, message):
    if not np.array_equal(left, right):
        raise ValueError(message)


def _vector(value, message):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(message)
    return array


def _source_frames(result, frames_by_world):
    if not isinstance(frames_by_world, Mapping):
        raise TypeError("device patch requires original entry frames by world")
    frames = {}
    for world, group in frames_by_world.items():
        if not isinstance(group, Mapping):
            raise TypeError("entry frames must be a shape mapping")
        for key, frame in group.items():
            if (not isinstance(frame, ContactShapeFrame) or key != frame.shape.key
                    or key.world != world or key in frames):
                raise ValueError("entry frame owner/world/key mismatch")
            frames[key] = frame
    if set(frames) != set(result.shape_keys):
        raise ValueError("device patch entry source membership differs")
    source = result.source
    if (source.get("schema") != "resident-convex-source-v1"
            or tuple(source["shape_keys"]) != result.shape_keys
            or tuple(source["shape_revisions"]) != result.shape_revisions
            or tuple(source["world_keys"]) != result.world_keys):
        raise ValueError("device patch source binding differs")
    model = source["model"]
    if tuple(shape.key for shape in model.shapes) != result.shape_keys:
        raise ValueError("device patch model source differs")
    for i, shape in enumerate(model.shapes):
        frame = frames[shape.key]
        if frame.shape != shape:
            raise ValueError("entry material IDs/radius/revision differs")
        world_row = result.world_keys.index(shape.key.world)
        origin = _vector(source["world_origins_m"][world_row], "nonfinite device source origin")
        first, last = map(int, source["vertex_offsets"][i:i + 2])
        declared = source["declared_local_points_m"][first:last]
        uploaded = source["uploaded_local_points_m"][first:last]
        _exact(source["vertex_ids"][first:last], shape.vertex_ids,
               "device patch material source IDs differ")
        if (declared.shape != (len(shape.vertex_ids), 3) or uploaded.shape != declared.shape
                or not np.isfinite(declared).all() or not np.isfinite(uploaded).all()):
            raise ValueError("device patch declared/uploaded source shape or finiteness differs")
        expected_group = -1 if frame.group_index is None else frame.group_index
        if (int(source["shape_groups"][i]) != expected_group
                or int(source["shape_world_rows"][i]) != world_row):
            raise ValueError("device patch physical owner/world differs")
        if frame.group_index is None:
            if source["upload_frame"][i] != "world_anchor_relative":
                raise ValueError("static device patch frame differs")
            _exact(declared, frame.world_points_m, "static declared source differs")
            local = declared - origin
            expected_pose = np.r_[np.zeros(3), np.eye(3).ravel()]
        else:
            if source["upload_frame"][i] != "body_local":
                raise ValueError("dynamic device patch frame differs")
            local = declared
            reconstructed = declared @ frame.rotation.T + frame.position_m
            # This is source reconstruction, not a geometric accuracy budget.
            source_rounding = 16 * np.finfo(float).eps * np.maximum(
                1, abs(reconstructed) + abs(frame.world_points_m))
            if np.any(abs(reconstructed - frame.world_points_m) > 1e-10 + source_rounding):
                raise ValueError("dynamic declared source differs from entry material")
            expected_pose = np.r_[frame.position_m - origin, frame.rotation.ravel()]
        _exact(uploaded, np.asarray(local, dtype=np.float32), "uploaded material source differs")
        _exact(source["point_quantization_error_m"][first:last],
               uploaded.astype(np.float64) - local, "material quantization binding differs")
        if (float(source["declared_radius_m"][i]) != shape.radius
                or float(source["uploaded_radius_m"][i]) != float(np.float32(shape.radius))
                or float(source["radius_quantization_error_m"][i])
                != float(np.float32(shape.radius)) - shape.radius):
            raise ValueError("device patch radius source differs")
        pose = np.asarray(source["pose"][i], np.float64)
        if not np.isfinite(pose).all():
            raise ValueError("nonfinite device patch source pose")
        _exact(pose[3:], expected_pose[3:], "device patch stored rotation differs")
        # The accepted anchor is represented separately from relative FK.
        # Never multiply this local bound by the world coordinate magnitude.
        local_scale = max(1.0, float(np.max(abs(local))), shape.radius)
        if np.linalg.norm(pose[:3] - expected_pose[:3]) > _SURFACE * local_scale:
            raise ValueError("device patch stored position differs")
    return frames


def _local_scale(first, second):
    clouds = (first.world_points_m, second.world_points_m)
    scale = 1.0
    for frame, cloud in zip((first, second), clouds, strict=True):
        diameter = max((float(np.linalg.norm(a - b)) for a in cloud for b in cloud), default=0)
        scale = max(scale, diameter + 2 * frame.shape.radius)
    scale = max(scale, float(np.linalg.norm(clouds[1].mean(axis=0) - clouds[0].mean(axis=0))))
    if not np.isfinite(scale):
        raise ValueError("device patch local scale is not representable")
    return scale


def patches_from_device(result, frames_by_world, *, candidates):
    """Construct strict samples without changing any source/site membership.

    The caller validates the result owner/registration/epoch/bank against its
    live source loan. Here all rows, raw words, original/uploaded material
    points and actual entry frames are checked before any force is evaluated.
    """
    if not isinstance(result, ConvexQueryResult):
        raise TypeError("expected the owned convex query result")
    pairs = tuple(sorted(tuple(pair) for pair in candidates))
    requested = tuple(tuple(sorted(pair)) for pair in result.requested_pairs)
    if (len(set(pairs)) != len(pairs) or tuple(row.pair for row in result.rows) != pairs
            or tuple(sorted(requested)) != pairs
            or result.request_row != tuple(pairs.index(pair) for pair in requested)
            or result.request_reversed != tuple(
                tuple(raw) != pair for raw, pair in zip(result.requested_pairs, requested, strict=True))):
        raise ValueError("device patch candidate membership differs")
    if any(pair not in result.source["allowed_pairs"] for pair in pairs):
        raise ValueError("device patch candidate is not registered")
    frames = _source_frames(result, frames_by_world)
    wire, samples, differences, patch_differences = result.patch, {}, {}, {}
    if wire.get("schema") != "resident-convex-patch-v1" or not wire.get("enabled"):
        raise ValueError("device patch output is absent")
    used = 0
    for row_index, row in enumerate(result.rows):
        if (row.status != "RESOLVED" or not row.gap_valid or not row.normal_valid
                or not row.witness_valid or row.patch.status != "RESOLVED"
                or not row.patch.device_resolved or row.patch.host_completed
                or not row.patch.skip_host_patch_search):
            raise ValueError("device patch is unresolved or has a resource failure")
        first, second = (frames[key] for key in row.pair)
        scale = _local_scale(first, second)
        _exact(row.normal, result.raw["normal"][row_index], "distance normal raw binding differs")
        _exact((row.gap_m, row.gap_lower_m, row.gap_upper_m, row.scale_m),
               result.raw["gap"][row_index], "distance gap raw binding differs")
        meta = wire["patch_meta"][row_index]
        if (int(result.raw["meta"][row_index, 0]) != 1 or int(meta[0]) != 1
                or int(meta[1]) != row.patch.dimension
                or int(meta[2]) != len(row.patch.sites) or int(meta[6]) != used
                or int(meta[3]) != 0 or int(meta[4]) != 1):
            raise ValueError("device patch compact layout/raw binding differs")
        if row.patch.dimension not in (0, 1, 2) or len(row.patch.sites) < row.patch.dimension + 1:
            raise ValueError("incomplete device patch")
        origin = _vector(row.origin_m, "invalid device patch origin")
        _exact(origin, result.source["world_origins_m"][result.world_keys.index(row.pair[0].world)],
               "device patch world origin differs")
        raw_normal = _vector(row.normal, "invalid device distance normal")
        norm = float(np.linalg.norm(raw_normal))
        if abs(norm - 1) > 2e-4:
            raise ValueError("device distance normal is outside the qualified unit domain")
        normal = raw_normal / norm
        weights = np.asarray([site.weight for site in row.patch.sites], np.float64)
        weight_sum = float(weights.sum())
        if (not np.isfinite(weights).all() or np.any(weights <= 0)
                or abs(weight_sum - 1) > _BARY_SUM):
            raise ValueError("invalid device patch quadrature")
        effective, changes = [], []
        for ordinal, raw_site in enumerate(row.patch.sites):
            index = used + ordinal
            for field, key in (("weight", "patch_weight"), ("gap_m", "patch_gap"),
                               ("normal", "patch_normal"), ("common_m", "patch_common")):
                _exact(getattr(raw_site, field), wire[key][index], "device site raw binding differs")
            site_normal = _vector(raw_site.normal, "invalid device site normal")
            if (abs(float(np.linalg.norm(site_normal)) - 1) > 2e-4
                    or np.linalg.norm(site_normal - normal) > _NORMAL):
                raise ValueError("device patch site normal disagrees with its distance row")
            points, addresses, offsets, side_changes = [], [], [], []
            for side, (raw, frame, sign) in enumerate(zip(
                    (raw_site.first, raw_site.second), (first, second), (1, -1), strict=True)):
                if raw.key != row.pair[side] or not raw.valid:
                    raise ValueError("device patch witness owner differs")
                count = int(wire["patch_mcount"][index, side])
                if count != len(raw.vertex_ids) or not 1 <= count <= 4:
                    raise ValueError("device patch material count differs")
                _exact(raw.vertex_ids, wire["patch_ids"][index, side, :count], "raw material IDs differ")
                _exact(raw.weights, wire["patch_bary"][index, side, :count], "raw barycentric differs")
                _exact(raw.offset_m, wire["patch_skin"][index, side], "raw skin differs")
                _exact(raw.surface_relative_m, wire["patch_surface"][index, side], "raw surface differs")
                ids, bary = raw.vertex_ids, np.asarray(raw.weights, np.float64)
                total = float(bary.sum())
                if (len(set(ids)) != count or any(node not in frame.shape.vertex_ids for node in ids)
                        or not np.isfinite(bary).all() or np.any(bary < 0)
                        or abs(total - 1) > _BARY_SUM):
                    raise ValueError("invalid device material barycentric representation")
                normalized = bary / total
                # Preserve every original ID, including zero coefficients.
                address = tuple((node, float(w)) for node, w in zip(ids, normalized, strict=True))
                relative_vertices = frame.world_points_m - origin
                core = sum((w * relative_vertices[frame.shape.vertex_ids.index(node)]
                            for node, w in address), np.zeros(3))
                offset = sign * frame.shape.radius * normal
                point_relative = core + offset
                raw_surface = _vector(raw.surface_relative_m, "invalid raw surface")
                if (np.linalg.norm(point_relative - raw_surface) > _SURFACE * scale
                        or np.linalg.norm(offset - raw.offset_m) > _SURFACE * scale):
                    raise ValueError("effective material surface exceeds the original local gate")
                points.append(point_relative + origin)
                offsets.append(tuple(offset))
                addresses.append(address)
                side_changes.append(dict(
                    barycentric_sum=total, barycentric_delta=normalized - bary,
                    surface_delta_m=point_relative - raw_surface,
                    skin_delta_m=offset - raw.offset_m))
            gap = float((points[1] - points[0]) @ normal)
            midpoint = points[0] / 2 + points[1] / 2
            gap_delta = gap - raw_site.gap_m
            common_delta = midpoint - origin - raw_site.common_m
            if abs(gap_delta) > _GAP * scale or np.linalg.norm(common_delta) > _SURFACE * scale:
                raise ValueError("effective gap/common point exceeds the original local gate")
            weight = float(weights[ordinal] / weight_sum)
            witness = CollisionWitness(
                gap, tuple(normal), tuple(points[0]), tuple(points[1]),
                addresses[0], addresses[1], "resident-device-patch-effective-v1",
                first_offset=offsets[0], second_offset=offsets[1])
            effective.append(ContactSite(witness, tuple(midpoint), weight))
            changes.append(dict(
                local_scale_m=scale, weight_sum=weight_sum,
                weight_delta=weight - raw_site.weight, gap_delta_m=gap_delta,
                normal_delta=normal - site_normal, common_delta_m=common_delta,
                material=tuple(side_changes)))
        # Normalizing individually qualified weights can still move the whole
        # first moment. Use this row's original anchor for both sums; absolute
        # world/body-local coordinates must not enlarge the local B1 gate.
        raw_moment = sum((site.weight * _vector(site.common_m, "invalid raw common point")
                          for site in row.patch.sites), np.zeros(3))
        effective_moment = sum((site.weight * (np.asarray(site.load_point) - origin)
                                for site in effective), np.zeros(3))
        moment_delta = effective_moment - raw_moment
        if (not np.isfinite(raw_moment).all() or not np.isfinite(effective_moment).all()
                or not np.isfinite(moment_delta).all()
                or np.linalg.norm(moment_delta) > _SURFACE * scale):
            raise ValueError("effective patch first moment exceeds the original local gate")
        samples[row.pair] = ContactPatchSample(
            tuple(effective), row.patch.dimension, "resident-device-patch-effective-v1")
        differences[row.pair] = tuple(changes)
        patch_differences[row.pair] = dict(
            origin_m=origin, local_scale_m=scale,
            raw_first_moment_m=raw_moment, effective_first_moment_m=effective_moment,
            first_moment_delta_m=moment_delta, first_moment_limit_m=_SURFACE * scale)
        used += len(row.patch.sites)
    return DevicePatchConsumption(result, samples, differences, patch_differences)
