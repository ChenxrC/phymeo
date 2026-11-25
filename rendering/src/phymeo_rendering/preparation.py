"""Prepare one visual asset table for all exposures in a resident session."""
from dataclasses import dataclass, replace
from collections import defaultdict
from hashlib import sha256
from itertools import product
import json
import math
import os
from pathlib import Path

from .cameras import validate_exposure
from .scene import Pose, VisualScene, compose_pose, validate_scene_frame


@dataclass(frozen=True, slots=True)
class PreparedScene:
    scene: VisualScene
    source_scene_key: str
    _report_json: str

    @property
    def report(self):
        """Independent JSON-ready copy of the preparation record."""
        return json.loads(self._report_json)

    def bind_view(self, frame, camera):
        """Bind a calibrated source exposure to the selected visual assets."""
        validate_exposure(frame, camera)
        if frame.key.scene_key != self.source_scene_key:
            raise ValueError("exposure does not belong to the prepared source scene")
        if self.scene.key != self.source_scene_key:
            key = replace(frame.key, scene_key=self.scene.key)
            frame = replace(frame, key=key)
            camera = replace(camera, frame_key=key)
        validate_scene_frame(self.scene, frame)
        return frame, camera


def _asset_ids(values, assets, kind):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"preserve_{kind} must be a collection of IDs")
    selected = set(values)
    missing = selected.difference(item.id for item in assets)
    if missing:
        raise ValueError(f"unknown {kind} requested for preservation: {missing!r}")
    return selected


def _protected(scene, selectors, preserve_meshes, preserve_materials):
    if isinstance(selectors, (str, bytes)):
        raise TypeError("preserve_detail must be a collection of IDs")
    meshes = _asset_ids(preserve_meshes, scene.meshes, 'meshes')
    materials = _asset_ids(preserve_materials, scene.materials, 'materials')
    mesh_ids = {m.id for m in scene.meshes}
    material_ids = {m.id for m in scene.materials}
    for selector in selectors:
        if type(selector) not in (str, int):
            raise TypeError("detail ID must be a node, mesh, material or instance ID")
        matches = [i for i in scene.instances
                   if selector in (i.node_id, i.mesh_id, i.material_id, i.id)]
        if not matches and selector not in mesh_ids and selector not in material_ids:
            raise ValueError(f"unknown detail ID {selector!r}")
        meshes.update(i.mesh_id for i in matches)
        materials.update(i.material_id for i in matches)
        if selector in mesh_ids:
            meshes.add(selector)
        if selector in material_ids:
            materials.add(selector)
    return meshes, materials


def _inverse_pose(pose):
    x, y, z, w = pose.orientation_xyzw
    inverse_rotation = Pose(orientation_xyzw=(-x, -y, -z, w))
    return compose_pose(inverse_rotation, Pose(tuple(-v for v in pose.position_m)))


def _coverage(corners, camera):
    """Conservative projected AABB using positive view depths, clipped to the film."""
    xs, ys, zs = zip(*corners)
    near, far = max(camera.near_m, -max(zs)), min(camera.far_m, -min(zs))
    if far < near:
        return 0.
    px = [camera.cx_px + camera.fx_px * x / d for x in (min(xs), max(xs)) for d in (near, far)]
    py = [camera.cy_px - camera.fy_px * y / d for y in (min(ys), max(ys)) for d in (near, far)]
    width = min(camera.width, max(px)) - max(0., min(px))
    height = min(camera.height, max(py)) - max(0., min(py))
    if width <= 0 or height <= 0:
        return 0.
    return max(width, height)


def _view_weights(scene, views, measured):
    placements = defaultdict(list)
    for frame, camera in views:
        validate_scene_frame(scene, frame)
        validate_exposure(frame, camera)
        nodes = {node.node_id: node.world_pose for node in frame.nodes}
        view_pose = _inverse_pose(camera.world_pose)
        for item in scene.instances:
            world = compose_pose(nodes[item.node_id], item.local_pose)
            placements[item.mesh_id].append((camera, compose_pose(view_pose, world)))

    def importance(mesh, decoded):
        measured[mesh.id] = 0.
        if not placements[mesh.id]:
            return 1.
        bounds = [(min(p[axis] for p in decoded.positions), max(p[axis] for p in decoded.positions))
                  for axis in range(3)]
        corners = [Pose(p) for p in product(*bounds)]
        for camera, transform in placements[mesh.id]:
            projected = _coverage([compose_pose(transform, p).position_m for p in corners], camera)
            measured[mesh.id] = max(measured[mesh.id], projected)
        # Bigger projected details get first claim on discretionary triangles.
        # Keeping a positive floor preserves small / unsampled assets too.
        return 1. + math.log2(1. + measured[mesh.id])

    return importance


def prepare_for_limits(scene, *, usage, cameras, preserve_detail, preserve_meshes, preserve_materials, cache_dir,
                       geometry_limits, color_limits, normal_limits, device):
    from ._prepare_geometry import fit_geometry
    from ._prepare_textures import fit_textures

    if usage not in ('film', 'preview', 'sensor'):
        raise ValueError("usage must be film, preview or sensor")
    protected_meshes, protected_materials = _protected(scene, preserve_detail, preserve_meshes, preserve_materials)
    views = tuple(cameras)
    projected = {}
    weight = _view_weights(scene, views, projected)
    if cache_dir is None:
        cache_dir = Path(os.environ.get('LOCALAPPDATA') or Path.home() / '.cache') / 'phymeo-rendering'
    cache = Path(cache_dir).resolve()
    geometry_scene, geometry = fit_geometry(
        scene, limits=geometry_limits, cache=cache, protected_meshes=protected_meshes,
        importance=weight, allow_adapt=usage != 'sensor')
    selected, textures = fit_textures(
        geometry_scene, color_limits=color_limits, normal_limits=normal_limits, cache=cache,
        protected_materials=protected_materials, allow_adapt=usage != 'sensor')
    adapted = selected is not scene
    if adapted:
        # scene.key owns source transforms, materials and static bindings. Only
        # the per-resource contents below can differ in the prepared manifest.
        manifest = [scene.key, [(m.id, geometry['meshes'][m.id]['output_sha256']) for m in selected.meshes]]
        for material in selected.materials:
            color = getattr(material, 'base_color_map', None)
            normal = getattr(material, 'normal_map', None)
            rough = getattr(material, 'roughness_map', None)
            manifest.append((material.id, color and color.sha256, normal and normal.sha256,
                             rough and (rough.image.sha256, rough.channel)))
        digest = sha256(json.dumps(manifest, separators=(',', ':')).encode()).hexdigest()
        selected = replace(selected, key=f'{scene.key}:prepared-{digest}')
    report = dict(schema='phymeo-rendering/prepared-scene-v1', usage=usage, device=device,
                  adapted=adapted, source_scene_key=scene.key, scene_key=selected.key,
                  camera_count=len(views), maximum_projected_span_px=projected,
                  protected_meshes=sorted(protected_meshes), protected_materials=sorted(protected_materials),
                  limits=dict(geometry=geometry_limits, color=color_limits, normal=normal_limits),
                  geometry=geometry, textures=textures)
    return PreparedScene(selected, scene.key, json.dumps(report, sort_keys=True, allow_nan=False))
