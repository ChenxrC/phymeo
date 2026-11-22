"""Vulkan offscreen raster backend for bounded opaque authored triangle meshes.

The native graphics implementation is independently built from _native/ and
_shaders/. No runtime, compiler or device is imported at module import time.
No path tracing, texture, environment-light or software fallback is provided.
"""

import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from ..outputs import Capture, Channel, ImagePlane
from ..pipeline import DefaultPassPlanner, RenderPath
from ..scene import VisualScene, compose_pose
from . import UnsupportedRequestError, ValidatedRenderSession
from ._native_device import NativeDevice, Parameters, Vertex, default_library, require_library


def _rotation(pose):
    norm = math.hypot(*pose.orientation_xyzw)
    x, y, z, w = (v / norm for v in pose.orientation_xyzw)
    return (
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    )


def _rotate(rotation, vector):
    return tuple(math.fsum(a*b for a, b in zip(row, vector, strict=True)) for row in rotation)


def _vertices(scene, meshes, frame):
    nodes = {node.node_id: node.world_pose for node in frame.nodes}
    materials = {material.id: material for material in scene.materials}
    count = sum(3 * len(meshes[item.mesh_id].triangles) for item in scene.instances)
    result = (Vertex * count)()
    cursor = 0
    for instance in scene.instances:
        pose = compose_pose(nodes[instance.node_id], instance.local_pose)
        rotation = _rotation(pose)
        mesh = meshes[instance.mesh_id]
        material = materials[instance.material_id]
        for triangle in mesh.triangles:
            triangle_id = cursor // 3
            for index in triangle:
                position = tuple(a+b for a, b in zip(
                    _rotate(rotation, mesh.positions[index]), pose.position_m, strict=True
                ))
                normal = _rotate(rotation, mesh.normals[index])
                length = math.hypot(*normal)
                if not math.isfinite(length) or length == 0:
                    raise ValueError(f"mesh {instance.mesh_id!r} has an invalid normal")
                vertex = result[cursor]
                vertex.position[:] = (*position, 1.0)
                vertex.normal_metallic[:] = (*(x/length for x in normal), material.metallic)
                vertex.color_roughness[:] = (*material.base_color, material.roughness)
                vertex.ids[:] = (instance.id, triangle_id, 0, 0)
                cursor += 1
    return result


def _parameters(scene, request, triangle_count, exposure):
    camera = request.camera
    rotation = _rotation(camera.world_pose)
    position = camera.world_pose.position_m
    result = Parameters()
    # R^T is world-to-camera. Flatten by columns for GLSL's default matrix layout.
    result.view[:] = (
        *rotation[0], 0.0, *rotation[1], 0.0, *rotation[2], 0.0,
        *(-math.fsum(rotation[i][j]*position[i] for i in range(3)) for j in range(3)), 1.0,
    )
    result.intrinsics[:] = (camera.fx_px, camera.fy_px, camera.cx_px, camera.cy_px)
    result.clipping[:] = (camera.width, camera.height, camera.near_m, camera.far_m)
    result.camera[:] = (*position, 1.0)
    for index, light in enumerate(scene.lights):
        length = math.hypot(*light.direction)
        result.light_directions[index][:] = (*(x/length for x in light.direction), 0.0)
        result.light_colors[index][:] = (*light.illuminance, 0.0)
    result.counts[:] = (len(scene.lights), triangle_count, int(Channel.RGB in request.channels), 0)
    result.options[:] = (exposure, 1e-5, 0.0, 0.0)
    return result


class _NativeDelegate:
    def __init__(self, scene, meshes, device, exposure):
        self.scene, self.meshes, self.device, self.exposure = scene, meshes, device, exposure
        self.limits = device.info()
        self.triangles = sum(len(meshes[item.mesh_id].triangles) for item in scene.instances)
        self.last_passes = ()
        self.last_request = None
        values = {name: [mesh.positions, mesh.normals, mesh.triangles] for name, mesh in meshes.items()}
        self.geometry_sha256 = hashlib.sha256(json.dumps(
            values, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()).hexdigest()

    def supports(self, request):
        camera = request.camera
        return (request.path is RenderPath.RASTER
                and camera.width <= self.limits["max_width"]
                and camera.height <= self.limits["max_height"]
                and camera.width * camera.height <= 4096 * 4096)

    def capture(self, frame, request):
        camera = request.camera
        plan = DefaultPassPlanner().plan(request)
        vertices = _vertices(self.scene, self.meshes, frame)
        parameters = _parameters(self.scene, request, self.triangles, self.exposure)
        raw = self.device.render(vertices, parameters, camera.width, camera.height)
        count = camera.width * camera.height
        # Native attachments are top-left tightly packed RGBA8/depth/ID/VALID.
        rgba = raw[:count * 4]
        rgb = bytearray(count * 3)
        rgb[0::3], rgb[1::3], rgb[2::3] = rgba[0::4], rgba[1::4], rgba[2::4]
        planes = {
            Channel.RGB: bytes(rgb),
            Channel.FORWARD_DEPTH: raw[count * 4:count * 8],
            Channel.INSTANCE_ID: raw[count * 8:count * 12],
            Channel.VALID: raw[count * 12:],
        }
        result = Capture(frame.key, frame.time_s, camera, tuple(
            ImagePlane(channel, planes[channel]) for channel in request.channels
        ))
        self.last_passes = tuple(stage.value for stage in plan)
        self.last_request = dict(width=camera.width, height=camera.height,
                                 channels=[channel.value for channel in request.channels])
        return result

    def diagnostics(self):
        return dict(
            backend="phymeo-native-vulkan", path="raster", device=self.device.info(),
            library_path=str(self.device.library_path), library_sha256=self.device.library_sha256,
            shader_sha256=dict(self.device.shader_sha256), scene_key=self.scene.key,
            decoded_geometry_sha256=self.geometry_sha256, triangles=self.triangles,
            light_count=len(self.scene.lights), max_lights=4, max_triangles=self.limits["max_vertices"]//3,
            primary_visibility="single-sample Vulkan raster, two-sided opaque triangles",
            lighting="direct GGX/Smith/Schlick metallic-roughness; no ambient or IBL",
            shadows="GPU fragment hard-shadow triangle rays", shadow_bias_m=1e-5,
            minimum_perceptual_roughness=0.045, exposure=self.exposure,
            display="clamp linear exposure to [0,1], then IEC sRGB once",
            readback="explicit synchronous HOST coherent staging after fence",
            attachment_formats=["RGBA8_UNORM", "R32_SFLOAT", "R32_UINT", "R8_UINT"],
            last_passes=list(self.last_passes), last_request=self.last_request,
            stage_execution="visibility raster + fused fragment shadows/lighting/display/sensors",
            path_trace=False, textures=False, physics_steps=0,
        )

    def close(self):
        self.device.close()


class NativeRenderSession(ValidatedRenderSession):
    """Checked session with native-specific device/provenance diagnostics."""

    def diagnostics(self):
        with self._lock:
            return self._delegate.diagnostics()


class NativeBackend:
    """Open one real Vulkan raster session; unsupported paths never fall back.

    Paths are explicit build/install artifacts. With no override, the loader
    searches only the package's private _native directory, never another engine.
    Validation is opt-in and requires the requested Khronos layer to exist.
    """

    def __init__(self, *, library_path=None, shader_directory=None, device_index=0,
                 validation=False, exposure=1.0):
        if isinstance(device_index, bool) or not isinstance(device_index, int):
            raise TypeError("device_index must be an integer, not bool")
        if not 0 <= device_index <= 2**32 - 1:
            raise ValueError("device_index must be a nonnegative uint32")
        if type(validation) is not bool:
            raise TypeError("validation must be bool")
        if isinstance(exposure, bool) or not isinstance(exposure, Real):
            raise TypeError("exposure must be a real number, not bool")
        try:
            finite = math.isfinite(exposure)
        except OverflowError:
            finite = False
        if not finite or not 0 <= exposure <= 3.4028234663852886e38:
            raise ValueError("exposure must be finite nonnegative FP32")
        self.library_path = Path(library_path) if library_path is not None else default_library()
        self.shader_directory = (Path(shader_directory) if shader_directory is not None
                                 else self.library_path.parent)
        self.device_index, self.validation, self.exposure = device_index, validation, exposure

    def open_scene(self, scene: VisualScene) -> NativeRenderSession:
        if not isinstance(scene, VisualScene):
            raise TypeError("scene must be VisualScene")
        require_library(self.library_path)
        if len(scene.lights) > 4:
            raise UnsupportedRequestError("native raster currently supports at most four directional lights")
        # Shared deterministic decoder; this is deliberately not a backend-local parser.
        from ..mesh_data import load_mesh

        meshes = {mesh.id: load_mesh(mesh) for mesh in scene.meshes}
        triangles = sum(len(meshes[item.mesh_id].triangles) for item in scene.instances)
        if triangles > 4096:
            raise UnsupportedRequestError("native raster currently supports at most 4096 triangles")
        device = NativeDevice(self.library_path, self.shader_directory, self.device_index, self.validation)
        try:
            if triangles * 3 > device.info()["max_vertices"]:
                raise UnsupportedRequestError("scene exceeds native device storage-buffer capacity")
            return NativeRenderSession(scene, _NativeDelegate(scene, meshes, device, self.exposure))
        except BaseException:
            device.close()
            raise

