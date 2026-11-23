"""Vulkan offscreen raster backend for bounded opaque authored triangle meshes.

The native graphics implementation is independently built from _native/ and
_shaders/. No runtime, compiler or device is imported at module import time.
No path tracing or software fallback is provided. Textures, uniform environment
and rectangular studio emitters require their explicit resident extensions.
"""

import ctypes as ct
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from ..assets import TexturedMaterial
from ..outputs import Capture, Channel, ImagePlane
from ..studio import StudioLighting
from ..pipeline import DefaultPassPlanner, RenderPath
from ..scene import VisualScene, DirectionalLight, compose_pose
from . import BackendUnavailableError, UnsupportedRequestError, ValidatedRenderSession
from ._native_resident import ResidentDevice, pack_scene
from ._native_ray_query import RayQueryResidentDevice
from ._native_studio import pack_studio
from ._native_texture import pack_textures
from ._native_environment import pack_environment
from ._native_display import configure_display
from ._native_msaa import configure_msaa
from ._native_normal import pack_normal_maps, configure_normal_maps
from ._native_reflection import reflection_declaration, configure_reflection
from ._native_indirect import indirect_declaration, configure_indirect
from ._native_progressive import configure_progressive
from ._native_spatial import configure_spatial
from ._native_device import (
    NativeDevice,
    Parameters,
    Vertex,
    default_library,
    require_library,
    validate_parameters,
)


def _rotation(pose):
    norm = math.hypot(*pose.orientation_xyzw)
    x, y, z, w = (v / norm for v in pose.orientation_xyzw)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _rotate(rotation, vector):
    return tuple(math.fsum(a * b for a, b in zip(row, vector, strict=True)) for row in rotation)


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
                position = tuple(
                    a + b
                    for a, b in zip(
                        _rotate(rotation, mesh.positions[index]), pose.position_m, strict=True
                    )
                )
                normal = _rotate(rotation, mesh.normals[index])
                length = math.hypot(*normal)
                if not math.isfinite(length) or length == 0:
                    raise ValueError(f"mesh {instance.mesh_id!r} has an invalid normal")
                vertex = result[cursor]
                vertex.position[:] = (*position, 1.0)
                vertex.normal_metallic[:] = (*(x / length for x in normal), material.metallic)
                vertex.color_roughness[:] = (*material.base_color, material.roughness)
                vertex.ids[:] = (instance.id, triangle_id, 0, 0)
                cursor += 1
    return result


def _instance_matrices(scene, frame):
    """World-from-mesh rigid transforms in the static instance table order."""
    nodes = {node.node_id: node.world_pose for node in frame.nodes}
    result = (ct.c_float * (16 * len(scene.instances)))()
    for index, instance in enumerate(scene.instances):
        pose = compose_pose(nodes[instance.node_id], instance.local_pose)
        rotation = _rotation(pose)
        values = tuple(value for column in range(3)
                       for value in (*[rotation[row][column] for row in range(3)], 0.))
        result[16*index:16*(index+1)] = (*values, *pose.position_m, 1.)
    if not all(math.isfinite(value) for value in result):
        raise ValueError('resident transform FP32 conversion is nonfinite')
    return result


def _parameters(scene, request, triangle_count, exposure, *, lights=None):
    camera = request.camera
    rotation = _rotation(camera.world_pose)
    position = camera.world_pose.position_m
    result = Parameters()
    # R^T is world-to-camera. Flatten by columns for GLSL's default matrix layout.
    result.view[:] = (
        *rotation[0],
        0.0,
        *rotation[1],
        0.0,
        *rotation[2],
        0.0,
        *(-math.fsum(rotation[i][j] * position[i] for i in range(3)) for j in range(3)),
        1.0,
    )
    result.intrinsics[:] = (camera.fx_px, camera.fy_px, camera.cx_px, camera.cy_px)
    result.clipping[:] = (camera.width, camera.height, camera.near_m, camera.far_m)
    result.camera[:] = (*position, 1.0)
    lights = scene.lights if lights is None else lights
    for index, light in enumerate(lights):
        length = math.hypot(*light.direction)
        result.light_directions[index][:] = (*(x / length for x in light.direction), 0.0)
        result.light_colors[index][:] = (*light.illuminance, 0.0)
    result.counts[:] = (len(lights), triangle_count, int(Channel.RGB in request.channels), 0)
    result.options[:] = (exposure, 1e-5, 0.0, 0.0)
    validate_parameters(result)
    return result


def _image_planes(raw, pixels, channels, attachment_mask):
    """Decode only requested tightly packed attachments into immutable planes."""
    requested=set(channels); values={}; offset=0
    for bit,(channel,width) in enumerate(zip(Channel,(4,4,4,1),strict=True)):
        if not attachment_mask&(1<<bit): continue
        length=pixels*width
        if channel in requested:
            part=raw[offset:offset+length]
            if channel is Channel.RGB:
                rgb=bytearray(pixels*3)
                rgb[0::3],rgb[1::3],rgb[2::3]=part[0::4],part[1::4],part[2::4]
                values[channel]=bytes(rgb)
            else: values[channel]=part
        offset+=length
    if offset!=len(raw) or set(values)!=requested:
        raise ValueError('native attachment packing does not match requested channels')
    return tuple(ImagePlane(channel,values[channel]) for channel in channels)


class _NativeDelegate:
    def __init__(self, scene, meshes, device, exposure, *, resident=False, studio=None, textures=None, environment=None, shadow_mode='software', area_light_grid=2, display_info=None, msaa_info=None, normal_info=None, reflection_info=None, indirect_info=None, progressive_history=None, spatial_info=None):
        self.scene, self.meshes, self.device, self.exposure = scene, meshes, device, exposure
        self.display_info = display_info or {'display_transform':'clip', 'display_transform_abi':None}
        self.resident = resident
        self.msaa_info = msaa_info or configure_msaa(None, 1)
        self.studio = deepcopy(studio)
        self.textures = deepcopy(textures)
        self.normal_info = deepcopy(normal_info)
        self.reflection_info = deepcopy(reflection_info)
        self.indirect_info = deepcopy(indirect_info)
        self.progressive_history = progressive_history
        self.spatial_info = spatial_info
        self._progressive_pending_publication = None
        self.effective_lights = scene.lights
        self.environment = deepcopy(environment)
        self.shadow_mode = shadow_mode
        self.area_light_grid = area_light_grid
        self.limits = device.info()
        self.triangles = sum(len(meshes[item.mesh_id].triangles) for item in scene.instances)
        self.last_passes = ()
        self.last_request = None
        values = {
            name: [mesh.positions, mesh.normals, mesh.triangles] for name, mesh in meshes.items()
        }
        self.geometry_sha256 = hashlib.sha256(
            json.dumps(values, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        ).hexdigest()

    def supports(self, request):
        camera = request.camera
        if not (
            request.path is RenderPath.RASTER
            and camera.width <= self.limits["max_width"]
            and camera.height <= self.limits["max_height"]
            and camera.width * camera.height <= 4096 * 4096
        ):
            return False
        try:
            _parameters(self.scene, request, self.triangles, self.exposure, lights=self.effective_lights)
        except (ValueError, OverflowError):
            return False
        return True

    def capture(self, frame, request):
        camera = request.camera
        plan = DefaultPassPlanner().plan(request)
        vertices = (
            _instance_matrices(self.scene, frame) if self.resident
            else _vertices(self.scene, self.meshes, frame)
        )
        parameters = _parameters(self.scene, request, self.triangles, self.exposure, lights=self.effective_lights)
        if self.resident:
            parameters.counts[3] = len(self.scene.instances)
        progressive_key = None
        progressive_reasons = []
        effective_lights = tuple(
            (tuple(parameters.light_directions[i]), tuple(parameters.light_colors[i]))
            for i in range(parameters.counts[0])
        )
        if self.progressive_history is not None and Channel.RGB in request.channels:
            progressive_key, progressive_reasons = self.progressive_history.prepare(frame, camera, effective_lights)
            self._progressive_pending_publication = (progressive_key, progressive_reasons)
            if self.device._progressive_begin(self.device._handle, int(bool(progressive_reasons))):
                raise RuntimeError(self.device._error())
        attachment_mask=15
        if self.resident and self.limits.get('resident_render_planes_available',False):
            attachment_mask=sum(1<<index for index,channel in enumerate(Channel) if channel in request.channels)
            raw=self.device.render_planes(vertices,parameters,camera.width,camera.height,attachment_mask)
        else:
            raw=self.device.render(vertices,parameters,camera.width,camera.height)
        result=Capture(frame.key,frame.time_s,camera,
            _image_planes(raw,camera.width*camera.height,request.channels,attachment_mask))
        self.last_passes = tuple(stage.value for stage in plan)
        if self.spatial_info is not None and Channel.RGB in request.channels and (progressive_reasons or not self.progressive_history.batches):
            self.last_passes += ('spatial-direct-guide', 'spatial-first-batch-display')
        self.last_request = dict(
            width=camera.width,
            height=camera.height,
            channels=[channel.value for channel in request.channels],
            native_attachment_mask=attachment_mask,
            native_output_bytes=len(raw),
            effective_directional_lights=[dict(direction=list(direction[:3]),illuminance=list(color[:3])) for direction,color in effective_lights],
        )
        return result

    def diagnostics(self):
        return dict(
            backend="phymeo-native-vulkan",
            path="raster",
            device=self.device.info(),
            library_path=str(self.device.library_path),
            library_sha256=self.device.library_sha256,
            shader_sha256=dict(self.device.shader_sha256),
            scene_key=self.scene.key,
            decoded_geometry_sha256=self.geometry_sha256,
            triangles=self.triangles,
            light_count=len(self.effective_lights),
            effective_directional_lights=[dict(direction=light.direction,illuminance=light.illuminance) for light in self.effective_lights],
            max_lights=4,
            max_triangles=(self.limits.get('resident_max_instanced_triangles') if self.resident
                           else self.limits['max_vertices'] // 3),
            resident=self.resident,
            shadow_mode=self.shadow_mode,
            area_light_grid=self.area_light_grid,
            area_light_samples_per_panel=self.area_light_grid**2,
            area_light_brdf_samples_per_panel=8,
            geometry_residency=('static indexed meshes; per-frame instance matrices' if self.resident
                                else 'expanded world triangles per capture'),
            msaa=dict(self.msaa_info),
            primary_visibility=("4-sample Vulkan color raster; independent pixel-center sensors" if self.msaa_info["actual_color_samples"]==4 else
                                "single-sample Vulkan raster, two-sided opaque triangles"),
            lighting=('native GGX/Smith/Schlick plus explicit resident studio shader' if self.studio else
                      'direct GGX/Smith/Schlick metallic-roughness; no ambient or IBL'),
            studio=deepcopy(self.studio),
            environment=deepcopy(self.environment),
            shadows=('GPU fragment Vulkan hardware ray queries against BLAS/TLAS; primary visibility remains raster' if self.shadow_mode=='ray_query' else
                     ('native resident finite-emitter visibility via mesh BVH; see bound shader' if self.studio
                     else ('GPU fragment hard-shadow mesh BVH rays' if self.resident
                           else 'GPU fragment hard-shadow triangle rays'))),
            shadow_bias_m=1e-5,
            minimum_perceptual_roughness=0.045,
            exposure=self.exposure,
            display=('clamp linear exposure to [0,1], then IEC sRGB once' if self.display_info['display_transform']=='clip'
                     else 'nonnegative linear exposure, Khronos PBR Neutral, then IEC sRGB once'),
            **self.display_info,
            readback='explicit synchronous host-visible staging after fence; actual flags in resident_readback when available',
            readback_channels_mode=('compact requested attachments' if self.resident and self.limits.get('resident_render_planes_available',False)
                                    else 'legacy all attachments'),
            published_ownership='immutable CPU bytes; private resident host buffer may be reused after each copy',
            attachment_formats=["RGBA32F -> GPU mean -> RGBA8_UNORM" if self.progressive_history else "RGBA8_UNORM", "R32_SFLOAT", "R32_UINT", "R8_UINT"],
            last_passes=list(self.last_passes),
            last_request=self.last_request,
            stage_execution=("visibility raster + bounded indirect lighting/sensors + GPU linear mean/display compute" if self.progressive_history else "visibility raster + fused fragment shadows/lighting/display/sensors") + ("; first/reset RGB adds direct-guide raster and guided display" if self.spatial_info else ""),
            lifecycle_counts="DLL-wide owned contexts, frames, and explicit Vulkan handles; pools own descriptor sets",
            path_trace=False,
            textures=self.textures is not None,
            texture_assets=deepcopy(self.textures),
            normal_maps=deepcopy(self.normal_info),
            reflection=deepcopy(self.reflection_info),
            indirect=deepcopy(self.indirect_info),
            progressive=self.progressive_history.diagnostics() if self.progressive_history else None,
            spatial_filter=(deepcopy(self.spatial_info) if self.spatial_info is not None else
                            dict(mode='none', last_capture_filtered=False, filtered_captures=0)),
            physics_steps=0,
        )

    def close(self):
        self.device.close()


class NativeRenderSession(ValidatedRenderSession):
    """Checked session with native-specific device/provenance diagnostics."""

    def diagnostics(self):
        with self._lock:
            return self._delegate.diagnostics()


    def capture(self, frame, request):
        # Native computation completes before the public Capture/scene gates.
        # Commit HOST history only after those gates; a rejected publication
        # discards the GPU mean rather than carrying unpublished data forward.
        with self._lock:
            try:
                result = super().capture(frame, request)
            except BaseException:
                if getattr(self._delegate, "_progressive_pending_publication", None) is not None:
                    device = self._delegate.device
                    device._progressive_reset(device._handle)
                    self._delegate.progressive_history.reset('failed_capture')
                    if self._delegate.spatial_info is not None:
                        self._delegate.spatial_info['last_capture_filtered'] = False
                    self._delegate._progressive_pending_publication = None
                raise
            pending = getattr(self._delegate, "_progressive_pending_publication", None)
            if pending is not None:
                self._delegate.progressive_history.commit(*pending)
                self._delegate._progressive_pending_publication = None
            spatial = getattr(self._delegate, 'spatial_info', None)
            if spatial is not None:
                filtered = pending is not None and self._delegate.progressive_history.batches == 1
                spatial['last_capture_filtered'] = filtered
                spatial['filtered_captures'] += int(filtered)
            return result

    def reset_accumulation(self):
        """Discard this session's progressive history; next RGB repeats batch zero."""
        with self._lock:
            self._check_open()
            delegate = self._delegate
            if delegate.progressive_history is None:
                raise UnsupportedRequestError('reset_accumulation requires progressive=True')
            if delegate.device._progressive_reset(delegate.device._handle):
                raise RuntimeError(delegate.device._error())
            delegate.progressive_history.reset()
            if delegate.spatial_info is not None:
                delegate.spatial_info['last_capture_filtered'] = False

    def set_directional_lights(self, lights):
        """Atomically override 0..4 directional lights in a progressive session."""
        with self._lock:
            self._check_open()
            if self._delegate.progressive_history is None:
                raise UnsupportedRequestError('set_directional_lights requires progressive=True')
            if not isinstance(lights, tuple) or any(not isinstance(light, DirectionalLight) for light in lights):
                raise TypeError('lights must be a tuple of DirectionalLight')
            if len(lights) > 4:
                raise ValueError('at most four directional lights are supported')
            for light in lights:
                DirectionalLight(light.direction, light.illuminance)
                length = math.hypot(*light.direction)
                values = (*[x / length for x in light.direction], *light.illuminance)
                if any(not math.isfinite(ct.c_float(x).value) for x in values):
                    raise ValueError('directional light is not finite in FP32')
            self._delegate.effective_lights = lights

    def read_spatial_guide(self):
        """Copy the latest first-batch display-direct guide; never advances sampling."""
        with self._lock:
            self._check_open()
            delegate = self._delegate
            if delegate.spatial_info is None or not delegate.spatial_info['last_capture_filtered']:
                raise UnsupportedRequestError('read_spatial_guide requires the latest accepted capture to be filtered RGB')
            width, height = delegate.progressive_history.key['extent']
            output = (ct.c_ubyte * (width * height * 4))()
            if delegate.device._spatial_read(delegate.device._handle, output, ct.sizeof(output)):
                raise RuntimeError(delegate.device._error())
            return dict(width=width, height=height, format='RGBA8_UNORM',
                        space='display-srgb-direct', data=bytes(output))

    def read_linear_mean(self):
        """Read the current GPU mean without a new exposure or sample batch."""
        with self._lock:
            self._check_open()
            delegate = self._delegate
            history = delegate.progressive_history
            if history is None or not history.batches:
                raise UnsupportedRequestError('read_linear_mean requires an accumulated RGB exposure')
            width, height = history.key['extent']
            output = (ct.c_ubyte * (width * height * 16))()
            if delegate.device._progressive_read(delegate.device._handle, output, ct.sizeof(output)):
                raise RuntimeError(delegate.device._error())
            return dict(width=width, height=height, batches=history.batches,
                        primary_samples_per_pixel=16 * history.batches,
                        format='RGBA32F_LE', space='scene-linear-before-exposure', data=bytes(output))

class NativeBackend:
    """Open one real Vulkan raster session; unsupported paths never fall back.

    Paths are explicit build/install artifacts. With no override, the loader
    searches only the package's private _native directory, never another engine.
    Validation is opt-in and requires the requested Khronos layer to exist.
    resident=True requires the resident ABI and shaders, uploads indexed assets
    once, and submits rigid instance matrices per frame. The default False keeps
    the original ABI2-only library contract; neither mode falls back to the other.
    studio=StudioLighting(...) is separately opt-in and requires the resident
    studio export. None retains only the declared scene lights.
    shadow_mode='ray_query' additionally requires resident=True, the hardware
    extension and dedicated shaders. Default 'software' retains the old path;
    an unsupported hardware request never falls back.
    area_light_grid=4 or 8 selects a separate resident fragment shader with
    midpoint panel integration; default 2 retains the original shader artifact.
    BRDF samples remain eight. Missing dense artifacts never fall back.
    reflection_mode='single_bounce' requires resident hardware ray query and a
    separate shader ABI. Only uploaded metallic>=.99 / roughness factor<=.10
    surfaces emit ideal mirror rays; zero receivers is legal. Hits use terminal
    lighting, misses keep the environment. No rough reflection or recursion.
    indirect_mode='diffuse_bounce' replaces diffuse IBL with sixteen stratified
    cosine samples of terminal nearest-hit light / raw environment. uint32 seed
    is stable at pixel XY, with no frame accumulation. Requires hardware resident
    shaders; may combine with single_bounce specular. Default none needs no ABI.
    progressive=True adds GPU RGBA32F scene-linear averaging before exposure and
    display, with indirect_bounces=1 or 2 and sixteen primary paths per batch.
    The second hop takes only one continuation per path. This opt-in requires
    MSAA1 and compatible compute/fragment/sky artifacts. Up to 4096 static RGB
    batches accumulate; camera/world/epoch/extent/node/light changes reset them.
    Non-RGB captures leave history alone. reset_accumulation starts batch zero;
    read_linear_mean explicitly copies the last accepted RGB mean without a new
    exposure. Static materials/panels belong to a session; direction lights may
    be changed atomically with set_directional_lights in progressive sessions.
    spatial_filter='guided' is an explicit progressive-only first/reset RGB
    preview: same-ID/valid, relative depth .003 and direct-guide RGB8 difference
    <=2 gate a 5x5 binomial kernel. The current direct/environment guide uses
    an extra GPU raster pass only on the first batch, including its shadow tests.
    Raw mean and later RGB batches use the original primary shader unchanged.
    A separate compute ABI and guide image are required; old devices
    never silently approximate it. read_spatial_guide is explicit debug readback.
    display_transform='pbr-neutral' requires a declared resident display ABI;
    default 'clip' preserves the previous output and never probes that extension.
    """

    def __init__(
        self,
        *,
        library_path=None,
        shader_directory=None,
        device_index=0,
        validation=False,
        exposure=1.0,
        resident=False,
        studio=None,
        shadow_mode='software',
        area_light_grid=2,
        display_transform='clip',
        msaa_samples=1,
        reflection_mode='environment',
        indirect_mode='none',
        indirect_seed=0,
        progressive=False,
        indirect_bounces=1,
        spatial_filter='none',
    ):
        if isinstance(device_index, bool) or not isinstance(device_index, int):
            raise TypeError("device_index must be an integer, not bool")
        if not 0 <= device_index <= 2**32 - 1:
            raise ValueError("device_index must be a nonnegative uint32")
        if type(resident) is not bool:
            raise TypeError('resident must be bool')
        self.resident = resident
        if not isinstance(display_transform, str):
            raise TypeError('display_transform must be clip or pbr-neutral')
        if display_transform not in ('clip', 'pbr-neutral'):
            raise ValueError('display_transform must be clip or pbr-neutral')
        if display_transform != 'clip' and not resident:
            raise ValueError('pbr-neutral display_transform requires resident=True')
        self.display_transform = display_transform
        if type(msaa_samples) is not int:
            raise TypeError('msaa_samples must be a strict integer, not bool')
        if msaa_samples not in (1,4):
            raise ValueError('msaa_samples must be 1 or 4')
        if msaa_samples!=1 and not resident:
            raise UnsupportedRequestError('4x MSAA requires resident=True')
        self.msaa_samples=msaa_samples
        if type(area_light_grid) is not int:
            raise TypeError('area_light_grid must be an integer, not bool')
        if area_light_grid not in (2, 4, 8):
            raise ValueError('area_light_grid must be 2, 4 or 8')
        if area_light_grid != 2 and not resident:
            raise UnsupportedRequestError('dense area-light quadrature requires resident=True')
        if not isinstance(shadow_mode,str):
            raise TypeError('shadow_mode must be software or ray_query')
        if shadow_mode not in ('software','ray_query'):
            raise ValueError('shadow_mode must be software or ray_query')
        if shadow_mode=='ray_query' and not resident:
            raise UnsupportedRequestError('hardware ray query shadows require resident=True')
        if not isinstance(reflection_mode,str):
            raise TypeError('reflection_mode must be environment or single_bounce')
        if reflection_mode not in ('environment','single_bounce'):
            raise ValueError('reflection_mode must be environment or single_bounce')
        if reflection_mode=='single_bounce' and (not resident or shadow_mode!='ray_query'):
            raise UnsupportedRequestError('single_bounce reflection requires resident=True and shadow_mode=ray_query')
        self.reflection_mode=reflection_mode
        if not isinstance(indirect_mode,str):raise TypeError('indirect_mode must be none or diffuse_bounce')
        if indirect_mode not in ('none','diffuse_bounce'):raise ValueError('indirect_mode must be none or diffuse_bounce')
        if type(indirect_seed) is not int:raise TypeError('indirect_seed must be a uint32 integer, not bool')
        if not 0<=indirect_seed<=2**32-1:raise ValueError('indirect_seed outside uint32')
        if indirect_mode!='none' and (not resident or shadow_mode!='ray_query'):
            raise UnsupportedRequestError('diffuse_bounce requires resident=True and shadow_mode=ray_query')
        self.indirect_mode,self.indirect_seed=indirect_mode,indirect_seed
        if type(progressive) is not bool:
            raise TypeError('progressive must be bool')
        if type(indirect_bounces) is not int:
            raise TypeError('indirect_bounces must be an integer, not bool')
        if indirect_bounces not in (1, 2):
            raise ValueError('indirect_bounces must be 1 or 2')
        if progressive and (not resident or shadow_mode != 'ray_query' or indirect_mode != 'diffuse_bounce' or msaa_samples != 1):
            raise UnsupportedRequestError('progressive requires resident ray_query diffuse_bounce with msaa_samples=1')
        if not progressive and indirect_bounces != 1:
            raise UnsupportedRequestError('two diffuse bounces require progressive=True')
        if not isinstance(spatial_filter, str):
            raise TypeError('spatial_filter must be none or guided')
        if spatial_filter not in ('none', 'guided'):
            raise ValueError('spatial_filter must be none or guided')
        if spatial_filter != 'none' and not progressive:
            raise UnsupportedRequestError('guided spatial filtering requires progressive=True')
        self.spatial_filter = spatial_filter
        self.progressive = progressive
        self.indirect_bounces = indirect_bounces
        self.shadow_mode = shadow_mode
        self.area_light_grid = area_light_grid
        if studio is not None:
            if not isinstance(studio, StudioLighting):
                raise TypeError('studio must be StudioLighting or None')
            if not resident:
                raise UnsupportedRequestError('native studio requires resident=True; no legacy approximation')
        self.studio = studio
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
        self.shader_directory = (
            Path(shader_directory) if shader_directory is not None else self.library_path.parent
        )
        self.device_index, self.validation, self.exposure = device_index, validation, exposure

    def open_scene(self, scene: VisualScene) -> NativeRenderSession:
        if not isinstance(scene, VisualScene):
            raise TypeError("scene must be VisualScene")
        if not self.resident and any(isinstance(m,TexturedMaterial) for m in scene.materials):
            raise UnsupportedRequestError('native base-color textures require resident=True; no legacy approximation')
        reflection = reflection_declaration(scene,self.reflection_mode)
        indirect=indirect_declaration(self.indirect_mode,self.indirect_seed)
        require_library(self.library_path)
        if len(scene.lights) > 4:
            raise UnsupportedRequestError(
                "native raster currently supports at most four directional lights"
            )
        # Shared deterministic decoder; this is deliberately not a backend-local parser.
        from ..mesh_data import load_mesh

        meshes = {mesh.id: load_mesh(mesh) for mesh in scene.meshes}
        triangles = sum(len(meshes[item.mesh_id].triangles) for item in scene.instances)
        if not self.resident and triangles > 4096:
            raise UnsupportedRequestError("native raster currently supports at most 4096 triangles")
        device_type = (RayQueryResidentDevice if self.shadow_mode=='ray_query' else
                       (ResidentDevice if self.resident else NativeDevice))
        bound_materials={instance.material_id for instance in scene.instances}
        has_normal_maps=any(isinstance(material,TexturedMaterial) and material.normal_map is not None
                            and material.id in bound_materials for material in scene.materials)
        device = device_type(
            self.library_path, self.shader_directory, self.device_index, self.validation,
            **({'area_light_grid': self.area_light_grid} if self.area_light_grid != 2 else {}),
            **({'normal_maps': True} if has_normal_maps else {}),
            **({'reflection': True} if self.reflection_mode=='single_bounce' else {}),
            **({'indirect': True} if self.indirect_mode!='none' else {}),
            **({'progressive': True} if self.progressive else {}),
            **({'spatial': True} if self.spatial_filter == 'guided' else {})
        )
        try:
            display_info = configure_display(device, self.display_transform)
            if not self.resident and triangles * 3 > device.info()["max_vertices"]:
                raise UnsupportedRequestError("scene exceeds native device storage-buffer capacity")
            if self.resident:
                device.upload(*pack_scene(scene, meshes, device.info()))
            texture_info = None
            if self.resident:
                textures = pack_textures(scene,meshes,device.info())
                if textures is not None:
                    device.set_textures(textures)
                    texture_info = textures.diagnostics
            environment_info = None
            if self.studio is not None and self.studio.environment_map is not None:
                if not device.info().get('resident_environment_available',False):
                    raise BackendUnavailableError('native resident directional environment ABI/shaders are unavailable')
            studio_info = None
            if self.studio is not None:
                payload, studio_info = pack_studio(self.studio)
                device.set_studio(payload)
                if self.studio.environment_map is not None:
                    from ..environment import prepare_native_environment
                    environment = pack_environment(prepare_native_environment(self.studio.environment_map))
                    device.set_environment(environment)
                    environment_info = environment.diagnostics
                    studio_info['environment']['active_lighting'] = False
                    studio_info['environment']['replaced_by'] = 'directional environment map'
            normal_info = configure_normal_maps(device, pack_normal_maps(scene, meshes, device.info())) if self.resident else None
            reflection_info=configure_reflection(device,reflection)
            indirect_info=configure_indirect(device,indirect)
            msaa_info = configure_msaa(device, self.msaa_samples)
            progressive_history = configure_progressive(device, self.progressive, self.indirect_bounces)
            spatial_info = configure_spatial(device, self.spatial_filter)
            if self.progressive:
                indirect_info = dict(indirect_info)
                indirect_info['bounces'] = self.indirect_bounces
                indirect_info['terminal_lighting'] = (
                    'bounded 16 primary paths; optional one cosine continuation at each first hit; '
                    'replace that hit diffuse IBL, retain direct/specular environment; final hit never recurses'
                )
                indirect_info['seed_scope'] = 'seed + capture-local pixel + 0-based batch; state changes reset batch zero'
            return NativeRenderSession(scene, _NativeDelegate(
                scene, meshes, device, self.exposure, resident=self.resident, studio=studio_info, textures=texture_info, environment=environment_info, shadow_mode=self.shadow_mode, area_light_grid=self.area_light_grid, display_info=display_info, msaa_info=msaa_info, normal_info=normal_info, reflection_info=reflection_info, indirect_info=indirect_info, progressive_history=progressive_history, spatial_info=spatial_info))
        except BaseException as primary:
            try:
                device.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup("native scene open and cleanup failed", [primary, cleanup])
            raise
