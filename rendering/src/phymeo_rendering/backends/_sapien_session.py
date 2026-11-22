"""SAPIEN 3.0.3 pure-render implementation, isolated from physics ownership.

API/source anchors (tag 3.0.3): wrapper/scene.py explicit systems;
src/sapien_renderer/camera_component.cpp synchronous CPU getImage; shipped
vulkan_shader/default/composite0.frag and rt/camera.rgen buffer semantics.
No viewer, actor builder, PhysX system, physics step, or CUDA interop is used.
"""

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from math import isfinite
from pathlib import Path
from struct import pack, unpack
from threading import RLock
from tempfile import TemporaryDirectory
import sys

from ..assets import TexturedMaterial
from ..outputs import Capture, Channel, ImagePlane
from ..pipeline import RenderPath, validate_capture_request
from ..scene import Pose, compose_pose
from . import BackendUnavailableError, UnsupportedRequestError
from ._sapien_pixels import encode_linear_image
from ._sapien_studio import StudioResources, geometry_budget, validate_scene_slots

# SAPIEN camera entities: +X forward, +Y left, +Z up. Public cameras:
# +X right, +Y up, -Z forward. Columns in public axes are (0,0,-1),
# (-1,0,0), (0,1,0); this is a proper rotation, not a reflected basis.
_ENTITY_CAMERA_IN_PUBLIC = Pose(orientation_xyzw=(-0.5, 0.5, 0.5, 0.5))
# The engine reads these process-global defaults while attaching a camera.
# Serialize this adapter's create/restore scopes. Applications using SAPIEN
# directly must not concurrently mutate its globals outside this adapter.
_CAMERA_DEFAULTS_LOCK = RLock()


def _shader_hash(directory):
    digest = sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _f32(value):
    try:
        return unpack("<f", pack("<f", float(value)))[0]
    except (OverflowError, ValueError):
        return float("inf")


# Separate CPU decoded and actual texture-upload budgets. A nonwhite linear
# factor requires 16 bytes/texel, not the 4 bytes/texel of an sRGB8 upload.
_MAX_TEXTURE_RGB_BYTES = 64 * 1024 * 1024
_MAX_TEXTURE_UPLOAD_BYTES = 128 * 1024 * 1024


def _prepare_textures(source, decoded):
    """Validate bindings and decode once per descriptor, before any GPU object.

    Equal file bytes at different URIs still have each descriptor checked. Once
    verified, immutable decoded bytes share a SHA identity; upload identity also
    includes the material's linear factor, so two uses cannot overwrite colour.
    """
    from ..texture_data import load_texture

    materials = {material.id: material for material in source.materials}
    bindings = []
    for instance in source.instances:
        if isinstance(materials[instance.material_id], TexturedMaterial):
            mesh = decoded[instance.mesh_id]
            if mesh.texcoords is None:
                raise ValueError(f"textured material {instance.material_id!r} requires TEXCOORD_0 on mesh {instance.mesh_id!r}")
            bindings.append({"instance_id": instance.id, "material_id": instance.material_id,
                             "mesh_id": instance.mesh_id, "mesh_source_sha256": mesh.source_sha256})
    by_descriptor, by_sha, by_material = {}, {}, {}
    uploads = {}
    rgb_bytes = upload_bytes = 0
    for material in source.materials:
        if not isinstance(material, TexturedMaterial):
            continue
        descriptor = material.base_color_map
        if descriptor not in by_descriptor:
            image = load_texture(descriptor)
            if image.source_sha256 not in by_sha:
                rgb_bytes += len(image.rgb8)
                if rgb_bytes > _MAX_TEXTURE_RGB_BYTES:
                    raise ValueError("texture decoded RGB byte budget exceeded")
                by_sha[image.source_sha256] = image
            by_descriptor[descriptor] = by_sha[image.source_sha256]
        image = by_descriptor[descriptor]
        by_material[material.id] = image
        factor = tuple(float(value) for value in material.base_color)
        key = (image.source_sha256, factor)
        if key not in uploads:
            white = factor == (1., 1., 1.)
            size = image.width * image.height * (4 if white else 16)
            upload_bytes += size
            if upload_bytes > _MAX_TEXTURE_UPLOAD_BYTES:
                raise ValueError("texture upload byte budget exceeded")
            uploads[key] = {
                "source_sha256": image.source_sha256,
                "decoded_rgb8_sha256": sha256(image.rgb8).hexdigest(),
                "width": image.width, "height": image.height, "factor": list(factor),
                "format": "R8G8B8A8Unorm" if white else "R32G32B32A32Sfloat",
                "srgb": white, "upload_bytes": size, "materials": [], "uris": [],
            }
        uploads[key]["materials"].append(material.id)
        if descriptor.uri not in uploads[key]["uris"]:
            uploads[key]["uris"].append(descriptor.uri)
    evidence = {
        "uploads": list(uploads.values()), "bindings": bindings,
        "decoded_rgb8_bytes": rgb_bytes, "upload_bytes": upload_bytes,
        "sampling": {"filter": "linear", "address": "repeat", "mipmap_levels": 1,
                     "uv_origin": "upper-left; unmodified glTF TEXCOORD_0"},
        "factor_policy": "white: sRGB8 hardware decode; nonwhite: standard EOTF times linear factor, FP32 texture; material uniform white",
        "shader_semantics": "SAPIEN 3.0.3 default/gbuffer.frag and rt/camera.rchit replace baseColor when diffuse texture exists",
        "limits": {"decoded_rgb8_bytes": _MAX_TEXTURE_RGB_BYTES,
                   "upload_bytes": _MAX_TEXTURE_UPLOAD_BYTES},
    }
    return by_material, evidence


def _texture_pixels(image, factor, np):
    """Prepare actual API pixels without an extra sRGB8 quantization.

    Both shipped 3.0.3 shader paths *replace* baseColor with the sampled texture.
    A nonwhite factor therefore travels in linear texels. Bilinear filtering
    commutes with this constant multiplication. The tiny FP64 LUT is rounded to
    FP32 once per channel; image dimensions do not enlarge the LUT.
    """
    rgb = np.frombuffer(image.rgb8, dtype=np.uint8).reshape(image.height, image.width, 3)
    if tuple(factor) == (1., 1., 1.):
        result = np.empty((image.height, image.width, 4), dtype=np.uint8)
        result[..., :3] = rgb
        result[..., 3] = 255
        return result, "R8G8B8A8Unorm", True
    lut = np.asarray([
        [(encoded / 12.92 if encoded <= .04045 else ((encoded + .055) / 1.055) ** 2.4) * float(value)
         for value in factor]
        for encoded in (index / 255. for index in range(256))
    ], dtype=np.float32)
    result = np.empty((image.height, image.width, 4), dtype=np.float32)
    for channel in range(3):
        result[..., channel] = lut[rgb[..., channel], channel]
    result[..., 3] = 1.
    return result, "R32G32B32A32Sfloat", False

class SapienSession:
    def __init__(self, source, config, sapien, np):
        self.source, self.config, self.sapien, self.np = source, config, sapien, np
        self.scene = None
        self.system = None
        self.instances = []
        self.lights = []
        self.ids = {}
        self.closed = False
        self._studio = None
        self._textures = {}
        self._sampling_temp = None
        self.diagnostics = {}
        self._lifecycle = {
            "studio_temporary_files_exist": False,
            "closed": False,
            "completed_captures": 0,
            "close_error": None,
            "attached_entities": None,
            "attached_cameras": None,
            "attached_render_bodies": None,
            "attached_lights": None,
            "attached_entities_before_clear": None,
            "attached_entities_after_clear": None,
            "attached_cameras_after_clear": None,
            "attached_render_bodies_after_clear": None,
            "attached_lights_after_clear": None,
            "scope": "this scene attachments only; no driver/cache/process-memory release claim",
        }
        self.shader = "default" if config.path is RenderPath.RASTER else "rt"
        self.shader_directory = (
            Path(sapien.__file__).resolve().parent / "vulkan_shader" / self.shader
        )
        self.rgb_target = "Lighting" if config.path is RenderPath.RASTER else "HdrColor"
        try:
            official_hash = _shader_hash(self.shader_directory)
            if config.sampling_mode == "shifted-halton":
                from ._sapien_sampling import make_seeded_pack

                self._sampling_temp = TemporaryDirectory(prefix="phymeo-rt-sampling-")
                self.shader_directory = make_seeded_pack(
                    self.shader_directory, Path(self._sampling_temp.name) / "rt", config.seed
                )
            # Decode using the same bounded loader as the native backend before
            # uploading. SAPIEN's independent URI/asset loader is never used.
            from ..mesh_data import load_mesh

            validate_scene_slots(len(source.meshes), len(source.instances))
            decoded = {mesh.id: load_mesh(mesh) for mesh in source.meshes}
            geometry = geometry_budget(
                tuple((len(decoded[item.mesh_id].positions), len(decoded[item.mesh_id].triangles))
                      for item in source.instances), mesh_slots=len(source.meshes))
            material_images, texture_evidence = _prepare_textures(source, decoded)
            texture_evidence["actual_uploads"] = []
            self.system = sapien.render.RenderSystem(config.device)
            self.scene = sapien.Scene([self.system])
            self.system.ambient_light = (0.0, 0.0, 0.0)
            materials = {}
            for material in source.materials:
                target = sapien.render.RenderMaterial()
                target.base_color = (*material.base_color, 1.0)
                if isinstance(material, TexturedMaterial):
                    image = material_images[material.id]
                    factor = tuple(float(value) for value in material.base_color)
                    key = (image.source_sha256, factor)
                    if key not in self._textures:
                        pixels, format_name, srgb = _texture_pixels(image, factor, np)
                        # Official 3.0.3 array constructor copies owned bytes;
                        # no SDK filename loader, rescale or vertical flip.
                        uploaded = sapien.render.RenderTexture2D(
                            pixels, format_name, mipmap_levels=1,
                            filter_mode="linear", address_mode="repeat", srgb=srgb,
                        )
                        actual = (uploaded.width, uploaded.height, uploaded.filter_mode,
                                  uploaded.address_mode, uploaded.mipmap_levels, uploaded.is_srgb)
                        if actual != (image.width, image.height, "linear", "repeat", 1, srgb):
                            raise RuntimeError("SAPIEN texture metadata disagrees with requested sampling")
                        self._textures[key] = uploaded
                        texture_evidence["actual_uploads"].append({
                            "source_sha256": image.source_sha256, "factor": list(factor),
                            "width": int(uploaded.width), "height": int(uploaded.height),
                            "format": uploaded.format, "srgb": bool(uploaded.is_srgb),
                            "filter": uploaded.filter_mode, "address": uploaded.address_mode,
                            "mipmap_levels": int(uploaded.mipmap_levels),
                            "cpu_upload_sha256": sha256(pixels.tobytes(order="C")).hexdigest(),
                            "cpu_upload_bytes": int(pixels.nbytes),
                        })
                    target.base_color_texture = self._textures[key]
                    target.base_color = (1., 1., 1., 1.)
                target.metallic = float(material.metallic)
                target.roughness = float(material.roughness)
                target.specular = 0.5
                materials[material.id] = target
            for instance in source.instances:
                mesh = decoded[instance.mesh_id]
                positions = np.asarray(mesh.positions, dtype=np.float32)
                normals = np.asarray(mesh.normals, dtype=np.float32)
                if not np.isfinite(positions).all() or not np.isfinite(normals).all():
                    raise ValueError(f"mesh {instance.mesh_id!r} is not representable in FP32")
                texcoords = (np.zeros((len(positions), 2), dtype=np.float32)
                             if mesh.texcoords is None else np.asarray(mesh.texcoords, dtype=np.float32))
                if not np.isfinite(texcoords).all():
                    raise ValueError(f"mesh {instance.mesh_id!r} TEXCOORD_0 is not representable in FP32")
                shape = sapien.render.RenderShapeTriangleMesh(
                    positions,
                    np.asarray(mesh.triangles, dtype=np.uint32),
                    normals,
                    texcoords,
                    materials[instance.material_id],
                )
                shape.local_pose = self._pose(instance.local_pose)
                body = sapien.render.RenderBodyComponent()
                body.attach(shape)
                entity = sapien.Entity()
                entity.name = f"phymeo-instance-{instance.id}"
                entity.add_component(body)
                self.scene.add_entity(entity)
                render_id = int(shape.per_scene_id)
                if render_id <= 0 or render_id in self.ids:
                    raise RuntimeError("SAPIEN produced an invalid or duplicate render ID")
                self.ids[render_id] = instance.id
                # Geometry admission uses precisely the FP32 uploaded vertices.
                self.instances.append((instance, entity, positions.astype(np.float64)))
            for light in source.lights:
                # The wrapper's add_directional_light still writes deprecated
                # component.pose in 3.0.3. Own the Entity transform directly.
                component = sapien.render.RenderDirectionalLightComponent()
                entity = sapien.Entity()
                entity.add_component(component)
                component.color = light.illuminance
                component.shadow = True
                component.shadow_map_size = 2048
                component.shadow_half_size = 10.0
                component.shadow_near = -10.0
                component.shadow_far = 10.0
                entity.pose = sapien.Pose(
                    (0.0, 0.0, 0.0), sapien.math.shortest_rotation((1.0, 0.0, 0.0), light.direction)
                )
                self.scene.add_entity(entity)
                self.lights.append(component)
            if config.studio is not None:
                self._studio = StudioResources(sapien, self.scene, config.studio)
                self._lifecycle["studio_temporary_files_exist"] = self._studio.temporary_files_exist
            # Compile and execute this selected shader independently. Merely
            # importing SAPIEN or finding an RT-capable device is insufficient.
            with self._camera(1, 1) as probe:
                self.scene.update_render()
                probe.take_picture()
                names = probe.get_picture_names()
                required = {self.rgb_target}
                if config.path is RenderPath.RASTER:
                    required.update(("Position", "Segmentation"))
                if not required.issubset(names):
                    raise RuntimeError(
                        f"SAPIEN shader lacks targets {sorted(required - set(names))}"
                    )
                pixels = probe.get_picture(self.rgb_target)
                if pixels.shape != (1, 1, 4) or not np.isfinite(pixels).all():
                    raise RuntimeError("SAPIEN shader probe returned invalid linear pixels")
            self.diagnostics = {
                "geometry": geometry,
                "textures": texture_evidence,
                "studio": deepcopy(self._studio.diagnostics) if self._studio is not None else None,
                "backend": "sapien-reference",
                "library_version": sapien.__version__,
                "sapien_scene_id": int(self.scene.id),
                "numpy_version": np.__version__,
                "requested_path": config.path.value,
                "effective_path": config.path.value,
                "requested_device": config.device,
                "device_name": self.system.device.name,
                "device_pci": self.system.device.pci_string,
                "device_description": str(self.system.device),
                "driver_version": None,
                "vulkan_api_version": None,
                "physics_systems": [],
                "physics_steps": 0,
                "shader_pack": self.shader,
                "shader_directory": str(self.shader_directory),
                "shader_sha256": _shader_hash(self.shader_directory),
                "official_shader_sha256": official_hash,
                "linear_rgb_target": self.rgb_target,
                "display": (f"exposure={config.exposure:g}; clamp linear [0,1]; standard sRGB OETF; half-up UNORM8"
                            if config.display_transform == "clip" else
                            f"exposure={config.exposure:g}; nonnegative linear Rec.709; Khronos PBR Neutral; standard sRGB OETF; half-up UNORM8"),
                "display_transform": config.display_transform,
                "display_transform_source": (None if config.display_transform == "clip" else
                    "https://github.com/KhronosGroup/ToneMapping/blob/b5a2eed5ddf6c2227090449399de9c7affb9e4c9/PBR_Neutral/pbrNeutral.glsl"),
                "exposure": float(config.exposure),
                "ambient_linear_rgb": [0.0, 0.0, 0.0],
                "shadow_map_size": 2048,
                "readback": "blocking SAPIEN get_picture CPU download, then owned bytes",
                "samples_per_pixel": config.samples_per_pixel if self.shader == "rt" else 1,
                "max_bounces": config.max_bounces if self.shader == "rt" else None,
                "denoiser": "none",
                "seed": config.seed,
                "seed_support": "not-user-configurable" if self._sampling_temp is None else "seeded-local-shader",
                "seed_note": "sampling_pattern specifies the actual shader and device scope",
                "sampling_pattern": self._sampling_pattern(),
                "accumulation": "fresh camera per exposure; no temporal history",
                "supported_channels": (
                    [Channel.RGB.value]
                    if self.shader == "rt"
                    else [channel.value for channel in Channel]
                ),
                "pt_clip_limit": "all vertices inside near+0.002/far and range<9999m, with FP32 clearance",
                "pt_precision_policy": "8*FP32-epsilon*transform-scale clearance; reject >=1% scene span; pragmatic admission, not a global error proof",
                "resolution_limit": 4096,
                "shader_probe": "selected shader executed and CPU target read back",
                "global_config_note": "camera default mutations serialized/restored inside this adapter only",
            }
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                exc.add_note(f"SAPIEN initialization cleanup also failed: {cleanup_error}")
            if isinstance(exc, (ValueError, TypeError, UnsupportedRequestError)):
                raise
            raise BackendUnavailableError(
                f"SAPIEN {self.shader} reference initialization failed: {exc}"
            ) from exc

    def _sampling_pattern(self):
        if self.shader != "rt":
            return None
        # Describe the actual installed pack. Do not advertise this sequence
        # for a different shader merely because its package version matches.
        source = (self.shader_directory / "camera.rgen").read_text(encoding="utf-8")
        if self.config.sampling_mode == "shifted-halton":
            return {
                "mode": "shifted-halton", "seed": self.config.seed,
                "primary_pixel_sequence": "per-pixel seed-dependent 2D shift modulo one of Halton bases 2 and 3",
                "path_rng": "TEA(pixel index + width*height*spp*frameCount, seed)",
                "frame_count_policy": "fresh camera per exposure",
                "arbitrary_seed_supported": True, "cross_device_bitwise_guarantee": False,
            }
        if (
            "uint magic = 0x9e3779b9;" not in source
            or "halton2d(frameCount * spp + i, 2, 3)" not in source
            or "gl_LaunchSizeEXT.x * gl_LaunchSizeEXT.y * spp * frameCount" not in source
        ):
            raise BackendUnavailableError(
                "installed RT shader lacks the audited fixed sampling pattern"
            )
        return {
            "mode": "official-fixed-sequence-fresh-camera",
            "magic_hex": "0x9e3779b9",
            "primary_pixel_sequence": "Halton bases 2 and 3; frameCount*spp+i",
            "path_rng": "TEA(pixel index + width*height*spp*frameCount, magic)",
            "frame_count_policy": "first render of a newly constructed camera each exposure",
            "arbitrary_seed_supported": False,
            "scope": "same installed shader/device/config; repeated public bytes audited in device test",
            "cross_device_bitwise_guarantee": False,
        }

    def _pose(self, public):
        # compose_pose normalizes only the derived calculation, keeping the
        # user's validated immutable exposure and scene values unchanged.
        pose = compose_pose(Pose(), public)
        p = tuple(float(value) for value in pose.position_m)
        q = tuple(float(value) for value in pose.orientation_xyzw)
        if not all(isfinite(_f32(value)) for value in (*p, *q)):
            raise UnsupportedRequestError("SAPIEN FP32 precision cannot represent this pose")
        return self.sapien.Pose(p, (q[3], q[0], q[1], q[2]))

    @contextmanager
    def _camera(self, width, height, public=None):
        render = self.sapien.render
        entity = self.sapien.Entity()
        camera = render.RenderCameraComponent(width, height, str(self.shader_directory))
        entity.add_component(camera)
        if public is not None:
            camera.set_perspective_parameters(
                public.near_m,
                public.far_m,
                public.fx_px,
                public.fy_px,
                public.cx_px,
                public.cy_px,
                0.0,
            )
            entity.pose = self._pose(compose_pose(public.world_pose, _ENTITY_CAMERA_IN_PUBLIC))
        if self.shader == "rt":
            camera.set_property("spp", self.config.samples_per_pixel)
            camera.set_property("maxDepth", self.config.max_bounces)
            camera.set_property("aperture", 0.0)
            camera.set_property("focusPlane", 1.0)
            camera.set_property("russianRoulette", 0)
        added = False
        try:
            with _CAMERA_DEFAULTS_LOCK:
                previous_msaa = render.get_msaa()
                previous_denoiser = render.get_ray_tracing_denoiser()
                try:
                    render.set_msaa(1)
                    render.set_ray_tracing_denoiser("none")
                    self.scene.add_entity(entity)
                    added = True
                finally:
                    render.set_msaa(previous_msaa)
                    render.set_ray_tracing_denoiser(previous_denoiser)
            yield camera
        finally:
            if added:
                self.scene.remove_entity(entity)

    def supports(self, request):
        validate_capture_request(request)
        camera = request.camera
        if request.path is not self.config.path or camera.width > 4096 or camera.height > 4096:
            return False
        if request.path is RenderPath.PATH_TRACE and request.channels != (Channel.RGB,):
            return False
        if not all(isfinite(_f32(value)) for value in camera.world_pose.position_m):
            return False
        converted = tuple(
            _f32(getattr(camera, name))
            for name in ("fx_px", "fy_px", "cx_px", "cy_px", "near_m", "far_m")
        )
        return (
            all(isfinite(x) for x in converted)
            and converted[0] > 0
            and converted[1] > 0
            and 0 < converted[4] < converted[5]
        )

    def _world_vertices(self, frame):
        np = self.np
        nodes = {node.node_id: node.world_pose for node in frame.nodes}
        points = []
        poses = []
        transform_scale = 1.0
        for instance, entity, vertices in self.instances:
            transform_scale = max(
                transform_scale,
                *(abs(float(value)) for value in nodes[instance.node_id].position_m),
                *(abs(float(value)) for value in instance.local_pose.position_m),
                float(np.abs(vertices).max()),
            )
            pose = compose_pose(nodes[instance.node_id], instance.local_pose)
            x, y, z, w = pose.orientation_xyzw
            rotation = np.array(
                (
                    (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                    (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                    (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
                )
            )
            points.append(vertices @ rotation.T + np.asarray(pose.position_m))
            poses.append((entity, self._pose(nodes[instance.node_id])))
        return (np.concatenate(points) if points else np.empty((0, 3))), poses, transform_scale

    def _admit_pt(self, points, camera, transform_scale):
        if not len(points):
            return
        np = self.np
        pose = compose_pose(Pose(), camera.world_pose)
        x, y, z, w = pose.orientation_xyzw
        forward = np.array((-2 * (x * z + y * w), -2 * (y * z - x * w), -(1 - 2 * (x * x + y * y))))
        delta = points - np.asarray(pose.position_m)
        depths = delta @ forward
        scale = max(
            transform_scale,
            float(np.abs(points).max()),
            *(abs(float(value)) for value in pose.position_m),
        )
        # A small, conservative guard for this FP32 reference adapter, not an
        # all-domain floating-point proof. Include the separate source/local
        # translations so large cancelling transforms cannot hide their scale.
        clearance = 8.0 * float(np.finfo(np.float32).eps) * scale
        span = float(np.ptp(points, axis=0).max())
        if not isfinite(clearance) or clearance >= 0.01 * span:
            raise UnsupportedRequestError(
                "SAPIEN PT FP32 precision is too coarse for this scene span"
            )
        # The upstream shader starts at near with tMin=.001 and does not
        # enforce far. Keep all vertices, including those outside the view,
        # away from both boundaries with the practical FP32 clearance above.
        if (
            not np.isfinite(delta).all()
            or np.any(depths <= max(camera.near_m, _f32(camera.near_m)) + 0.002 + clearance)
            or np.any(depths >= min(camera.far_m, _f32(camera.far_m)) - clearance)
            or np.any(np.linalg.norm(delta, axis=1) >= 9999.0 - clearance)
        ):
            raise UnsupportedRequestError(
                "SAPIEN PT clip admission requires every scene vertex strictly inside "
                "near+0.002m/far with FP32 clearance and within 9999m; "
                "shipped shader is not general clipping"
            )

    def capture(self, frame, request):
        if not self.supports(request):
            raise UnsupportedRequestError("SAPIEN session does not support this complete request")
        np = self.np
        points, poses, transform_scale = self._world_vertices(frame)
        if request.path is RenderPath.PATH_TRACE:
            self._admit_pt(points, request.camera, transform_scale)
        for entity, pose in poses:
            entity.pose = pose
        if len(points):
            center = (points.min(axis=0) + points.max(axis=0)) * 0.5
            extent = max(0.01, float(np.linalg.norm(points - center, axis=1).max()) * 1.01)
            for light in self.lights:
                light.entity.pose = self.sapien.Pose(center, light.entity.pose.q)
                light.shadow_half_size = extent
                light.shadow_near = -extent
                light.shadow_far = extent
        with self._camera(request.camera.width, request.camera.height, request.camera) as camera:
            self.scene.update_render()
            camera.take_picture()
            planes = {}
            if Channel.RGB in request.channels:
                linear = camera.get_picture(self.rgb_target)
                if linear.shape != (request.camera.height, request.camera.width, 4):
                    raise ValueError("SAPIEN linear RGB readback shape mismatch")
                planes[Channel.RGB] = encode_linear_image(linear[..., :3], exposure=self.config.exposure,
                                                          display_transform=self.config.display_transform)
            if any(channel is not Channel.RGB for channel in request.channels):
                position = camera.get_picture("Position")
                segmentation = camera.get_picture("Segmentation")
                expected = (request.camera.height, request.camera.width, 4)
                if position.shape != expected or segmentation.shape != expected:
                    raise ValueError("SAPIEN sensor readback shape mismatch")
                if segmentation.dtype.kind != "u" or segmentation.dtype.itemsize != 4:
                    raise ValueError("SAPIEN segmentation must be uint32, not floating color")
                render_ids = segmentation[..., 0]
                if not np.isfinite(position[..., 3]).all():
                    raise ValueError("SAPIEN visibility depth is not finite")
                valid = render_ids != 0
                if not np.array_equal(valid, position[..., 3] < 1.0):
                    raise ValueError("SAPIEN segmentation and position visibility disagree")
                public_ids = np.zeros(render_ids.shape, dtype="<u4")
                for identity in np.unique(render_ids):
                    if identity == 0:
                        continue
                    if int(identity) not in self.ids:
                        raise ValueError(f"unknown SAPIEN render ID {int(identity)}")
                    public_ids[render_ids == identity] = self.ids[int(identity)]
                depth = np.where(valid, -position[..., 2], np.inf).astype("<f4")
                if not np.isfinite(depth[valid]).all() or np.any(depth[valid] <= 0):
                    raise ValueError("SAPIEN hit depth must be finite and positive metres")
                planes[Channel.INSTANCE_ID] = public_ids.tobytes(order="C")
                planes[Channel.FORWARD_DEPTH] = depth.tobytes(order="C")
                planes[Channel.VALID] = valid.astype("u1").tobytes(order="C")
            result = Capture(
                frame.key,
                frame.time_s,
                request.camera,
                tuple(ImagePlane(channel, planes[channel]) for channel in request.channels),
            )
        self._lifecycle["completed_captures"] += 1
        return result

    def _attachment_counts(self):
        return {
            "attached_entities": len(self.scene.get_entities()),
            "attached_cameras": len(self.system.cameras),
            "attached_render_bodies": len(self.system.render_bodies),
            "attached_lights": len(self.system.lights),
        }

    def diagnostics_snapshot(self):
        result = deepcopy(self.diagnostics)
        observation = deepcopy(self._lifecycle)
        if self.scene is not None:
            observation.update(self._attachment_counts())
        result["lifecycle"] = observation
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._lifecycle["closed"] = True
        try:
            if self.scene is not None:
                self._lifecycle.update(self._attachment_counts())
                self._lifecycle["attached_entities_before_clear"] = self._lifecycle[
                    "attached_entities"
                ]
                self.scene.clear()
                # Observe the same real Scene/System before dropping Python
                # handles. These are not guessed zeros from an empty list owned
                # by this adapter, and they say nothing about driver caches.
                counts = self._attachment_counts()
                self._lifecycle.update(counts)
                self._lifecycle.update(
                    {f"{key}_after_clear": value for key, value in counts.items()}
                )
                if any(counts.values()):
                    raise RuntimeError(
                        "SAPIEN scene clear left attached render entities/components"
                    )
        except Exception as exc:
            self._lifecycle["close_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._textures.clear()
            self.instances.clear()
            self.lights.clear()
            self.ids.clear()
            self.scene = None
            self.system = None
            if self._sampling_temp is not None:
                self._sampling_temp.cleanup()
            if self._studio is not None:
                pending_error = sys.exception()
                try:
                    self._studio.close()
                except Exception as cleanup_error:
                    if pending_error is not None:
                        pending_error.add_note(f"studio source cleanup also failed: {cleanup_error}")
                    else:
                        self._lifecycle["close_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                        raise
                finally:
                    self._lifecycle["studio_temporary_files_exist"] = self._studio.temporary_files_exist
