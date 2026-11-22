"""SAPIEN 3.0.3 pure-render implementation, isolated from physics ownership.

API/source anchors (tag 3.0.3): wrapper/scene.py explicit systems;
src/sapien_renderer/camera_component.cpp synchronous CPU getImage; shipped
vulkan_shader/default/composite0.frag and rt/camera.rgen buffer semantics.
No viewer, actor builder, PhysX system, physics step, or CUDA interop is used.
"""
from contextlib import contextmanager
from hashlib import sha256
from math import isfinite
from pathlib import Path
from struct import pack, unpack
from threading import RLock

from ..outputs import Capture, Channel, ImagePlane
from ..pipeline import RenderPath, validate_capture_request
from ..scene import Pose, compose_pose
from . import BackendUnavailableError, UnsupportedRequestError
from ._sapien_pixels import encode_linear_rgb


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


class SapienSession:
    def __init__(self, source, config, sapien, np):
        self.source, self.config, self.sapien, self.np = source, config, sapien, np
        self.scene = None
        self.system = None
        self.instances = []
        self.lights = []
        self.ids = {}
        self.closed = False
        self.shader = "default" if config.path is RenderPath.RASTER else "rt"
        self.shader_directory = Path(sapien.__file__).resolve().parent / "vulkan_shader" / self.shader
        self.rgb_target = "Lighting" if config.path is RenderPath.RASTER else "HdrColor"
        try:
            # Decode using the same bounded loader as the native backend before
            # uploading. SAPIEN's independent URI/asset loader is never used.
            from ..mesh_data import load_mesh
            decoded = {mesh.id: load_mesh(mesh) for mesh in source.meshes}
            self.system = sapien.render.RenderSystem(config.device)
            self.scene = sapien.Scene([self.system])
            self.system.ambient_light = (0.0, 0.0, 0.0)
            materials = {}
            for material in source.materials:
                target = sapien.render.RenderMaterial()
                target.base_color = (*material.base_color, 1.0)
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
                shape = sapien.render.RenderShapeTriangleMesh(
                    positions, np.asarray(mesh.triangles, dtype=np.uint32), normals,
                    np.zeros((len(positions), 2), dtype=np.float32), materials[instance.material_id])
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
                self.lights.append(self.scene.add_directional_light(
                    light.direction, light.illuminance, shadow=True, shadow_map_size=2048))
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
                    raise RuntimeError(f"SAPIEN shader lacks targets {sorted(required - set(names))}")
                pixels = probe.get_picture(self.rgb_target)
                if pixels.shape != (1, 1, 4) or not np.isfinite(pixels).all():
                    raise RuntimeError("SAPIEN shader probe returned invalid linear pixels")
            self.diagnostics = {
                "backend": "sapien-reference", "library_version": sapien.__version__,
                "numpy_version": np.__version__, "requested_path": config.path.value,
                "effective_path": config.path.value, "requested_device": config.device,
                "device_name": self.system.device.name, "device_pci": self.system.device.pci_string,
                "device_description": str(self.system.device),
                "driver_version": None, "vulkan_api_version": None,
                "physics_systems": [], "physics_steps": 0,
                "shader_pack": self.shader, "shader_directory": str(self.shader_directory),
                "shader_sha256": _shader_hash(self.shader_directory),
                "linear_rgb_target": self.rgb_target,
                "display": "exposure=1; clamp linear [0,1]; standard sRGB OETF; half-up UNORM8",
                "ambient_linear_rgb": [0.0, 0.0, 0.0], "shadow_map_size": 2048,
                "readback": "blocking SAPIEN get_picture CPU download, then owned bytes",
                "samples_per_pixel": config.samples_per_pixel if self.shader == "rt" else 1,
                "max_bounces": config.max_bounces if self.shader == "rt" else None,
                "denoiser": "none", "seed": None, "seed_support": "not-user-configurable",
                "seed_note": "rt shader uses fixed magic plus pixel/spp/frameCount; no cross-device bitwise claim",
                "accumulation": "fresh camera per exposure; no temporal history",
                "supported_channels": ([Channel.RGB.value] if self.shader == "rt"
                                       else [channel.value for channel in Channel]),
                "pt_clip_limit": "all uploaded vertices must lie inside near+0.002 < depth < far and camera range <9999m",
                "resolution_limit": 4096,
                "shader_probe": "selected shader executed and CPU target read back",
                "global_config_note": "camera default mutations serialized/restored inside this adapter only",
            }
        except Exception as exc:
            self.close()
            if isinstance(exc, (ValueError, TypeError)):
                raise
            raise BackendUnavailableError(
                f"SAPIEN {self.shader} reference initialization failed: {exc}") from exc

    def _pose(self, public):
        # compose_pose normalizes only the derived calculation, keeping the
        # user's validated immutable exposure and scene values unchanged.
        pose = compose_pose(Pose(), public)
        p = tuple(float(value) for value in pose.position_m)
        q = tuple(float(value) for value in pose.orientation_xyzw)
        if not all(isfinite(_f32(value)) for value in (*p, *q)):
            raise ValueError("pose is not representable by the SAPIEN FP32 transform")
        return self.sapien.Pose(p, (q[3], q[0], q[1], q[2]))

    @contextmanager
    def _camera(self, width, height, public=None):
        render = self.sapien.render
        entity = self.sapien.Entity()
        camera = render.RenderCameraComponent(width, height, str(self.shader_directory))
        entity.add_component(camera)
        if public is not None:
            camera.set_perspective_parameters(public.near_m, public.far_m, public.fx_px,
                                             public.fy_px, public.cx_px, public.cy_px, 0.0)
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
        converted = tuple(_f32(getattr(camera, name)) for name in
                          ("fx_px", "fy_px", "cx_px", "cy_px", "near_m", "far_m"))
        return (all(isfinite(x) for x in converted) and converted[0] > 0 and converted[1] > 0
                and 0 < converted[4] < converted[5])

    def _world_vertices(self, frame):
        np = self.np
        nodes = {node.node_id: node.world_pose for node in frame.nodes}
        points = []
        poses = []
        for instance, entity, vertices in self.instances:
            pose = compose_pose(nodes[instance.node_id], instance.local_pose)
            x, y, z, w = pose.orientation_xyzw
            rotation = np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                                 (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                                 (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))
            points.append(vertices @ rotation.T + np.asarray(pose.position_m))
            poses.append((entity, self._pose(nodes[instance.node_id])))
        return np.concatenate(points) if points else np.empty((0, 3)), poses

    def _admit_pt(self, points, camera):
        if not len(points):
            return
        np = self.np
        pose = compose_pose(Pose(), camera.world_pose)
        x, y, z, w = pose.orientation_xyzw
        forward = np.array((-2*(x*z+y*w), -2*(y*z-x*w), -(1-2*(x*x+y*y))))
        delta = points - np.asarray(pose.position_m)
        depths = delta @ forward
        # The upstream shader starts at near with tMin=.001 and does not
        # enforce far. Full triangle vertices strictly inside this slab prove
        # no triangle crosses either boundary; distance bounds its tMax.
        if (not np.isfinite(delta).all() or np.any(depths <= camera.near_m + 0.002)
                or np.any(depths >= camera.far_m)
                or np.any(np.linalg.norm(delta, axis=1) >= 9999.0)):
            raise UnsupportedRequestError(
                "SAPIEN PT clip admission requires every scene vertex strictly inside "
                "near+0.002m/far and within 9999m; shipped shader is not general clipping")

    def capture(self, frame, request):
        if not self.supports(request):
            raise UnsupportedRequestError("SAPIEN session does not support this complete request")
        np = self.np
        points, poses = self._world_vertices(frame)
        if request.path is RenderPath.PATH_TRACE:
            self._admit_pt(points, request.camera)
        for entity, pose in poses:
            entity.pose = pose
        if len(points):
            center = (points.min(axis=0) + points.max(axis=0)) * 0.5
            extent = max(0.01, float(np.linalg.norm(points - center, axis=1).max()) * 1.01)
            for light in self.lights:
                light.pose = self.sapien.Pose(center, light.pose.q)
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
                planes[Channel.RGB] = encode_linear_rgb(linear[..., :3].flat)
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
            return Capture(frame.key, frame.time_s, request.camera,
                           tuple(ImagePlane(channel, planes[channel]) for channel in request.channels))

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.scene is not None:
            self.scene.clear()
        self.instances.clear()
        self.lights.clear()
        self.ids.clear()
        self.scene = None
        self.system = None
