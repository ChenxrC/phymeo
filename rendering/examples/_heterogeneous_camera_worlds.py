"""Eight authored manipulation models, live TYX endpoint poses and two cameras per world."""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from _panda_scene import box, look_at, write_mesh
from _r4_archive import _round_mesh
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch, MuJoCoTyxModelSpec, MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.manipulation_tasks import H, manipulation_scenes
from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.cameras import Camera
from phymeo_rendering.device_pose_packing import DeviceCameraMount, DeviceVisualBatchBinding, DeviceVisualBinding
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import DirectionalLight, Instance, Pose, VisualScene


WORLD_KEYS = (41, 7, 305, 11, 203, 59, 997, 2)
FIXED_POSE = look_at((.84, -.85, .56), (.21, 0., .10), (0., 0., 1.))
TOOL_FROM_CAMERA = look_at((-.12, -.22, .16), (.005, .018, -.14), (0., 0., 1.))
MOUNT_ID = 'wrist'


def _configuration_hash(model):
    """Fingerprint physical arrays independent of world keys and XML/model names."""
    digest = sha256()
    for name in ('body_parentid', 'body_pos', 'body_quat', 'body_mass', 'body_inertia',
                 'jnt_type', 'jnt_bodyid', 'jnt_axis', 'jnt_range', 'jnt_limited',
                 'geom_bodyid', 'geom_type', 'geom_pos', 'geom_quat', 'geom_size',
                 'dof_damping', 'dof_armature', 'actuator_trnid', 'actuator_gear'):
        a = np.ascontiguousarray(getattr(model, name))
        digest.update(f'{name}|{a.dtype.str}|{a.shape}|'.encode('ascii'))
        digest.update(a.tobytes())
    return digest.hexdigest()


def _registered_visual(authored, model, directory):
    import mujoco as mj

    kinds = {int(mj.mjtGeom.mjGEOM_PLANE): 'plane', int(mj.mjtGeom.mjGEOM_BOX): 'box',
             int(mj.mjtGeom.mjGEOM_SPHERE): 'sphere', int(mj.mjtGeom.mjGEOM_CAPSULE): 'capsule'}
    directory.mkdir(exist_ok=True, parents=True)
    meshes, materials, instances, rows, nodes = [], [], [], [], {}
    with TemporaryDirectory(dir=directory) as temp:
        for i in range(model.ngeom):
            name = model.geom(i).name
            kind = kinds[int(model.geom_type[i])]
            size = tuple(map(float, model.geom_size[i]))
            if kind == 'plane':
                # MuJoCo's physical plane is infinite. Use the authored visual
                # x/y half extents, with the same local z=0 and upward normals.
                x, y = size[:2]
                if min(x, y) <= 0:
                    raise ValueError('expected authored positive visual plane half extents')
                write_mesh(temp, name, ((-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0)),
                           ((0, 0, 1),)*4, ((0, 1, 2), (0, 2, 3)))
            elif kind == 'box':
                box(temp, name, tuple(2*s for s in size))
            else:
                _round_mesh(temp, name, kind, size)
            source = Path(temp)/(name+'.gltf')
            digest = sha256(source.read_bytes()).hexdigest()
            final = directory/f'{name}-{digest}.gltf'
            # Publishing complete immutable bytes lets multiple subsets reuse
            # one asset directory without observing a partially written glTF.
            if not final.exists() or sha256(final.read_bytes()).hexdigest() != digest:
                os.replace(source, final)
            meshes.append(Mesh(name, final.resolve().as_uri()))
            materials.append(Material(name, tuple(map(float, model.geom_rgba[i, :3])), 0., .65))
            body_id = int(model.geom_bodyid[i])
            node = model.body(body_id).name
            nodes[node] = body_id
            w, x, y, z = map(float, model.geom_quat[i])
            pose = Pose(tuple(map(float, model.geom_pos[i])), (x, y, z, w))
            instances.append(Instance(i+1, name, name, node, pose))
            rows.append(dict(instance_id=i+1, geom=name, body=node, body_id=body_id,
                             primitive=kind, native_size=list(size), mesh_sha256=digest))
    scene = VisualScene(f'camera-p4-v1:{authored.source_sha256}', tuple(meshes), tuple(materials), tuple(instances),
                        (DirectionalLight((.36, -.48, -.8), (1., 1., 1.)),))
    mount = DeviceCameraMount(MOUNT_ID, int(model.body('tool').id), TOOL_FROM_CAMERA)
    return scene, nodes, mount, rows


@dataclass
class CameraWorlds:
    keys: tuple
    batch: MuJoCoTyxHeterogeneousBatch
    workspace: object
    stream: object
    models: dict
    scenes: dict
    node_bodies: dict
    mounts: dict
    bindings: dict
    batch_binding: DeviceVisualBatchBinding
    controls: dict
    manifest: dict

    def step(self, count=1):
        if type(count) is not int or count < 0:
            raise ValueError('step count must be a nonnegative integer')
        for _ in range(count):
            if self.batch.step(H) != dict.fromkeys(self.keys, True):
                raise RuntimeError('TYX failed to accept every requested world step')
        return self.batch.last_step

    def snapshot(self):
        import torch

        frame = self.workspace.refresh(expected_revision=self.batch.checkpoint_revision)
        with torch.cuda.stream(self.stream):
            return self.workspace.body_pose_snapshot(frame)

    def pack(self, snapshot):
        return self.batch_binding.pack(snapshot, producer_stream=self.stream)

    def checkpoint(self):
        return self.batch.checkpoint()

    def restore(self, checkpoint):
        return self.batch.restore(checkpoint, expected_revision=self.batch.checkpoint_revision)

    def replay(self, checkpoint, count):
        """Restore the held actions and endpoint, advance count real steps, then observe."""
        if type(count) is not int or count < 0:
            raise ValueError('replay count must be a nonnegative integer')
        self.restore(checkpoint)
        self.step(count)
        return self.snapshot()

    def reset(self, keys=None):
        selected = self.keys if keys is None else tuple(keys)
        if not selected or any(type(k) is not int or k not in self.keys for k in selected) or len(set(selected)) != len(selected):
            raise ValueError('reset expects a nonempty subset of the active world keys')
        receipt = self.batch.reset(selected)
        # Native reset clears held inputs on selected owners. Restore the same
        # constant action used by the eight-model 32-step physics qualification.
        self.batch.set_controls(self.controls)
        return receipt

    def requests(self, key, session, exposure, *, size=(320, 240)):
        width, height = size
        if exposure.key.world_id != str(key) or exposure.key.scene_key != self.scenes[key].key:
            raise ValueError('camera requests need an exposure of the requested world and scene')
        if type(width) is not int or type(height) is not int or min(width, height)<1:
            raise ValueError('expected positive integer camera dimensions')
        lens = dict(width=width, height=height, fx_px=.96*width, fy_px=.96*width,
                    cx_px=width/2, cy_px=height/2, near_m=.012, far_m=4.)
        fixed = Camera('fixed', exposure.key, FIXED_POSE, **lens)
        mounted = session.mounted_camera(exposure, self.mounts[key].id, **lens)
        return tuple(CaptureRequest(camera, RenderPath.RASTER, tuple(Channel)) for camera in (fixed, mounted))


@contextmanager
def open_worlds(asset_dir, keys=WORLD_KEYS, *, device=0):
    """Select any nonempty ordered subset. Original keys select models even in singleton runs."""
    import torch

    keys = tuple(keys)
    if not keys or any(type(k) is not int or k not in WORLD_KEYS for k in keys) or len(set(keys)) != len(keys):
        raise ValueError('keys must be an ordered, unique, nonempty subset of WORLD_KEYS')
    if type(device) is not int or device<0:
        raise ValueError('device must be a CUDA ordinal')
    catalog = dict(zip(WORLD_KEYS, manipulation_scenes(), strict=True))
    specs, worlds, models, scenes, nodes, mounts, records = [], [], {}, {}, {}, {}, []
    for key in keys:
        authored = catalog[key]
        spec = MuJoCoTyxModelSpec(authored.key, compile_mujoco_xml(authored.mjcf, require_asset=False))
        model = spec.compilation.fresh_native_model()
        models[key] = model
        specs.append(spec)
        worlds.append(MuJoCoTyxWorldSpec(key, authored.key))
        scenes[key], nodes[key], mounts[key], geometry = _registered_visual(
            authored, model, Path(asset_dir)/authored.key)
        records.append(dict(world_key=key, model_key=authored.key, scene_key=scenes[key].key,
                            robot_joint_kinds=authored.robot_kinds, physical_sha256=_configuration_hash(model),
                            source_sha256=authored.source_sha256, nq=int(model.nq), nv=int(model.nv),
                            nu=int(model.nu), nbody=int(model.nbody), ngeom=int(model.ngeom),
                            mounted_body='tool', mounted_body_id=mounts[key].body_id, geometry=geometry))
    physical_count = len({r['physical_sha256'] for r in records})
    robot_count = len({r['robot_joint_kinds'] for r in records})
    if physical_count != len(keys) or robot_count != len(keys):
        raise ValueError('selected native models must have distinct physical configurations and arm joint sequences')
    with ExitStack() as stack:
        stack.enter_context(torch.cuda.device(device))
        batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(tuple(specs), tuple(worlds), mode='cuda'))
        workspace = stack.enter_context(batch.prepare_device_workspace(include_bodies=True))
        if workspace.binding['device'] != device or batch.world_keys != keys or batch.timestep != H:
            raise RuntimeError('actual TYX workspace differs from the requested CUDA device, key order, or timestep')
        stream = torch.cuda.Stream(device=device)
        bindings = {key: DeviceVisualBinding(scenes[key], workspace, key, nodes[key], camera_mounts=(mounts[key],))
                    for key in keys}
        binding = DeviceVisualBatchBinding(tuple(bindings[key] for key in keys))
        controls = {key: MuJoCoControl(np.full(models[key].nu, .025, np.float32)) for key in keys}
        batch.set_controls(controls)
        for row in records:
            layout = batch.layout[row['world_key']]
            row.update(compiled_model_sha256=layout.model_sha256, archive_sha256=layout.archive_sha256,
                       primary_sha256=layout.primary_sha256, bucket=layout.bucket)
        manifest = dict(world_keys=list(keys), world_count=len(keys), model_count=len(specs),
                        physical_configuration_count=physical_count, arm_joint_sequence_count=robot_count,
                        backend=batch.capabilities['backend'], native_bucket_count=batch.capabilities['bucket_count'],
                        timestep_s=H, worlds=records, fixed_world_pose=asdict(FIXED_POSE),
                        wrist_body_from_camera=asdict(TOOL_FROM_CAMERA),
                        camera_intrinsics='square pixels fx=fy=0.96*width; centered principal point; clip .012 to 4 m',
                        source='phymeo.training.manipulation_tasks.manipulation_scenes',
                        control='constant 0.025 float32 per actuator, no external generalized or body force',
                        initial='native qpos0, zero velocity/activation and time',
                        visuals='native opaque geoms only; source plane half extents; 12-sector round mesh; sites omitted')
        yield CameraWorlds(keys, batch, workspace, stream, models, scenes, nodes, mounts, bindings,
                           binding, controls, manifest)
