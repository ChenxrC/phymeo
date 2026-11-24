"""Pack accepted CUDA body poses into the resident rasterizer's std430 transforms."""
from dataclasses import dataclass
from itertools import accumulate
from math import hypot, sqrt

from .scene import FrameKey, Pose, VisualScene


@dataclass(frozen=True, slots=True)
class DeviceCameraMount:
    """Camera in the local frame of a physics body, including nonvisual bodies."""

    id: str
    body_id: int
    local_pose: Pose = Pose()

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("camera mount requires a nonblank id")
        if type(self.body_id) is not int or self.body_id < 0 or not isinstance(self.local_pose, Pose):
            raise ValueError("invalid camera body or local pose")


def _metadata_pose(rows):
    # Choose the largest quaternion component to avoid division by a small
    # trace near a half turn. Only the public metadata uses this conversion.
    r = rows
    trace = r[0][0] + r[1][1] + r[2][2]
    squares = [1 + 2*r[i][i] - trace for i in range(3)] + [1 + trace]
    i = max(range(4), key=squares.__getitem__)
    q = [0.] * 4
    q[i] = sqrt(squares[i]) / 2
    scale = 4*q[i]
    if i == 3:
        for j in range(3):
            a, b = (j+1)%3, (j+2)%3
            q[j] = (r[b][a]-r[a][b])/scale
    else:
        a, b = (i+1)%3, (i+2)%3
        q[a] = (r[a][i]+r[i][a])/scale
        q[b] = (r[b][i]+r[i][b])/scale
        q[3] = (r[b][a]-r[a][b])/scale
    length = hypot(*q)
    return Pose(tuple(row[3] for row in rows), tuple(value/length for value in q))


class DeviceVisualBinding:
    """Static instance-to-body binding for one scene and one native physics world.

    node_bodies maps every visual node to a native body id. The world body (0)
    supports fixed visuals, and multiple instances may attach to one body.
    CUDA Torch is loaded only when a device binding is constructed.
    """

    def __init__(self, scene, workspace, world_key, node_bodies, *, camera_mounts=()):
        import torch
        from .backends.native import _rotation

        source = workspace.binding
        if not isinstance(scene, VisualScene) or not scene.instances:
            raise ValueError("expected a nonempty resident visual scene")
        if source["mode"] != "cuda" or not source["endpoint_body_kinematics"]:
            raise ValueError("binding requires CUDA accepted endpoint body observations")
        self.scene, self.world_key = scene, world_key
        self._source = source
        self._device = torch.device("cuda", source["device"])
        models = {k: model for k, model, _, _ in source["source"]["worlds"]}
        keys = source["world_keys"]
        self._row = keys.index(world_key)
        counts = [len(workspace.model_metadata[models[key]]["body_names"]) for key in keys]
        self._offsets = tuple(accumulate(counts, initial=0))
        if set(node_bodies) != {instance.node_id for instance in scene.instances}:
            raise ValueError("node-to-body mapping must cover the scene's exact visual nodes")
        if any(type(i) is not int or not 0 <= i < counts[self._row] for i in node_bodies.values()):
            raise ValueError("visual node references an invalid native body")
        self.camera_mounts = tuple(camera_mounts)
        if (len(self.camera_mounts) > 64 or
                any(not isinstance(m, DeviceCameraMount) or m.body_id >= counts[self._row] for m in self.camera_mounts)):
            raise ValueError("invalid camera mount layout or body")
        self._camera_slots = {mount.id: i for i, mount in enumerate(self.camera_mounts)}
        if len(self._camera_slots) != len(self.camera_mounts):
            raise ValueError("duplicate camera mount id")
        records = [(node_bodies[i.node_id], i.local_pose) for i in scene.instances]
        records.extend((m.body_id, m.local_pose) for m in self.camera_mounts)
        indices, locals_ = [], []
        for body_id, pose in records:
            indices.append(self._offsets[self._row] + body_id)
            locals_.append([(*axis, p) for axis, p in zip(_rotation(pose), pose.position_m, strict=True)]
                           + [(0., 0., 0., 1.)])
        self._indices = torch.tensor(indices, dtype=torch.int64, device=self._device)
        self._local = torch.tensor(locals_, dtype=torch.float64, device=self._device)
        self._init_done = torch.cuda.Event()
        self._init_done.record(torch.cuda.current_stream(self._device))
        self.layout_upload_bytes = self._indices.nbytes + self._local.nbytes
        self.packs = self.status_readback_bytes = self.packed_output_bytes = 0
        self.camera_metadata_readback_bytes = 0

    def camera_slot(self, camera_id):
        return self._camera_slots.get(camera_id)

    def camera_metadata(self, packed, slot):
        rows = packed[len(self.scene.instances)+slot, 0, :, :3].T.contiguous().cpu()
        self.camera_metadata_readback_bytes += rows.nbytes
        return _metadata_pose(rows.tolist())

    def frame_key(self, snapshot):
        frame = snapshot.frame
        if frame.binding is not self._source or snapshot.body_offsets != self._offsets:
            raise ValueError("foreign workspace or incompatible body layout")
        return FrameKey(self.scene.key, str(self.world_key), frame.world_boundaries[self._row][1],
                        frame.revision, frame.ticks[self._row], "accepted_endpoint")

    def pack(self, snapshot):
        """Return two contiguous column-major mat4 values per instance on the caller's stream."""
        import torch

        self.frame_key(snapshot)
        src = snapshot.transforms
        if (not isinstance(src, torch.Tensor) or src.device != self._device
                or src.dtype != torch.float32 or tuple(src.shape) != (self._offsets[-1], 3, 4)
                or not src.is_contiguous()):
            raise ValueError("expected contiguous accepted FP32 body transforms on the bound CUDA device")
        torch.cuda.current_stream(self._device).wait_event(self._init_done)
        # Accumulate the local rigid transform in double, then invert the actual
        # rounded FP32 world matrix, matching resident_transforms in resident.inc.
        parent = src.index_select(0, self._indices).to(torch.float64)
        world = torch.bmm(parent, self._local).to(torch.float32)
        rotation = world[:, :, :3].to(torch.float64)
        a, b, c = rotation.unbind(dim=2)
        cross_bc = torch.linalg.cross(b, c, dim=1)
        det = (a * cross_bc).sum(dim=1)
        cofactor = torch.stack((cross_bc, torch.linalg.cross(c, a, dim=1),
                                torch.linalg.cross(a, b, dim=1)), dim=1)
        inverse = (cofactor / det[:, None, None]).to(torch.float32)
        inverse_p = -torch.bmm(inverse.to(torch.float64), world[:, :, 3:].to(torch.float64))
        count = self._local.shape[0]
        packed = torch.zeros((count, 2, 4, 4), dtype=torch.float32, device=self._device)
        packed[:, 0, :, :3] = world.transpose(1, 2)
        packed[:, 1, :3, :3] = inverse.transpose(1, 2)
        packed[:, 1, 3, :3] = inverse_p.squeeze(2)
        packed[:, :, 3, 3] = 1.

        gram = torch.bmm(rotation.transpose(1, 2), rotation)
        unit = torch.eye(3, device=self._device, dtype=torch.float64)
        valid = torch.isfinite(packed).all() & (det > 0).all() & ((gram - unit).abs() <= 1e-4).all()
        self.status_readback_bytes += valid.element_size()
        if not valid.item():
            raise ValueError("nonfinite, scaled, sheared or reflected visual transform")
        self.packs += 1
        self.packed_output_bytes += packed.nbytes
        return packed
