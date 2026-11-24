"""Pack accepted CUDA body poses into the resident rasterizer's std430 transforms."""
from itertools import accumulate

from .scene import FrameKey, VisualScene


class DeviceVisualBinding:
    """Static instance-to-body binding for one scene and one native physics world.

    node_bodies maps every visual node to a native body id. The world body (0)
    supports fixed visuals, and multiple instances may attach to one body.
    CUDA Torch is loaded only when a device binding is constructed.
    """

    def __init__(self, scene, workspace, world_key, node_bodies):
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
        indices, locals_ = [], []
        for instance in scene.instances:
            indices.append(self._offsets[self._row] + node_bodies[instance.node_id])
            pose = instance.local_pose
            locals_.append([(*axis, p) for axis, p in zip(_rotation(pose), pose.position_m, strict=True)]
                           + [(0., 0., 0., 1.)])
        self._indices = torch.tensor(indices, dtype=torch.int64, device=self._device)
        self._local = torch.tensor(locals_, dtype=torch.float64, device=self._device)
        self._init_done = torch.cuda.Event()
        self._init_done.record(torch.cuda.current_stream(self._device))
        self.layout_upload_bytes = self._indices.nbytes + self._local.nbytes
        self.packs = self.status_readback_bytes = self.packed_output_bytes = 0

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
        count = len(self.scene.instances)
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
