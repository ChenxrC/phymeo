"""Optional endpoint body query using existing native geometry kernels.

All writable scratch belongs to the public workspace. Static device fields and
accepted q/v are borrowed only for the duration of a locked refresh.
"""

import numpy as np

from ..mujoco_device_controls import _require
from ._model import fp32
from ._tree_model import native_frames

BODY_FIELDS = ("body_position", "body_rotation", "body_linear_velocity", "body_angular_velocity")


class BodyObserver:
    def __init__(self, workspace):
        from . import _floating_kernels, _tree_kernels
        from ._body_observation_kernels import pack_bodies

        self.workspace = workspace
        self.pack = pack_bodies
        self.scratch, self.static, self.floating = [], [], []
        batch = workspace._batch
        for bank in workspace._banks:
            for name in BODY_FIELDS:
                tail = (3, 3) if name == "body_rotation" else (3,)
                bank[name] = workspace._buffer(
                    np.zeros(
                        (len(batch.world_keys), workspace._dimensions["nbody"], *tail), np.float32
                    )
                )
        for owner in batch._buckets:
            self.floating.append(
                bool(np.any(owner._model.jnt_type == owner._mj.mjtJoint.mjJNT_FREE))
            )
            n, b, d = len(owner.world_keys), owner._model.nbody, max(8, owner._plan.nv)
            self.scratch.append(
                {
                    name: workspace._buffer(np.zeros(shape, np.float32))
                    for name, shape in (
                        ("motion", (n, b, 7, 3)),
                        ("rotation", (n, b, 3, 3)),
                        ("axes", (n, d, 3)),
                        ("anchors", (n, d, 3)),
                        ("work", (n, 8, 3)),
                        ("work_rotation", (n, 3, 3)),
                    )
                }
            )
            if hasattr(owner._device, "geometry"):
                self.static.append(None)
            else:
                # Legacy constant-mass routes have no tree buffers. Share only
                # static frame lowering, without repeating solver admission.
                model = owner._model
                body, joints, joint_data = native_frames(owner._mj, model)
                vectors = np.zeros((9, max(8, model.nq)), np.float32)
                vectors[7], vectors[8] = -np.inf, np.inf
                fields = {
                    "parents": np.asarray(model.body_parentid, np.int32),
                    "ranges": np.stack((model.body_jntadr, model.body_jntnum), axis=1).astype(
                        np.int32
                    ),
                    "body_data": fp32(body, "body observation frames"),
                    "joints": joints,
                    "joint_data": fp32(joint_data, "body observation native joint data"),
                    "vectors": vectors,
                }
                floating = bool(np.any(joints[:, 0] == 3))
                self.static.append(
                    (
                        _floating_kernels.geometry if floating else _tree_kernels.geometry,
                        floating,
                        {k: workspace._buffer(v) for k, v in fields.items()},
                    )
                )

    def refresh_bucket(self, index, owner, destination, options):
        ws, device = self.workspace, owner._device
        fields = self.static[index]
        if fields is None:
            geometry = device.geometry[device.active]
            floating = self.floating[index]
            inputs = [
                getattr(device, k)
                for k in ("parents", "ranges", "body_data", "joints", "joint_data", "vectors")
            ]
        else:
            geometry, floating, buffers = fields
            inputs = [
                buffers[k]
                for k in ("parents", "ranges", "body_data", "joints", "joint_data", "vectors")
            ]
        prefix = [len(owner.world_keys), owner._model.nbody, owner._plan.nv]
        if floating:
            prefix.append(owner._model.njnt)
        scratch = self.scratch[index]
        with ws._tx.use_runtime(ws._runtime):
            geometry(
                *prefix,
                *inputs,
                device.q[device.active],
                device.v[device.active],
                *(
                    scratch[k]
                    for k in ("motion", "rotation", "axes", "anchors", "work", "work_rotation")
                ),
                ws._status[index],
                **options,
            )
            ws._counts["kernels"] += 1
            self.pack(
                len(owner.world_keys),
                owner._model.nbody,
                ws._dimensions["nbody"],
                ws._rows[index],
                scratch["motion"],
                scratch["rotation"],
                *(ws._banks[destination][k] for k in BODY_FIELDS),
                ws._status[index],
                **options,
            )
            ws._counts["kernels"] += 1
            if ws._stream is None:
                ws._runtime.sync()
                status = ws._status[index].copy()
            else:
                ws._stream.synchronize()
                status = ws._status[index].to_numpy(stream=ws._stream)
                ws._stream.synchronize()
                ws._counts["status_downloaded_bytes"] += status.nbytes
            _require(not np.any(status), "invalid endpoint body observation")
