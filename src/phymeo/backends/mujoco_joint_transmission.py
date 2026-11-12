"""Immutable native JOINT transmission and same-step device force publication."""

from dataclasses import dataclass

import numpy as np

from ._mujoco_tyx_runtime import MuJoCoTyxStepReceipt


def _freeze(value, dtype):
    array = np.ascontiguousarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite joint transmission publication")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class MuJoCoTyxJointTransmission:
    """Complete native-address moments and actual scalar/generalized device forces.

    ids columns are actor, joint, joint type, qpos address, DOF address and width.
    Original gear and requested control retain FP64; stored moments and all
    computed forces retain their actual FP32 representation. Zero moments remain.
    """

    ids: np.ndarray
    gear: np.ndarray
    row_offsets: np.ndarray
    dof_indices: np.ndarray
    moment: np.ndarray
    input_control: np.ndarray
    clamped_control: np.ndarray
    raw_force: np.ndarray
    scalar_force: np.ndarray
    generalized_force: np.ndarray
    schema: str = "native-joint-motor-transmission-v1"
    phase: str = "tyx-euler-entry-joint-transmission"

    def __post_init__(self):
        integer = ("ids", "row_offsets", "dof_indices")
        original = ("gear", "input_control")
        for name in self.__dataclass_fields__:
            if name not in ("schema", "phase"):
                dtype = (
                    np.int32 if name in integer else np.float64 if name in original else np.float32
                )
                object.__setattr__(self, name, _freeze(getattr(self, name), dtype))
        if self.ids.ndim != 2 or self.ids.shape[1] != 6:
            raise ValueError("joint transmission ids shape")
        nu = len(self.ids)
        if not 1 <= nu <= 32 or self.gear.shape != (nu, 6):
            raise ValueError("joint transmission actor layout")
        if self.row_offsets.shape != (nu + 1,) or self.row_offsets[0] != 0:
            raise ValueError("joint transmission CSR offsets")
        nnz = int(self.row_offsets[-1])
        if not 1 <= nnz <= 96 or self.dof_indices.shape != (nnz,) or self.moment.shape != (nnz,):
            raise ValueError("joint transmission CSR entries")
        if not np.array_equal(np.diff(self.row_offsets), self.ids[:, 5]):
            raise ValueError("joint transmission CSR row widths")
        if not np.isin(self.ids[:, 5], (1, 3)).all():
            raise ValueError("joint transmission native width")
        if self.generalized_force.ndim != 1 or not 1 <= self.generalized_force.size <= 8:
            raise ValueError("joint transmission generalized force shape")
        if np.any(self.dof_indices < 0) or np.any(self.dof_indices >= self.generalized_force.size):
            raise ValueError("joint transmission DOF address")
        for name in ("input_control", "clamped_control", "raw_force", "scalar_force"):
            if getattr(self, name).shape != (nu,):
                raise ValueError("joint transmission actor field shape")
        if (
            self.schema != "native-joint-motor-transmission-v1"
            or self.phase != "tyx-euler-entry-joint-transmission"
        ):
            raise ValueError("joint transmission publication schema")


@dataclass(frozen=True, kw_only=True)
class MuJoCoTyxJointMotorStepReceipt(MuJoCoTyxStepReceipt):
    transmission: MuJoCoTyxJointTransmission


def publish_joint_transmission(plan, result, index, input_control):
    """Bind the same mechanics invocation; do not recompute an actuator force."""
    layout = plan.joint_transmission
    fields = result["joint_transmission_fields"][index, : plan.nu]
    return MuJoCoTyxJointTransmission(
        ids=layout.ids,
        gear=layout.gear,
        row_offsets=layout.row_offsets,
        dof_indices=layout.dof_indices,
        moment=layout.moment,
        input_control=input_control,
        clamped_control=fields[:, 0],
        raw_force=fields[:, 1],
        scalar_force=fields[:, 2],
        generalized_force=result["actuator"][index, : plan.nv],
    )
