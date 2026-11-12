"""Public immutable device actuation evidence for admitted native domains."""

from dataclasses import dataclass

import numpy as np

from ._mujoco_tyx_runtime import MuJoCoTyxStepReceipt


def _freeze(value, dtype):
    array = np.ascontiguousarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite actuation publication")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class MuJoCoTyxActuation:
    """Actual input, integrated activation and force from the same device attempt.

    derivative is the native raw act_dot. integration_derivative differs for
    disabled groups (zero), and neither includes the subsequent actrange clamp.
    ids columns: qpos,dof,joint,actadr,early,disabled,ctrl/act/force-limited.
    parameters columns: gear,gain,bias[3],ctrl[2],force[2],tau_eff,act[2].
    """

    ids: np.ndarray
    parameters: np.ndarray
    input_act: np.ndarray
    output_act: np.ndarray
    activation_derivative: np.ndarray
    integration_derivative: np.ndarray
    clamped_control: np.ndarray
    force_input: np.ndarray
    raw_force: np.ndarray
    scalar_force: np.ndarray
    generalized_force: np.ndarray
    schema: str = "native-scalar-filterexact-actuation-v1"
    phase: str = "tyx-euler-entry-actuation-and-next-activation"

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name not in ("schema", "phase"):
                object.__setattr__(
                    self,
                    name,
                    _freeze(getattr(self, name), np.int32 if name == "ids" else np.float32),
                )
        nu, na = self.ids.shape[0], self.input_act.size
        if self.ids.shape != (nu, 9) or self.parameters.shape != (nu, 12):
            raise ValueError("actuation source layout")
        for name in ("input_act", "output_act", "activation_derivative", "integration_derivative"):
            if getattr(self, name).shape != (na,):
                raise ValueError("activation publication shape")
        for name in ("clamped_control", "force_input", "raw_force", "scalar_force"):
            if getattr(self, name).shape != (nu,):
                raise ValueError("actuator publication shape")
        if self.generalized_force.ndim != 1:
            raise ValueError("generalized actuator force shape")
        if (
            self.schema
            not in (
                "native-scalar-filterexact-actuation-v1",
                "native-joint-filterexact-actuation-v1",
                "native-free-ball-joint-filterexact-actuation-v1",
            )
            or self.phase != "tyx-euler-entry-actuation-and-next-activation"
        ):
            raise ValueError("actuation publication schema")
        if self.schema == "native-free-ball-joint-filterexact-actuation-v1" and not (
            1 <= nu <= 32 and 1 <= na <= 32 and 1 <= self.generalized_force.size <= 35
        ):
            raise ValueError("free joint actuation publication capacity")


@dataclass(frozen=True, kw_only=True)
class MuJoCoTyxActivationStepReceipt(MuJoCoTyxStepReceipt):
    actuation: MuJoCoTyxActuation


def _actuation_schema(plan):
    """Derive the publication domain from the admitted plan, never the payload."""
    if getattr(plan, "joint_transmission", None) is not None:
        if plan.kind == 3 and getattr(plan, "native_joint_layout", False):
            return "native-free-ball-joint-filterexact-actuation-v1"
        return "native-joint-filterexact-actuation-v1"
    return "native-scalar-filterexact-actuation-v1"


def publish_actuation(plan, result, index):
    """Bind actual downloaded arrays; never recompute a force or state."""
    layout = plan.activation
    actor = result["actuator_fields"][index, : plan.nu]
    return MuJoCoTyxActuation(
        ids=layout.ids,
        parameters=layout.values,
        input_act=result["input_act"][index, : layout.na],
        output_act=result["act"][index, : layout.na],
        activation_derivative=result["activation_derivative"][index, : layout.na],
        integration_derivative=result["integration_derivative"][index, : layout.na],
        clamped_control=actor[:, 0],
        force_input=actor[:, 1],
        raw_force=actor[:, 2],
        scalar_force=actor[:, 3],
        generalized_force=result["actuator"][index, : plan.nv],
        schema=_actuation_schema(plan),
    )
