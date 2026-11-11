"""Owned native initialization, separate from structural qpos0 and motor caps."""

from dataclasses import dataclass

import numpy as np

INTEGRATION_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "act",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "eq_active",
    "mocap_pos",
    "mocap_quat",
)


@dataclass(frozen=True)
class NativeInitialState:
    """Full MuJoCo integration input snapshot tied to the compiled MJB identity.

    Uses native q/DOF indices and WXYZ quaternion order. qfrc_applied is an
    independent external generalized force, never an actuator feedforward.
    Derived acceleration/contact buffers are deliberately not stored here.
    """

    model_sha256: str
    fields: tuple[tuple[str, np.ndarray], ...]

    def __post_init__(self):
        if tuple(name for name, _ in self.fields) != INTEGRATION_FIELDS:
            raise ValueError("initial state must contain every integration field in schema order")
        owned = []
        for name, value in self.fields:
            a = np.asarray(value)
            if a.dtype.kind not in "fbiu" or not np.isfinite(a).all():
                raise ValueError(f"initial {name} must be finite numeric data")
            owned.append((name, np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)))
        object.__setattr__(self, "fields", tuple(owned))

    @classmethod
    def from_data(cls, compilation, data):
        """Capture caller-owned CPU data; validate against a fresh model at use."""
        return cls(
            compilation.source.model_sha256,
            tuple((name, np.asarray(getattr(data, name))) for name in INTEGRATION_FIELDS),
        )

    def apply(self, compilation, data):
        if self.model_sha256 != compilation.source.model_sha256:
            raise ValueError("initial state belongs to another compiled model")
        # Complete validation precedes any writes, including FP32 representability.
        for name, value in self.fields:
            target = np.asarray(getattr(data, name))
            with np.errstate(over="ignore", invalid="ignore"):
                finite = np.isfinite(value.astype(np.float32)).all()
            if value.shape != target.shape or not finite:
                raise ValueError(f"initial {name} shape or float32 representation is invalid")
            if name == "eq_active" and not np.isin(value, [0, 1]).all():
                raise ValueError("initial eq_active requires boolean values")
        fields = dict(self.fields)
        model = data.model
        quaternions = list(fields["mocap_quat"])
        # MuJoCo enum free=0, ball=1. This module stays free of eager MuJoCo imports.
        for kind, address in zip(model.jnt_type, model.jnt_qposadr, strict=True):
            if kind in (0, 1):
                start = int(address) + (3 if kind == 0 else 0)
                quaternions.append(fields["qpos"][start : start + 4])
        if any(abs(float(np.linalg.norm(q)) - 1.0) > 1e-10 for q in quaternions):
            raise ValueError("initial native quaternion must be unit WXYZ")
        for name, value in self.fields:
            if name == "time":
                data.time = float(value)
            else:
                getattr(data, name)[:] = value
