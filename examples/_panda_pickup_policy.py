"""Two-parameter pickup target correction from measured poses, not reset labels.

This checkpoint stores a target policy, not simulator/controller state. Gains
are direct finite numbers; an optimizer may impose its own search bounds.
Only world XY and world-Z heading are corrected. Height and tilt are nominal.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = "panda-pickup-policy-v1"
PARAMETER_NAMES = ("observed_xy_gain", "observed_yaw_gain")
OBSERVATION_CONTRACT = "settle-end/measured-part_grasp_center/world-xy-and-world-z-heading"
TARGET_CONTRACT = "nominal-height-and-tilt/unchanged-assumed-tcp-to-tip"


def _pose(value, name):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite 4x4 pose")
    if not np.array_equal(pose[3], [0., 0., 0., 1.]):
        raise ValueError(f"{name} must have a homogeneous last row")
    rotation = pose[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-6)
            or abs(float(np.linalg.det(rotation)) - 1.) > 1e-6):
        raise ValueError(f"{name} must contain a proper rotation")
    return pose


def observed_yaw_delta(nominal, observed):
    """Shortest signed world-Z change of the local +X heading, in [-pi, pi]."""
    a, b = _pose(nominal, "nominal"), _pose(observed, "observed")
    x, y = a[:2, 0], b[:2, 0]
    if np.linalg.norm(x) < 1e-8 or np.linalg.norm(y) < 1e-8:
        raise ValueError("pickup heading is undefined for a vertical local +X axis")
    return math.atan2(float(x[0]*y[1] - x[1]*y[0]), float(x @ y))


@dataclass(frozen=True)
class PickupPolicy:
    parameters: tuple[float, float] = (0., 0.)

    def __post_init__(self):
        try:
            values = tuple(self.parameters)
        except TypeError as exc:
            raise TypeError("pickup policy requires exactly two direct gains") from exc
        if len(values) != 2 or any(isinstance(x, (bool, np.bool_)) for x in values):
            raise ValueError("pickup policy requires exactly two finite numeric gains")
        if any(not isinstance(x, (int, float, np.integer, np.floating)) for x in values):
            raise TypeError("pickup gains must be numeric, not coerced strings")
        values = tuple(float(x) for x in values)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("pickup gains must be finite")
        object.__setattr__(self, "parameters", values)

    def target(self, nominal_grasp_world, observed_part_grasp_world):
        nominal = _pose(nominal_grasp_world, "nominal")
        observed = _pose(observed_part_grasp_world, "observed")
        result = nominal.copy()
        xy_gain, yaw_gain = self.parameters
        # Preserve the old target's actual floating-point values at zero gain.
        if xy_gain == 1.:
            result[:2, 3] = observed[:2, 3]
        elif xy_gain != 0.:
            result[:2, 3] += xy_gain * (observed[:2, 3] - nominal[:2, 3])
        if yaw_gain != 0.:
            angle = yaw_gain * observed_yaw_delta(nominal, observed)
            if not math.isfinite(angle):
                raise ValueError("pickup yaw correction overflowed")
            c, s = math.cos(angle), math.sin(angle)
            result[:3, :3] = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ nominal[:3, :3]
        if not np.isfinite(result).all():
            raise ValueError("pickup target correction overflowed")
        result.setflags(write=False)
        return result

    def to_checkpoint(self):
        return dict(schema=SCHEMA, parameter_names=list(PARAMETER_NAMES),
                    parameters=list(self.parameters), observation=OBSERVATION_CONTRACT,
                    target=TARGET_CONTRACT)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        expected = {"schema", "parameter_names", "parameters", "observation", "target"}
        if type(checkpoint) is not dict or set(checkpoint) != expected:
            raise ValueError("pickup checkpoint has missing or unknown fields")
        if (checkpoint["schema"] != SCHEMA
                or checkpoint["parameter_names"] != list(PARAMETER_NAMES)
                or checkpoint["observation"] != OBSERVATION_CONTRACT
                or checkpoint["target"] != TARGET_CONTRACT):
            raise ValueError("pickup checkpoint version, parameter order or coordinate contract differs")
        if type(checkpoint["parameters"]) is not list:
            raise ValueError("checkpoint parameters must be a two-element JSON array")
        return cls(checkpoint["parameters"])

    def save(self, path):
        """Create a new checkpoint; never overwrite a prior training receipt."""
        payload = json.dumps(self.to_checkpoint(), sort_keys=True, indent=2, allow_nan=False) + "\n"
        with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)

    @classmethod
    def load(cls, path):
        def unique_fields(pairs):
            result = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError(f"duplicate checkpoint field: {name}")
                result[name] = value
            return result

        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=unique_fields)
        return cls.from_checkpoint(value)
