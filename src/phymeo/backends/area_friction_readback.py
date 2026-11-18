"""Law-aware HOST decoding of mixed v2 accepted area observation rows.

The base reader owns phase, catalog, dA, body ownership, native-pair exclusion,
world-moment representation and full-payload digest validation. This subclass
extends its ABI/width, row-kind, constitutive payload and immutable sample hooks.
It neither authenticates device data nor replaces the complete receipt audit:
there is no M/J/common residual here, and no assertion that an exported relative
velocity was independently recomputed from the actual endpoint configuration.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..geometry.area_pressure import AreaPressureIdentity
from .area_friction_law import (
    LAW,
    PRECISION,
    SCHEMA,
    AreaFrictionPairLaw,
    capture_pair_laws,
    law_binding,
)
from .area_pressure_readback import ABI as PRESSURE_ABI
from .area_pressure_readback import AreaPressureEntryReader, AreaPressureSample, _require
from .mujoco_tyx._model import fp32

ABI = ("mujoco-tyx-area-device-entry-v2", *PRESSURE_ABI[1:])
NORMAL_LAW = "frozen-surface-elastic-backward-euler-v1"


@dataclass(frozen=True)
class AreaFrictionSample(AreaPressureSample):
    tangential_force_on_a_world: tuple
    relative_velocity_world: tuple | None
    mu: float | None
    slip_speed_m_s: float | None
    law: str


def _capture_binding(value, pairs, identities):
    if value is None:
        return None
    _require(type(value) is tuple and len(value) == 4, "friction law binding layout")
    _require(
        all(type(x) is str for x in value[:3]) and value[:3] == (LAW, SCHEMA, PRECISION),
        "friction law binding version",
    )
    rows = value[3]
    _require(
        type(rows) is tuple and 1 <= len(rows) <= 16 and len(rows) == len(pairs),
        "friction law binding pair count",
    )
    laws = []
    for row in rows:
        _require(type(row) is tuple and len(row) == 6, "friction law pair fields")
        laws.append(AreaFrictionPairLaw(AreaPressureIdentity(*row[:4]), row[4], row[5]))
    laws = capture_pair_laws(pairs, tuple(laws))
    for law in laws:
        _require(
            float(fp32(law.mu, "readback mu").item()) >= 0
            and float(fp32(law.slip_speed_m_s, "readback slip speed").item()) > 0,
            "packed friction parameter domain",
        )
    _require(
        tuple(law.identity for law in laws) == identities, "friction material identity binding"
    )
    _require(law_binding(laws) == value, "noncanonical friction law binding")
    return laws


class AreaFrictionEntryReader(AreaPressureEntryReader):
    """Read native/normal/friction worlds together through the explicit v2 ABI.

    Normal-only rows retain their original force semantics and digest data; their
    new eight columns are zero padding, so no actual relative velocity or friction
    parameter is claimed for those rows. Invalid/undeclared rows remain all zero.
    """

    ABI = ABI
    DATA_WIDTH = 23
    SAMPLE_TYPE = AreaFrictionSample

    def __init__(self, workspace, identities_by_model):
        super().__init__(workspace, identities_by_model)
        binding = self.frame_binding
        _require(
            binding.get("contact_tokens") is True
            and binding.get("entry_body_kinematics") is True
            and binding.get("entry_phase") == "accepted-step-input",
            "friction observation entry phase",
        )
        declared = binding.get("area_laws")
        pairs = binding["area_pair_names"]
        _require(
            isinstance(declared, Mapping) and set(declared) == set(pairs),
            "area_laws must cover every model, including native models",
        )
        laws = {
            key: _capture_binding(declared[key], pairs[key], self.identities.get(key, ()))
            for key in pairs
        }
        source = binding["source"]
        buckets = source["buckets"]
        _require(type(buckets) is tuple, "source buckets")
        by_world = {}
        for world, key, bucket, _ in source["worlds"]:
            _require(type(bucket) is int and 0 <= bucket < len(buckets), "law source bucket")
            row = buckets[bucket]
            _require(isinstance(row, Mapping), "law source bucket fields")
            expected = None if laws[key] is None else law_binding(laws[key])
            _require(row.get("area_friction") == expected, "law/source bucket mismatch")
            by_world[world] = laws[key]
        self._laws_by_world = MappingProxyType(by_world)
        self._law_bindings = MappingProxyType(
            {key: None if rows is None else law_binding(rows) for key, rows in laws.items()}
        )

    def validate(self, arrays, frame):
        _require(
            frame.binding.get("area_laws") == self._law_bindings,
            "captured law declarations changed",
        )
        rows = super().validate(arrays, frame)
        for row in rows.values():
            for start in range(0, len(row.samples), 3):
                triangle = row.samples[start : start + 3]
                _require(
                    len(triangle) == 3
                    and all(sample.area_m2 == triangle[0].area_m2 for sample in triangle),
                    "triangle quadrature area differs",
                )
        return rows

    def _entry_kind(self, world, valid):
        return valid == (1 if self._laws_by_world[world] is None else 2)

    def _validate_values(self, world, ordinal, values):
        laws = self._laws_by_world[world]
        if laws is None:
            _require(not np.any(values[:, 15:]), "normal-only row has friction extension")
            return super()._validate_values(world, ordinal, values)
        law = laws[ordinal]
        mu = float(fp32(law.mu, "readback mu").item())
        slip = float(fp32(law.slip_speed_m_s, "readback slip speed").item())
        _require(mu >= 0 and slip > 0, "packed friction parameter domain")
        _require(
            np.all(values[:, 21] == mu) and np.all(values[:, 22] == slip),
            "actual friction parameters differ from declaration",
        )
        p = values.astype(np.float64)
        normal, fn, force = p[:, 3:6], p[:, 8], p[:, 9:12]
        ft, relative = p[:, 15:18], p[:, 18:21]
        squared = np.sum(normal * normal, axis=1)
        _require(
            np.all(np.abs(squared - 1) <= 2e-6) and np.all(p[:, 7] <= 0),
            "friction normal/gap domain",
        )
        # Divide by n.n: packed normals are retained, not silently normalized.
        tangent = relative - normal * (np.sum(normal * relative, axis=1) / squared)[:, None]
        expected_ft = (
            -mu
            * fn[:, None]
            * tangent
            / np.sqrt(np.sum(tangent * tangent, axis=1) + slip * slip)[:, None]
        )
        expected_force = fn[:, None] * normal + ft
        allowance = 0.002 + 0.002 * np.abs(expected_ft)
        _require(np.all(np.abs(ft - expected_ft) <= allowance), "final-normal smooth tangent law")
        _require(
            np.all(np.abs(force - expected_force) <= 0.002 + 0.002 * np.abs(expected_force)),
            "total force differs from normal plus tangent",
        )
        _require(
            np.all(
                np.abs(np.sum(normal * ft, axis=1)) <= np.sum(np.abs(normal) * allowance, axis=1)
            ),
            "nontangent friction force",
        )
        limit = mu * fn
        _require(
            np.all(np.linalg.norm(ft, axis=1) <= limit + 0.002 + 0.002 * limit),
            "friction exceeds final normal capacity",
        )
        # Engineering representation bound, not an added exact power-sign gate.
        _require(
            np.all(np.sum(ft * tangent, axis=1) <= np.sum(allowance * np.abs(tangent), axis=1)),
            "positive friction power",
        )

    def _sample_extra(self, world, ordinal, value):
        if self._laws_by_world[world] is None:
            return dict(
                tangential_force_on_a_world=(0.0, 0.0, 0.0),
                relative_velocity_world=None,
                mu=None,
                slip_speed_m_s=None,
                law=NORMAL_LAW,
            )
        return dict(
            tangential_force_on_a_world=tuple(map(float, value[15:18])),
            relative_velocity_world=tuple(map(float, value[18:21])),
            mu=float(value[21]),
            slip_speed_m_s=float(value[22]),
            law=LAW,
        )
