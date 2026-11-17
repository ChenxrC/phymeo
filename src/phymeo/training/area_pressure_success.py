"""Explicit role and history validation for accepted area-pressure observations.

This HOST evaluator consumes solved forces in newtons, not a new contact model.
Native contact and area sample identities remain separate.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..backends.area_pressure_readback import ABI, AreaPressureEntryReader, _identity_key
from ..geometry.area_pressure import AreaPressureIdentity, _identity


def _require(ok, message):
    if not ok:
        raise ValueError("manipulation area entry: " + message)


@dataclass(frozen=True)
class AreaPressureRoleBinding:
    identity: AreaPressureIdentity
    object_side: str | None
    role: str

    def __post_init__(self):
        object.__setattr__(self, "identity", _identity(self.identity))
        _require(
            self.object_side is None
            or (type(self.object_side) is str and self.object_side in ("A", "B")),
            "object_side must be A/B/None",
        )
        _require(
            type(self.role) is str and self.role in ("finger0", "finger1", "support", "other"),
            "unknown role",
        )
        _require(
            self.object_side is not None or self.role == "other", "absent object has no load role"
        )


class AreaSuccessAdapter:
    """Immutable declarations plus stateless complete-frame validation."""

    def __init__(self, workspace, roles, addresses):
        binding = workspace.binding
        _require(binding.get("area_pressure") is True, "area observation workspace is required")
        _require(
            tuple(
                binding.get(k) for k in ("area_schema", "area_force", "area_moment", "area_measure")
            )
            == ABI,
            "area observation ABI differs",
        )
        pairs = binding["area_pair_names"]
        expected = {key for key, value in pairs.items() if value}
        _require(
            isinstance(roles, Mapping) and set(roles) == expected,
            "roles must exactly cover area models",
        )
        roles = {key: roles[key] for key in expected}
        self.keys = tuple(binding["world_keys"])
        self.capacity = binding["area_capacity"]
        self.worlds = {}
        signature = []
        for world, key, _, _ in binding["source"]["worlds"]:
            meta = workspace.model_metadata[key]
            geom_bodies = tuple(int(v) for v in meta["geom_bodyid"])
            object_body, _, _, fingers, support = addresses[world]
            declared = tuple(pairs[key])
            configured = roles.get(key, ())
            _require(
                type(configured) is tuple and len(configured) == len(declared),
                "one ordered role per declared pair is required",
            )
            body_names = tuple(meta["body_names"])
            role_bodies = {f"finger{i}": geom_bodies[g] for i, g in enumerate(fingers)}
            role_bodies["support"] = geom_bodies[support]
            checked = []
            for pair, role in zip(declared, configured, strict=True):
                _require(type(role) is AreaPressureRoleBinding, "role binding type")
                role = AreaPressureRoleBinding(role.identity, role.object_side, role.role)
                identity = role.identity
                _require(
                    tuple(pair) == (identity.compliant_body, identity.rigid_body),
                    "role pair differs from model declaration",
                )
                body_a = body_names.index(identity.compliant_body)
                body_b = body_names.index(identity.rigid_body)
                actual_side = (
                    "A" if body_a == object_body else "B" if body_b == object_body else None
                )
                _require(
                    role.object_side == actual_side, "object side differs from actual Scene body"
                )
                other = body_b if actual_side == "A" else body_a
                named = (
                    [name for name, body in role_bodies.items() if body == other]
                    if actual_side
                    else []
                )
                _require(
                    not named or (len(named) == 1 and role.role == named[0]),
                    "named counterpart must use its unambiguous Scene role",
                )
                if role.role != "other":
                    other = body_b if actual_side == "A" else body_a
                    _require(
                        role.role in role_bodies and other == role_bodies[role.role],
                        "role counterpart differs from named Scene geometry owner",
                    )
                    _require(
                        list(role_bodies.values()).count(other) == 1,
                        "shared body makes the declared role ambiguous",
                    )
                checked.append((role, body_a, body_b))
            self.worlds[world] = (tuple(checked), geom_bodies)
            signature.append(
                (
                    world,
                    key,
                    tuple(
                        (_identity_key(r.identity), r.object_side, r.role, a, b)
                        for r, a, b in checked
                    ),
                )
            )
        self.binding = ("manipulation-area-success-v1", ABI, tuple(signature))

        self._entries = AreaPressureEntryReader(
            workspace,
            {key: tuple(role.identity for role in values) for key, values in roles.items()},
        )

    def validate(self, arrays, frame):
        """Decode once, then aggregate task roles without changing the digest."""
        entries = self._entries.validate(arrays, frame)
        result = {}
        for world in self.keys:
            configured, _ = self.worlds[world]
            loads, support_z = [0.0, 0.0], 0.0
            for ordinal, (role, _, _) in enumerate(configured):
                samples = tuple(s for s in entries[world].samples if s.pair_index == ordinal)
                if role.role.startswith("finger"):
                    loads[int(role.role[-1])] += float(
                        np.sum([s.normal_force_n for s in samples], dtype=np.float64)
                    )
                elif role.role == "support":
                    sign = 1 if role.object_side == "A" else -1
                    support_z += sign * float(
                        np.sum([s.force_on_a_world[2] for s in samples], dtype=np.float64)
                    )
            result[world] = (tuple(loads), support_z, entries[world].digest)
        return result
