"""Bind real load sites to explicitly registered rigid material stations.

This is a bounded adapter, not a contact matcher or a friction solver. A local
anchor declaration must identify exactly one current loaded site. The opposite
surface witness may change within an explicitly declared chart, but that drift
is never substituted for the motion of the entry material point.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import fsum, hypot, isclose
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..constraints.robot_contact import RobotContactMaterial
from ..core.validation import finite_real
from ..solvers.collision_loads import NormalLayer, PairLoad
from ..solvers.tangent_history import MaterialAddress, StationKey, StationSample, _sample
from .contact_events import (
    ContactComponent,
    ContactOwner,
    SceneContactEvents,
    SceneContactRecord,
)
from .contact_history import (
    ContactIdentity,
    ContactMaterialAddress,
    ContactMaterialSite,
    _owned_load,
    _weights,
)

MAX_MATERIAL_STATIONS = 128


def _index(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError("invalid " + name)
    return int(value)


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("invalid " + name)
    return tuple(map(float, array))


def _owner(value):
    if not isinstance(value, ContactOwner):
        raise TypeError("expected ContactOwner")
    world, body = _index(value.world, "owner world"), _index(value.body, "physical owner")
    kinds = {"rigid": "rigid_com", "robot_link": "robot_model_link", "robot_root": "robot_root"}
    if (
        not isinstance(value.kind, str)
        or value.kind not in kinds
        or not isinstance(value.local_frame, str)
        or value.local_frame
        not in (
            ("robot_root", "robot_model_link")
            if value.kind == "robot_root"
            else (kinds[value.kind],)
        )
    ):
        raise ValueError("station motion supports rigid/link/root owners only")
    if value.kind == "rigid":
        if value.link_index is not None or value.link_name is not None:
            raise ValueError("rigid owner cannot carry link fields")
        index, name = None, None
    else:
        if not isinstance(value.link_name, str) or not value.link_name:
            raise ValueError("robot owner needs a computational link name")
        index = _index(value.link_index, "robot link index", minimum=-1)
        root_index = -1 if value.local_frame == "robot_root" else 0
        if (value.kind == "robot_root" and index != root_index) or (
            value.kind == "robot_link" and index < 0
        ):
            raise ValueError("invalid robot link/root index")
        name = str(value.link_name)
    return ContactOwner(world, body, str(value.kind), str(value.local_frame), index, name)


@dataclass(frozen=True)
class OwnerPose:
    """An actual COM, link, fixed-root or floating-root frame with owned values."""

    owner: ContactOwner
    position_m: tuple
    rotation_world: tuple

    def __post_init__(self):
        rotation = np.asarray(self.rotation_world, dtype=float)
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
            or not isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-10, rel_tol=0)
        ):
            raise ValueError("owner rotation must be proper orthonormal")
        object.__setattr__(self, "owner", _owner(self.owner))
        object.__setattr__(self, "position_m", _vector(self.position_m, "owner position"))
        object.__setattr__(
            self, "rotation_world", tuple(tuple(map(float, row)) for row in rotation)
        )


@dataclass(frozen=True)
class MaterialStationRegistration:
    """A stable token and an authored local anchor, never an output row index.

    Chart IDs assert caller-validated same-chart continuity on each side; they
    do not prove a feature transition. B3 still resets on a changed owner,
    material support or topology revision. Optional first_feature/second_feature
    declare the complete chart vertex IDs; current weights must remain a subset.
    Without this declaration the current sparse support remains the feature.
    match_tolerance_m is an explicit
    local-coordinate roundoff allowance, not a nearest-site ranking radius.
    """

    key: StationKey
    first: CollisionKey
    second: CollisionKey
    anchor_side: str
    anchor_local_m: tuple
    tangent_axis_local: tuple
    first_chart_id: int | None
    second_chart_id: int | None
    topology_revision: int = 0
    match_tolerance_m: float = 1e-8
    first_feature: tuple[int, ...] | None = None
    second_feature: tuple[int, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.key, StationKey):
            raise TypeError("expected explicit StationKey")
        key = StationKey(
            _index(self.key.world, "station world"), _index(self.key.token, "station token")
        )
        if (
            not isinstance(self.first, CollisionKey)
            or not isinstance(self.second, CollisionKey)
            or not self.first < self.second
            or self.first.world != self.second.world
            or self.first.world != key.world
        ):
            raise ValueError("station requires canonical same-world shape keys")
        if not isinstance(self.anchor_side, str) or self.anchor_side not in {"first", "second"}:
            raise ValueError("anchor_side must be first or second")
        axis = _vector(self.tangent_axis_local, "material tangent axis")
        length = hypot(*axis)
        if not np.isfinite(length) or length <= 0:
            raise ValueError("material tangent axis must be nonzero")
        tolerance = finite_real(self.match_tolerance_m, "anchor match tolerance")
        if tolerance < 0:
            raise ValueError("anchor match tolerance must be nonnegative")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "anchor_side", str(self.anchor_side))
        object.__setattr__(self, "anchor_local_m", _vector(self.anchor_local_m, "local anchor"))
        object.__setattr__(self, "tangent_axis_local", tuple(x / length for x in axis))
        object.__setattr__(self, "match_tolerance_m", tolerance)
        object.__setattr__(
            self, "topology_revision", _index(self.topology_revision, "topology revision")
        )
        for name in ("first_chart_id", "second_chart_id"):
            value = getattr(self, name)
            object.__setattr__(
                self, name, None if value is None else _index(value, "material chart")
            )
        for name in ("first_feature", "second_feature"):
            value = getattr(self, name)
            if value is not None:
                value = tuple(sorted(_index(node, "declared feature vertex") for node in value))
                if not value or len(set(value)) != len(value):
                    raise ValueError("declared feature needs distinct material vertex IDs")
            object.__setattr__(self, name, value)


def _material(address, owner):
    if not isinstance(address, ContactMaterialAddress) or address.kind != owner.kind:
        raise ValueError("material address differs from its physical owner")
    weights = _weights(address.weights)
    if any(w < 0 for _, w in weights) or not isclose(
        fsum(w for _, w in weights), 1.0, abs_tol=1e-10, rel_tol=0
    ):
        raise ValueError("material weights must be a nonnegative partition")
    return ContactMaterialAddress(
        owner.kind, weights, _vector(address.local_point_m, "local material point")
    )


def _record(value):
    if not isinstance(value, SceneContactRecord) or not isinstance(value.identity, ContactIdentity):
        raise TypeError("expected a real SceneContactRecord")
    identity = value.identity
    if (
        not isinstance(identity.first, CollisionKey)
        or not isinstance(identity.second, CollisionKey)
        or not identity.first < identity.second
        or identity.first.world != identity.second.world
    ):
        raise ValueError("contact record requires canonical same-world shapes")
    first, second = _owner(value.first_owner), _owner(value.second_owner)
    for owner, key in ((first, identity.first), (second, identity.second)):
        if owner.world != key.world or (owner.kind == "rigid" and owner.body != key.body):
            raise ValueError("contact material owner differs from its registered shape")
    components = []
    for component in value.components:
        if not isinstance(component, ContactComponent) or not isinstance(component.load, PairLoad):
            raise TypeError("expected unchanged contact component and PairLoad")
        law_type = {"normal_layer": NormalLayer, "robot_contact_material": RobotContactMaterial}
        if not isinstance(component.material_kind, str) or component.material_kind not in law_type:
            raise ValueError("unknown original contact law")
        if not isinstance(component.law, law_type[component.material_kind]):
            raise ValueError("contact material kind differs from its original law")
        load = _owned_load(component.load)
        if (load.first, load.second) != (identity.first, identity.second):
            raise ValueError("component belongs to a different shape pair")
        if load.force_n < 0 or len(component.material) != len(load.sites):
            raise ValueError("contact material/site counts or normal loads disagree")
        material = []
        for address, site in zip(component.material, load.sites, strict=True):
            if (
                not isinstance(address, ContactMaterialSite)
                or site.force_n < 0
                or site.fraction <= 0
            ):
                raise ValueError("invalid material site or negative normal load")
            a, b = _material(address.first, first), _material(address.second, second)
            fraction = finite_real(address.fraction, "material fraction")
            if (
                fraction != site.fraction
                or a.weights != site.material.first_weights
                or b.weights != site.material.second_weights
            ):
                raise ValueError("material addresses differ from the original load site")
            material.append(ContactMaterialSite(a, b, fraction))
        if not isclose(
            fsum(s.force_n for s in load.sites), load.force_n, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError("site normal loads differ from their pair total")
        components.append(
            ContactComponent(
                load, tuple(material), str(component.material_kind), replace(component.law)
            )
        )
    for flag in (value.material_changed, value.support_changed):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError("invalid material change flag")
    return SceneContactRecord(
        ContactIdentity(identity.first, identity.second, _index(identity.generation, "episode", 1)),
        first,
        second,
        tuple(components),
        bool(value.material_changed),
        bool(value.support_changed),
    )


def _pose(frames, owner):
    if not isinstance(frames, Mapping) or owner not in frames:
        raise ValueError("missing actual owner pose")
    pose = frames[owner]
    if not isinstance(pose, OwnerPose) or pose.owner != owner:
        raise ValueError("owner pose is bound to a different owner")
    return OwnerPose(pose.owner, pose.position_m, pose.rotation_world)


@dataclass(frozen=True)
class MaterialStationBinding:
    """Exact current load provenance; indices never identify a persistent station."""

    key: StationKey
    record: SceneContactRecord
    component_index: int
    site_index: int

    @property
    def component(self):
        return self.record.components[self.component_index]

    @property
    def site(self):
        return self.component.load.sites[self.site_index]

    @property
    def first_owner(self):
        return self.record.first_owner

    @property
    def second_owner(self):
        return self.record.second_owner

    @property
    def first_material_local_m(self):
        return self.component.material[self.site_index].first.local_point_m

    @property
    def second_material_local_m(self):
        return self.component.material[self.site_index].second.local_point_m

    @property
    def application_point_first_m(self):
        return self.site.application_point_first_m

    @property
    def application_point_second_m(self):
        return self.site.application_point_second_m


@dataclass(frozen=True)
class MaterialStationSelection:
    """One owned input-time selection with no mutable episode clock or solver state."""

    sample_time_s: float
    samples: tuple[StationSample, ...]
    bindings: tuple[MaterialStationBinding, ...]
    _entry_frames: Mapping
    _application_local: tuple

    def measure_motion(self, end_frames):
        """Return actual second-minus-first material and application displacements.

        Both use the entry local points advected by actual end poses. The caller
        supplies poses from its physical candidate and performs motion/work gates;
        this method neither reconstructs the end closest point nor accepts a step.
        """
        end = {owner: _pose(end_frames, owner) for owner in self._entry_frames}

        def displacement(owner, local):
            start, stop = self._entry_frames[owner], end[owner]
            return (
                np.asarray(stop.position_m)
                - start.position_m
                + (np.asarray(stop.rotation_world) - start.rotation_world) @ local
            )

        material, application = [], []
        for binding, (a, b) in zip(self.bindings, self._application_local, strict=True):
            material.append(
                displacement(binding.second_owner, binding.second_material_local_m)
                - displacement(binding.first_owner, binding.first_material_local_m)
            )
            application.append(
                displacement(binding.second_owner, b) - displacement(binding.first_owner, a)
            )
        output = []
        for values in (material, application):
            array = np.asarray(values, dtype=np.float64).reshape(len(self.samples), 3)
            if not np.isfinite(array).all():
                raise ValueError("actual material motion is not finite")
            output.append(np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape))
        return tuple(output)


@dataclass(frozen=True, init=False)
class MaterialStationRegistry:
    """At most 128 explicit declarations, matching the coupled tangent core cap.

    Unconfigured pairs are outside this opt-in adapter. Every positive-normal
    site on a configured pair must have exactly one matching declaration, and
    a declaration cannot consume two samples. Empty/inactive pairs are valid.
    """

    registrations: tuple[MaterialStationRegistration, ...]
    pairs: tuple[tuple[CollisionKey, CollisionKey], ...]
    _by_pair: Mapping

    def __init__(self, registrations):
        owned, keys, pairs = [], set(), {}
        for index, raw in enumerate(registrations):
            if index >= MAX_MATERIAL_STATIONS:
                raise ValueError("material station capacity exceeded")
            if not isinstance(raw, MaterialStationRegistration):
                raise TypeError("expected MaterialStationRegistration")
            registration = replace(raw)
            if registration.key in keys:
                raise ValueError("duplicate declared station key")
            keys.add(registration.key)
            owned.append(registration)
            pairs.setdefault((registration.first, registration.second), []).append(registration)
        object.__setattr__(self, "registrations", tuple(sorted(owned, key=lambda r: r.key)))
        object.__setattr__(self, "pairs", tuple(sorted(pairs)))
        object.__setattr__(
            self, "_by_pair", MappingProxyType({p: tuple(rows) for p, rows in pairs.items()})
        )

    def bind(self, entry_events, entry_frames, *, sample_time_s):
        """Bind actual input-time normal loads; do not advance B3 or a scene.

        Public scene contact_events describe the last accepted step's input.
        A new physical step must stage a view of its current normal sample;
        passing the prior accepted view with the new time is rejected here.
        """
        if not isinstance(entry_events, SceneContactEvents):
            raise TypeError("expected SceneContactEvents")
        time = finite_real(sample_time_s, "station sample time")
        if time < 0 or time != finite_real(entry_events.sample_time_s, "event sample time"):
            raise ValueError("station and event sample times disagree")
        if not isinstance(entry_frames, Mapping):
            raise TypeError("entry_frames must map actual ContactOwners to OwnerPoses")
        selected, frames, seen = {}, {}, set()
        for raw in entry_events.active:
            if not isinstance(raw, SceneContactRecord):
                raise TypeError("expected SceneContactRecord")
            pair = raw.identity.first, raw.identity.second
            if pair not in self._by_pair:
                continue
            if pair in seen:
                raise ValueError("duplicate selected pair record")
            seen.add(pair)
            record = _record(raw)
            declarations = self._by_pair[pair]
            for ci, component in enumerate(record.components):
                for si, site in enumerate(component.load.sites):
                    if site.force_n == 0:
                        continue
                    address = component.material[si]
                    matches = [
                        r
                        for r in declarations
                        if hypot(
                            *(
                                np.asarray(getattr(address, r.anchor_side).local_point_m)
                                - r.anchor_local_m
                            )
                        )
                        <= r.match_tolerance_m
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            "positive load site has missing or ambiguous material registration"
                        )
                    registration = matches[0]
                    if registration.key in selected:
                        raise ValueError("one declared station matches duplicate load sites")
                    binding = MaterialStationBinding(registration.key, record, ci, si)
                    local_application = []
                    for side in ("first", "second"):
                        owner = getattr(record, side + "_owner")
                        if owner not in frames:
                            frames[owner] = _pose(entry_frames, owner)
                        pose = frames[owner]
                        rotation = np.asarray(pose.rotation_world)
                        local = getattr(address, side).local_point_m
                        surface = getattr(site.material, side + "_point")
                        if (
                            hypot(*(np.asarray(pose.position_m) + rotation @ local - surface))
                            > registration.match_tolerance_m
                        ):
                            raise ValueError(
                                "entry owner pose does not reconstruct the sampled material point"
                            )
                        point = getattr(site, "application_point_" + side + "_m")
                        local_application.append(
                            tuple((np.asarray(point) - pose.position_m) @ rotation)
                        )
                    anchor_owner = getattr(record, registration.anchor_side + "_owner")
                    axis = (
                        np.asarray(frames[anchor_owner].rotation_world)
                        @ registration.tangent_axis_local
                    )
                    normal = np.asarray(component.load.normal)
                    tangent = axis - (axis @ normal) * normal
                    length = hypot(*tangent)
                    if not np.isfinite(length) or length <= 1e-12:
                        raise ValueError("material tangent axis is parallel to the sampled normal")
                    tangent /= length
                    basis = np.column_stack((tangent, np.cross(normal, tangent), normal))
                    materials = []
                    for side in ("first", "second"):
                        part = getattr(address, side)
                        support = tuple(node for node, _ in part.weights)
                        declared = getattr(registration, side + "_feature")
                        feature = support if declared is None else declared
                        if not set(support).issubset(feature):
                            raise ValueError("sample weights leave declared " + side + "_feature")
                        materials.append(
                            MaterialAddress(
                                getattr(record, side + "_owner"),
                                getattr(record.identity, side),
                                registration.topology_revision,
                                feature,
                                part.weights,
                                part.local_point_m,
                                getattr(registration, side + "_chart_id"),
                            )
                        )
                    sample = _sample(
                        StationSample(
                            registration.key, *materials, tuple(normal), basis, site.force_n
                        )
                    )
                    selected[registration.key] = sample, binding, tuple(local_application)
        ordered = [selected[key] for key in sorted(selected)]
        return MaterialStationSelection(
            time,
            tuple(row[0] for row in ordered),
            tuple(row[1] for row in ordered),
            MappingProxyType(frames),
            tuple(row[2] for row in ordered),
        )
