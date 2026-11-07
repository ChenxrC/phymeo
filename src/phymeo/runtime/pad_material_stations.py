"""Declare bounded pad material stations from registered authored geometry.

This is a pure declaration adapter. It does not sample contact, choose a nearest
feature, enable friction, advance state, or infer a persistent token. A declared
counter-face is a caller assertion; the real loaded witness must still satisfy
MaterialStationRegistry.bind at every accepted physical attempt.
"""

from collections.abc import Mapping
from itertools import product
from numbers import Integral

import numpy as np

from ..builders._robot_asset import RobotAsset
from ..constraints.collision_shapes import CollisionKey
from ..core.robot import RobotModel
from ..dynamics.joint_mimic import MimicRobotAsset
from ..solvers.sphere_link_contact import LinkSolid
from ..solvers.tangent_history import StationKey
from .material_stations import MAX_MATERIAL_STATIONS, MaterialStationRegistration, MaterialStationRegistry
from .robot_payload_scene import PadFace, _pad_points


_BITS = tuple(product((0, 1), repeat=3))


def _integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _resolve(tree, pad):
    if not isinstance(pad, PadFace):
        raise TypeError("pads must contain original PadFace declarations")
    selector = pad.shape_key
    found = [col for col in tree.collisions if (
        col.shape.key == selector if isinstance(selector, CollisionKey)
        else (col.link_name, col.name) == (selector.link_name, selector.collision_name)
    )]
    if len(found) != 1 or not found[0].name_is_authored:
        raise ValueError("pad must select one unambiguous authored collision")
    col = found[0]
    if col.link_index < 0 or col.shape.kind != "hull" or col.shape.radius != 0:
        raise ValueError("material pad requires a moving, unrounded cuboid")
    return col


def _face_ids(shape, axis, side):
    bit = int(side > 0)
    return tuple(sorted(shape.vertex_ids[i] for i, bits in enumerate(_BITS) if bits[axis] == bit))


def _counter_feature(shape, vertices, declared):
    try:
        ids = tuple(_integer(value, "counter feature vertex") for value in declared)
    except TypeError as error:
        raise ValueError("counter feature must contain material vertex IDs") from error
    if not ids or len(set(ids)) != len(ids) or not set(ids).issubset(shape.vertex_ids):
        raise ValueError("counter feature contains duplicate or unknown material vertices")
    ids = tuple(sorted(ids))
    if shape.kind == "sphere":
        LinkSolid("payload", "payload", 0, "sphere", vertices, shape.radius)
        if ids != tuple(shape.vertex_ids) or len(ids) != 1 or np.any(np.asarray(vertices)[0]):
            raise ValueError("sphere counter feature must be its actual COM-centred material vertex")
    elif shape.kind == "hull":
        box = LinkSolid("payload", "payload", 0, "box", vertices, shape.radius)
        # The native sampled-pad payload operator uses a COM-centred axis-aligned
        # local box; its world pose can have any supported proper orientation.
        if not np.allclose(box._center, 0, atol=1e-12, rtol=0) or not np.allclose(
            box._axes, np.eye(3), atol=1e-12, rtol=0
        ):
            raise ValueError("pad payload requires the native COM-local cuboid")
        faces = {_face_ids(shape, axis, side) for axis in range(3) for side in (-1, 1)}
        if ids not in faces:
            raise ValueError("counter feature must be one complete cuboid face, not the whole box")
    else:
        raise ValueError("pad stations support sphere or cuboid payloads only")
    return ids


def material_stations_from_pads(
    scene, *, robot, payload_key, pads, station_keys, counter_features, chart_ids
):
    """Return an owned registry without reading or publishing any contact history.

    robot is (world, physical body). All three mappings are keyed by authored
    (link_name, collision_name). station_keys follows each declared offsets_m
    order; chart_ids contains the explicitly asserted (pad, payload) chart IDs.
    Reordered declarations are allowed, but must describe the actual registered
    pad face and its complete set of offsets. Stable tokens are never inferred.

    Scope: ordinary fixed-root and affine-mimic assets, moving cuboid pads, and
    an existing dynamic sphere/native cuboid payload in the same world. Runtime
    face migration, floating-root assets, and new material sampling are excluded.
    """
    from .multi_robot_scene import MultiRobotScene
    from .robot_scene import RobotScene

    if not isinstance(scene, (RobotScene, MultiRobotScene)):
        raise TypeError("pad stations require an existing public robot scene")
    if not isinstance(robot, tuple) or len(robot) != 2:
        raise ValueError("robot must be a (world, physical body) tuple")
    address_key = CollisionKey(robot[0], robot[1], 0)
    address = address_key.world, address_key.body
    if not isinstance(payload_key, CollisionKey) or payload_key.world != address[0]:
        raise ValueError("payload must be a registered same-world CollisionKey")
    supplied = []
    count = 0
    for pad in pads:
        if not isinstance(pad, PadFace):
            raise TypeError("pads must contain PadFace declarations")
        count += len(pad.offsets_m)
        if count > MAX_MATERIAL_STATIONS:
            raise ValueError("material station capacity exceeded")
        supplied.append(pad)
    if not supplied:
        raise ValueError("at least one registered pad must be declared")
    for mapping in (station_keys, counter_features, chart_ids):
        if not isinstance(mapping, Mapping):
            raise TypeError("station keys, counter features and charts must be mappings")

    with scene._lock:
        scene._require_open()
        if address not in scene.assets:
            raise ValueError("unknown registered robot owner")
        asset = scene.assets[address]
        tree = asset.tree if isinstance(asset, MimicRobotAsset) else asset
        if not isinstance(tree, RobotAsset) or not isinstance(tree.model, RobotModel):
            raise ValueError("pad declarations currently support fixed-root ordinary or mimic assets")
        spec = scene._spec if isinstance(scene, RobotScene) else scene._participants[address].spec
        registered = {}
        for pad in spec.pads:
            col = _resolve(tree, pad)
            name = col.link_name, col.name
            if name in registered:
                raise ValueError("registered pad collision is ambiguous")
            registered[name] = (pad.axis, pad.side, frozenset(pad.offsets_m))
        colliders = tuple(_resolve(tree, pad) for pad in supplied)
        names = tuple((col.link_name, col.name) for col in colliders)
        if len(set(names)) != len(names):
            raise ValueError("duplicate declared pad collision")
        if any(set(mapping) != set(names) for mapping in (station_keys, counter_features, chart_ids)):
            raise ValueError("material mappings must exactly cover the declared authored pad names")
        for pad, name in zip(supplied, names, strict=True):
            if registered.get(name) != (pad.axis, pad.side, frozenset(pad.offsets_m)):
                raise ValueError("declared pad differs from the scene's actual sampling geometry")
        owners = [addr for addr, keys in scene.body_keys.items() if payload_key in keys]
        if len(owners) != 1 or owners[0] not in scene.rigid_rows or owners[0] not in scene._items:
            raise ValueError("payload must belong to a registered free rigid owner")
        item = scene._items[owners[0]]
        if item.shape.key != payload_key or item.inverse_mass <= 0:
            raise ValueError("sampled pads require a dynamic rigid payload")
        shapes = {shape.key: shape for shape in scene._model.shapes}
        excluded = set(scene._model.excluded)
        indices, points, source_keys, _excluded = _pad_points(tree, tuple(supplied))
        rows, offset = [], 0
        for pad, col, name in zip(supplied, colliders, names, strict=True):
            key = scene.shape_key(address[1], *name)
            if key.world != address[0] or key not in scene.body_keys[address]:
                raise ValueError("pad collision belongs to a different registered owner")
            pair = tuple(sorted((key, payload_key)))
            if pair in excluded:
                raise ValueError("declared material pair is excluded from the scene")
            shape = shapes[key]
            if shape.vertex_ids != col.shape.vertex_ids:
                raise ValueError("pad material vertex IDs differ from its registered asset")
            box = LinkSolid(col.name, col.link_name, col.link_index, "box", col.local_vertices)
            axis = box._axes[next(i for i in range(3) if i != pad.axis)]
            convert = getattr(tree, "authored_to_model_points", None)
            if convert is not None:
                axis = convert(col.link_name, axis) - convert(col.link_name, np.zeros(3))
            tokens = tuple(station_keys[name])
            if len(tokens) != len(pad.offsets_m) or any(not isinstance(token, StationKey) for token in tokens):
                raise ValueError("each declared offset requires one explicit StationKey")
            charts = tuple(chart_ids[name])
            if len(charts) != 2:
                raise ValueError("each pad requires explicit pad and payload chart IDs")
            charts = tuple(_integer(value, "material chart ID") for value in charts)
            features = (_face_ids(shape, pad.axis, pad.side),
                        _counter_feature(item.shape, item.vertices, counter_features[name]))
            side = "first" if pair[0] == key else "second"
            if side == "second":
                charts, features = charts[::-1], features[::-1]
            for j, token in enumerate(tokens):
                i = offset + j
                if indices[i] != col.link_index or source_keys[i] != col.shape.key:
                    raise ValueError("pad material mapping disagrees with the actual sampler")
                rows.append(MaterialStationRegistration(
                    token, *pair, side, tuple(points[i]), tuple(axis), *charts,
                    first_feature=features[0], second_feature=features[1],
                ))
            offset += len(tokens)
        return MaterialStationRegistry(rows)
