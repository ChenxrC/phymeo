"""C10 independent finite-contact inputs and full floating response.

AUTHORING_UNEXECUTED. NumPy/stdlib only: no Phymeo, TYX or Newton imports.
The accepted C9 homogeneous time-jet reference supplies full inertia/J/bias and
one semi-implicit drift. Contact geometry below is independent finite-plane or
sphere-pair algebra; F07 deliberately supplies only a literal first geometry
anchor and treats the actual uploaded wrench as known input to the response.
"""

import copy
import importlib.util
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def base():
    path = Path(__file__).with_name("resident_floating_reference.py")
    spec = importlib.util.spec_from_file_location("c10_time_jet_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), float)


def _children(data, parents, kinds, offsets, axes, masses, inertia=None):
    n = len(parents)
    data["children"] = dict(
        parents=np.array(parents, int),
        kinds=tuple(kinds),
        offsets=np.asarray(offsets, float).reshape(n, 3),
        axes=np.asarray(axes, float).reshape(n, 3),
        orientations=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        mass=np.asarray(masses, float),
        com=np.zeros((n, 3)),
        inertia=np.full((n, 3), 0.25) if inertia is None else np.asarray(inertia, float),
    )
    data.update(
        q=np.r_[np.zeros(3), [0.0, 0.0, 0.0, 1.0], np.zeros(n)],
        v=np.zeros(n + 6),
        drive=np.zeros((n, 3)),
        effort=np.zeros(n),
        force=np.zeros((n + 1, 3)),
        torque=np.zeros((n + 1, 3)),
    )


def _shape(data, group, kind, points, radius=0.0):
    index = len(data["shapes"])
    data["shapes"].append(
        dict(
            group=group,
            kind=kind,
            points=np.asarray(points, float).reshape(-1, 3),
            radius=float(radius),
        )
    )
    return index


def _sphere(data, group, point=(0.0, 0.0, 0.0)):
    return _shape(data, group, "sphere", [point], 1 / 8)


def _wall(data, normal, point, *, half=(1.0, 1.0), depth=1 / 4):
    normal, point = np.asarray(normal, float), np.asarray(point, float)
    seed = np.eye(3)[int(np.argmin(abs(normal)))]
    u = np.cross(normal, seed)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    points = np.array(
        [
            point + a * u + b * v - d * normal
            for a, b, d in product((-half[0], half[0]), (-half[1], half[1]), (0.0, depth))
        ]
    )
    index = _shape(data, None, "hull", points)
    return dict(index=index, normal=normal, point=point, u=u, v=v, half=np.asarray(half, float))


def _against_wall(data, shape, wall, *, damping=0.0, kind="sphere_plane"):
    data["rules"].append(
        dict(
            kind=kind,
            first=shape,
            second=wall["index"],
            wall=wall,
            stiffness=64.0,
            damping=float(damping),
        )
    )


def case(name, *, n=None, variant=None, placement="base", boost=False):
    if name == "F08":
        data = base().case("P05", n=n)
    else:
        data = base().case("P03")
        _children(data, [], [], [], [], [])
    data.update(
        name=name,
        variant=variant,
        shapes=[],
        rules=[],
        reference_scope="independent_geometry_and_full_response",
        kp=np.zeros(len(data["v"]) - 6),
        kd=np.zeros(len(data["v"]) - 6),
        caps=np.full(len(data["v"]) - 6, 64.0),
        delay=0,
    )
    if name in ("F01", "F09"):
        data["q"][2] = 3 / 32 if name == "F01" else 1 / 8 - 1 / 4096
        if name == "F09":
            data["v"][2], data["gravity"][2] = 1 / 8, -8.0
        _against_wall(data, _sphere(data, 0), _wall(data, [0, 0, 1], [0, 0, 0]))
    elif name == "F02":
        _children(data, [-1], ["prismatic"], [[0, 0, 0]], [[1, 0, 0]], [1.0])
        group = 0 if variant == "root" else 1
        if variant == "motor":
            data["drive"][0, 2] = 3.0
        _against_wall(
            data, _sphere(data, group), _wall(data, [1, 0, 0], [-3 / 32, 0, 0], depth=13 / 32)
        )
    elif name in ("F03", "F07"):
        _children(
            data,
            [-1, -1],
            ["prismatic"] * 2,
            [[-7 / 64, 0, 0], [7 / 64, 0, 0]],
            [[1, 0, 0]] * 2,
            [2.0, 1.0],
        )
        if name == "F03":
            a, b = _sphere(data, 1), _sphere(data, 2)
            data["rules"].append(
                dict(kind="sphere_pair", first=a, second=b, stiffness=64.0, damping=0.0)
            )
        else:
            data["reference_scope"] = "literal_first_geometry_then_uploaded_wrench_response"
            data["children"]["offsets"][:] = [[0, 0, 0], [7 / 32, 0, 0]]
            kinds, radii = ["sphere"] * 2, [1 / 8] * 2
            points = [np.zeros((1, 3)), np.zeros((1, 3))]
            if variant == "hull":
                kinds, radii = ["hull"] * 2, [0.0] * 2
                points = [box([-1 / 8] * 3, [1 / 8] * 3)] * 2
            elif variant in ("cap_side", "cap_end", "cap_miss"):
                kinds[0] = "capsule"
                points[0] = np.array([[0, 0, -1 / 4], [0, 0, 1 / 4]], float)
            elif variant == "CC":
                kinds, radii = ["capsule"] * 2, [0.05] * 2
                points = [np.array([[-0.5, 0, 0], [0.5, 0, 0]])] * 2
            elif variant == "HH":
                kinds, radii = ["hull"] * 2, [0.0] * 2
                ring = [(-0.5, -0.1), (0.5, -0.1), (0.6, 0), (0.3, 0.1), (-0.5, 0.1)]
                points = [np.array([(x, y, z) for z in (-0.1, 0.1) for x, y in ring])] * 2
            else:
                raise ValueError(variant)
            if variant in ("cap_end", "cap_miss", "CC", "HH"):
                distance = dict(cap_end=1 / 4 + 7 / 32, cap_miss=1 / 4 + 9 / 32, CC=0.09, HH=0.19)[
                    variant
                ]
                data["children"]["axes"][:] = [0, 0, 1]
                data["children"]["offsets"][1] = [0, 0, distance]
            a, b = (_shape(data, i + 1, kinds[i], points[i], radii[i]) for i in range(2))
            data["rules"].append(
                dict(kind="literal", first=a, second=b, stiffness=64.0, damping=0.0)
            )
            data["first_gap"] = (
                1 / 32 if variant == "cap_miss" else -0.01 if variant in ("CC", "HH") else -1 / 32
            )
            data["first_normal"] = np.array(
                [0.0, 0.0, 1.0]
                if variant in ("cap_end", "cap_miss", "CC", "HH")
                else [1.0, 0.0, 0.0]
            )
    elif name == "F04":
        _children(data, [-1], ["revolute"], [[0, 0, 0]], [[0, 0, 1]], [1.0])
        data["v"][5], data["v"][6], data["drive"][0, 2] = 1 / 16, -1 / 32, 1.0
        _against_wall(
            data,
            _sphere(data, 1, [0, 1 / 4, 0]),
            _wall(data, [1, 0, 0], [-3 / 32, 0, 0], depth=13 / 32),
            damping=4.0,
        )
    elif name == "F05":
        if variant == "origin":
            data["root_com"] = np.array([1 / 8, 0.0, 0.0])
            data["root_inertia"] = np.array([1 / 4, 3 / 8, 1 / 2])
            _against_wall(data, _sphere(data, 0), _wall(data, [0, 1, 0], [0, -3 / 32, 0]))
        elif variant == "frame":
            data["root_inertia"] = np.array([1 / 4, 1 / 2, 1 / 2])
            data["root_inertia_frame"] = np.array([0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)])
            for z, normal in ((-1 / 4, 1), (1 / 4, -1)):
                _against_wall(
                    data,
                    _sphere(data, 0, [0, 0, z]),
                    _wall(data, [0, normal, 0], [0, -normal * 3 / 32, z]),
                )
        else:
            raise ValueError(variant)
    elif name == "F06":
        data["root_mass"] = 3.0
        data["root_com"] = np.array([1 / 12, 0.0, 0.0])
        data["root_inertia"] = np.array([3 / 10, 53 / 120, 13 / 24])
        tool = _shape(data, 0, "hull", box([3 / 16, -1 / 16, -1 / 16], [5 / 16, 1 / 16, 1 / 16]))
        wall = _wall(data, [0, 1, 0], [1 / 4, -1 / 64, 0], half=(1 / 8, 1 / 8), depth=15 / 64)
        _against_wall(data, tool, wall, kind="box_plane")
    elif name == "F08":
        group = n if variant != "body15" else 15
        observed = base().jets(data, data["q"], data["v"])
        center = observed["position"][group]
        _against_wall(data, _sphere(data, group), _wall(data, [1, 0, 0], center - [3 / 32, 0, 0]))
        if variant == "both":
            _against_wall(
                data,
                _sphere(data, 0),
                _wall(data, [0, 1, 0], observed["position"][0] - [0, 3 / 32, 0]),
            )
    elif name == "F08_branch":
        _children(
            data,
            [-1, 0, 0],
            ["revolute", "prismatic", "prismatic"],
            [[0, 0, 0], [0, 1 / 4, 0], [7 / 32, 1 / 4, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [1.0, 2.0, 1.0],
            [[0.25] * 3, [0.125] * 3, [0.125] * 3],
        )
        a, b = _sphere(data, 2), _sphere(data, 3)
        data["rules"].append(
            dict(kind="sphere_pair", first=a, second=b, stiffness=64.0, damping=0.0)
        )
    else:
        raise ValueError(name)
    count = len(data["v"]) - 6
    data.update(kp=np.zeros(count), kd=np.zeros(count), caps=np.full(count, 64.0))
    data["placement"], data["boost"] = placement, boost
    if boost:
        if name != "F03":
            raise ValueError("common boost holdout is the isolated two-dynamic-body pair")
        data["v"][:3] += [0.25, -0.125, 0.0625]
    if placement != "base":
        if placement != "cycle":
            raise ValueError(placement)
        quat = np.array([0.5, 0.5, 0.5, 0.5])
        Q, shift = base().quat_matrix(quat), np.array([0.5, -0.25, 0.75])
        data["q"][:3] = Q @ data["q"][:3] + shift
        data["q"][3:7] = base().quat_product(quat, data["q"][3:7])
        data["v"][:3], data["v"][3:6] = Q @ data["v"][:3], Q @ data["v"][3:6]
        data["gravity"] = Q @ data["gravity"]
        data["force"], data["torque"] = data["force"] @ Q.T, data["torque"] @ Q.T
        for shape in data["shapes"]:
            if shape["group"] is None:
                shape["points"] = shape["points"] @ Q.T + shift
        for rule in data["rules"]:
            if "wall" in rule:
                wall = rule["wall"]
                wall["point"] = Q @ wall["point"] + shift
                for key in ("normal", "u", "v"):
                    wall[key] = Q @ wall[key]
        if "first_normal" in data:
            data["first_normal"] = Q @ data["first_normal"]
    return copy.deepcopy(data)


def world_points(shape, terms):
    group = shape["group"]
    return (
        shape["points"].copy()
        if group is None
        else terms["position"][group] + shape["points"] @ terms["rotation"][group].T
    )


def _velocity(group, point, terms):
    if group is None:
        return np.zeros(3)
    return terms["com_velocity"][group] + np.cross(
        terms["angular_velocity"][group], point - terms["com_position"][group]
    )


def contact_wrenches(data, q, v, *, terms=None):
    """Independent finite domains, including actual support switches for F06."""
    terms = base().mechanics(data, q, v) if terms is None else terms
    wrench, rows = np.zeros((len(v) - 5, 6)), []
    for rule in data["rules"]:
        first, second = (data["shapes"][rule[side]] for side in ("first", "second"))
        a, b = world_points(first, terms), world_points(second, terms)
        g1, g2 = first["group"], second["group"]
        if rule["kind"] == "literal":
            raise ValueError("F07 needs its declared uploaded-wrench response audit")
        if rule["kind"] == "sphere_pair":
            delta = b[0] - a[0]
            distance = np.linalg.norm(delta)
            if distance == 0:
                raise ValueError("independent sphere pair has no unique normal")
            normal = -delta / distance  # force direction on first
            p1, p2 = a[0] - first["radius"] * normal, b[0] + second["radius"] * normal
            gap, sites = distance - first["radius"] - second["radius"], [(p1, p2, 1.0)]
        else:
            wall, normal = rule["wall"], rule["wall"]["normal"]
            if rule["kind"] == "sphere_plane":
                center = a[0]
                distance = (center - wall["point"]) @ normal
                gap = distance - first["radius"]
                sites = [(center - first["radius"] * normal, center - distance * normal, 1.0)]
            elif rule["kind"] == "box_plane":
                distances = (a - wall["point"]) @ normal
                gap = float(np.min(distances))
                # Exact support set of the independently transformed cube. Its
                # symmetry makes the centroid the uniform vertex average; no
                # product hull routine or patch weights enter this reference.
                ids = np.flatnonzero(abs(distances - gap) <= 1e-12)
                sites = [(a[i], a[i] - gap * normal, 1 / len(ids)) for i in ids]
            else:
                raise ValueError(rule["kind"])
            for _, foot, _ in sites:
                uv = np.array([(foot - wall["point"]) @ wall[key] for key in ("u", "v")])
                if np.any(abs(uv) > wall["half"] + 1e-12):
                    raise ValueError("reference leaves the declared finite wall interior")
        if gap > 0:
            continue
        for p1, p2, weight in sites:
            point = (p1 + p2) / 2
            v1, v2 = _velocity(g1, point, terms), _velocity(g2, point, terms)
            rate = (v1 - v2) @ normal
            magnitude = (
                max(0.0, -rule["stiffness"] * gap - rule["damping"] * rate) if gap < 0 else 0.0
            )
            force = magnitude * weight * normal
            for group, signed in ((g1, force), (g2, -force)):
                if group is not None:
                    wrench[group, :3] += signed
                    wrench[group, 3:] += np.cross(point - terms["com_position"][group], signed)
            rows.append(
                dict(
                    pair=(rule["first"], rule["second"]),
                    gap_m=float(gap),
                    weight=float(weight),
                    first_point=p1,
                    second_point=p2,
                    point=point,
                    force_on_first=force,
                    first_velocity=v1,
                    second_velocity=v2,
                )
            )
    return wrench, rows


def response(data, q, v, wrench, *, h, drive=None, external_packet=None):
    """Known full body-COM wrench -> all root/joint coordinates, one drift."""
    terms = base().mechanics(data, q, v)
    external_packet = base().external(data, q) if external_packet is None else external_packet
    input_external = base().applied(terms, *external_packet)
    generalized = base().applied(terms, np.zeros(len(v) - 6), wrench[:, :3], wrench[:, 3:])
    selected = data["drive"] if drive is None else np.asarray(drive, float)
    motor = base().bounded_drive(
        terms["mass"],
        terms["bias"],
        input_external + generalized,
        q,
        v,
        selected,
        data["kp"],
        data["kd"],
        data["caps"],
        h,
    )
    total = input_external + generalized
    total[6:] += motor
    acceleration = np.linalg.solve(terms["mass"], total - terms["bias"])
    next_v = np.asarray(v) + h * acceleration
    next_q = base().drift(data, q, next_v, h)
    return (
        next_q,
        next_v,
        dict(
            **terms,
            input_external=input_external,
            contact_generalized=generalized,
            drive=motor,
            acceleration=acceleration,
            wrench=np.array(wrench, copy=True),
        ),
    )


def advance(data, q, v, *, h, drive=None, external_packet=None):
    terms = base().mechanics(data, q, v)
    wrench, rows = contact_wrenches(data, q, v, terms=terms)
    next_q, next_v, result = response(
        data, q, v, wrench, h=h, drive=drive, external_packet=external_packet
    )
    return next_q, next_v, dict(**result, contact_rows=rows)
