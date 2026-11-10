"""Independent literals for the first K1 patches.

NumPy and the standard library only. Expected points come from the declared
finite spheres, capsule axes, box faces, and prism faces. This module does not
import Phymeo, TYX, or Newton.
"""

from itertools import product

import numpy as np

U32 = 2.0**-23


def ss_overlap():
    normal = np.array([0.0, 0.0, 1.0])
    return {
        "name": "SS-overlap",
        "family": "SS",
        "gap_m": -0.01,
        "normal": normal,
        "dimension": 0,
        "length_m": 1.0,
        "sites": ({
            "weight": 1.0,
            "surfaces": (np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.04])),
            "ids": ((100,), (200,)),
            "bary": ((1.0,), (1.0,)),
        },),
    }


def cc_parallel():
    normal = np.array([0.0, 0.0, 1.0])
    ends = (-0.25, 0.5)
    parameters = (0.25, 1.0)
    b_parameters = (0.0, 0.75)
    sites = []
    for parameter, b_parameter, end in zip(parameters, b_parameters, ends, strict=True):
        sites.append({
            "weight": 0.5,
            "surfaces": (np.array([end, 0.0, 0.05]), np.array([end, 0.0, 0.04])),
            "ids": ((100, 101), (200, 201)),
            "bary": ((1.0 - parameter, parameter), (1.0 - b_parameter, b_parameter)),
        })
    return {
        "name": "cc_parallel",
        "family": "CC",
        "gap_m": -0.01,
        "normal": normal,
        "dimension": 1,
        "length_m": 1.1,
        "sites": tuple(sites),
        "case": "cc_parallel",
    }


def bb_overlap():
    normal = np.array([0.0, 0.0, 1.0])
    corners = ((-0.6, -0.1), (0.6, -0.1), (0.6, 0.1), (-0.6, 0.1))
    first_ids = {(-0.6, -0.1): 101, (-0.6, 0.1): 103, (0.6, -0.1): 105, (0.6, 0.1): 107}
    second_ids = {(-0.6, -0.1): 200, (-0.6, 0.1): 202, (0.6, -0.1): 204, (0.6, 0.1): 206}
    sites = []
    for x, y in corners:
        sites.append({
            "weight": 0.25,
            "surfaces": (np.array([x, y, 0.1]), np.array([x, y, 0.09])),
            "ids": ((first_ids[(x, y)],), (second_ids[(x, y)],)),
            "bary": ((1.0,), (1.0,)),
        })
    return {
        "name": "BB-overlap",
        "family": "BB",
        "gap_m": -0.01,
        "normal": normal,
        "dimension": 2,
        "length_m": float(np.sqrt(1.52)),
        "sites": tuple(sites),
    }


def _length(*groups):
    span = 1.0
    for points, radius in groups:
        points = np.asarray(points, dtype=float)
        wide = max(float(np.linalg.norm(a - b)) for a in points for b in points) + 2.0 * radius
        span = max(span, wide)
    return span


def sc_overlap():
    normal = np.array([0.0, 0.0, 1.0])
    capsule = np.array([[-0.5, 0.0, 0.09], [0.5, 0.0, 0.09]])
    return {
        "name": "SC-overlap",
        "family": "SC",
        "gap_m": -0.01,
        "normal": normal,
        "dimension": 0,
        "length_m": _length((np.zeros((1, 3)), 0.05), (capsule, 0.05)),
        "sites": ({
            "weight": 1.0,
            "surfaces": (np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.04])),
            "ids": ((100,), (200, 201)),
            "bary": ((1.0,), (0.5, 0.5)),
        },),
    }


def _support_point(vertices, ids):
    cores = {int(node): np.asarray(point, dtype=float) for node, point in zip(ids, vertices, strict=True)}
    return {
        "weight": 1.0,
        "surfaces": (np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.04])),
        "ids": ((100,), None),
        "bary": ((1.0,), None),
        "allowed": (None, tuple(cores)),
        "cores": (None, cores),
        "radii": (0.05, 0.0),
    }


def sb_overlap():
    normal = np.array([0.0, 0.0, 1.0])
    box = np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1))), dtype=float)
    box[:, 2] += 0.14
    face = np.isclose(box[:, 2], 0.04)
    ids = np.arange(200, 208)[face]
    return {
        "name": "SB-overlap",
        "family": "SB",
        "gap_m": -0.01,
        "normal": normal,
        "dimension": 0,
        "length_m": _length((np.zeros((1, 3)), 0.05), (box, 0.0)),
        "sites": (_support_point(box[face], ids),),
    }


def sh_overlap():
    normal = np.array([0.0, 0.0, 1.0])
    ring = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0.0), (0.3, 0.1), (-0.5, 0.1))
    prism = np.array([(x, y, z + 0.14) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    face = np.isclose(prism[:, 2], 0.04)
    ids = np.arange(200, 210)[face]
    return {
        "name": "SH-overlap",
        "family": "SH",
        "gap_m": -0.01,
        "normal": normal,
        "dimension": 0,
        "length_m": _length((np.zeros((1, 3)), 0.05), (prism, 0.0)),
        "sites": (_support_point(prism[face], ids),),
    }


def ss_sep():
    return {
        "name": "SS-sep",
        "family": "SS",
        "gap_m": 0.02,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 0,
        "length_m": 1.0,
        "sites": ({
            "weight": 1.0,
            "surfaces": (np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.07])),
            "ids": ((100,), (200,)),
            "bary": ((1.0,), (1.0,)),
        },),
    }


def sc_sep():
    capsule = np.array([[-0.5, 0.0, 0.12], [0.5, 0.0, 0.12]])
    return {
        "name": "SC-sep",
        "family": "SC",
        "gap_m": 0.02,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 0,
        "length_m": _length((np.zeros((1, 3)), 0.05), (capsule, 0.05)),
        "sites": ({
            "weight": 1.0,
            "surfaces": (np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.07])),
            "ids": ((100,), (200, 201)),
            "bary": ((1.0,), (0.5, 0.5)),
        },),
    }


def _separated_face(vertices, ids):
    cores = {int(node): np.asarray(point, dtype=float) for node, point in zip(ids, vertices, strict=True)}
    return {
        "weight": 1.0,
        "surfaces": (np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.07])),
        "ids": ((100,), None),
        "bary": ((1.0,), None),
        "allowed": (None, tuple(cores)),
        "cores": (None, cores),
        "radii": (0.05, 0.0),
    }


def sb_sep():
    box = np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1))), dtype=float)
    box[:, 2] += 0.17
    face = np.isclose(box[:, 2], 0.07)
    ids = np.arange(200, 208)[face]
    return {
        "name": "SB-sep",
        "family": "SB",
        "gap_m": 0.02,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 0,
        "length_m": _length((np.zeros((1, 3)), 0.05), (box, 0.0)),
        "sites": (_separated_face(box[face], ids),),
    }


def sh_sep():
    ring = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0.0), (0.3, 0.1), (-0.5, 0.1))
    prism = np.array([(x, y, z + 0.17) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    face = np.isclose(prism[:, 2], 0.07)
    ids = np.arange(200, 210)[face]
    return {
        "name": "SH-sep",
        "family": "SH",
        "gap_m": 0.02,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 0,
        "length_m": _length((np.zeros((1, 3)), 0.05), (prism, 0.0)),
        "sites": (_separated_face(prism[face], ids),),
    }


def _touch_face(vertices, ids):
    cores = {int(node): np.asarray(point, dtype=float) for node, point in zip(ids, vertices, strict=True)}
    return {
        "weight": 1.0,
        "surfaces": (np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.05])),
        "ids": ((100,), None),
        "bary": ((1.0,), None),
        "allowed": (None, tuple(cores)),
        "cores": (None, cores),
        "radii": (0.05, 0.0),
    }


def ss_touch():
    return {
        "name": "SS-touch",
        "family": "SS",
        "gap_m": 0.0,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 0,
        "length_m": 1.0,
        "sites": ({
            "weight": 1.0,
            "surfaces": (np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.05])),
            "ids": ((100,), (200,)),
            "bary": ((1.0,), (1.0,)),
        },),
    }


def sc_touch():
    capsule = np.array([[-0.5, 0.0, 0.10], [0.5, 0.0, 0.10]])
    return {
        "name": "SC-touch",
        "family": "SC",
        "gap_m": 0.0,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 0,
        "length_m": _length((np.zeros((1, 3)), 0.05), (capsule, 0.05)),
        "sites": ({
            "weight": 1.0,
            "surfaces": (np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.05])),
            "ids": ((100,), (200, 201)),
            "bary": ((1.0,), (0.5, 0.5)),
        },),
    }


def sb_touch():
    box = np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1))), dtype=float)
    box[:, 2] += 0.15
    face = np.isclose(box[:, 2], 0.05)
    ids = np.arange(200, 208)[face]
    return {
        "name": "SB-touch",
        "family": "SB",
        "gap_m": 0.0,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 0,
        "length_m": _length((np.zeros((1, 3)), 0.05), (box, 0.0)),
        "sites": (_touch_face(box[face], ids),),
    }


def sh_touch():
    ring = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0.0), (0.3, 0.1), (-0.5, 0.1))
    prism = np.array([(x, y, z + 0.15) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    face = np.isclose(prism[:, 2], 0.05)
    ids = np.arange(200, 210)[face]
    return {
        "name": "SH-touch",
        "family": "SH",
        "gap_m": 0.0,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 0,
        "length_m": _length((np.zeros((1, 3)), 0.05), (prism, 0.0)),
        "sites": (_touch_face(prism[face], ids),),
    }


def cc_overlap():
    sites = []
    for parameter, end in ((0.0, -0.5), (1.0, 0.5)):
        sites.append({
            "weight": 0.5,
            "surfaces": (np.array([end, 0.0, 0.05]), np.array([end, 0.0, 0.04])),
            "ids": ((100, 101), (200, 201)),
            "bary": ((1.0 - parameter, parameter), (1.0 - parameter, parameter)),
        })
    capsule = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    return {
        "name": "CC-overlap",
        "family": "CC",
        "gap_m": -0.01,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 1,
        "length_m": _length((capsule, 0.05), (capsule + (0.0, 0.0, 0.09), 0.05)),
        "sites": tuple(sites),
    }


def _capsule_on_face(name, family, shape, face):
    cores = {int(node): np.asarray(point, dtype=float) for node, point in zip(face[1], face[0], strict=True)}
    sites = []
    for parameter, end in ((0.0, -0.5), (1.0, 0.5)):
        sites.append({
            "weight": 0.5,
            "surfaces": (np.array([end, 0.0, 0.05]), np.array([end, 0.0, 0.04])),
            "ids": ((100, 101), None),
            "bary": ((1.0 - parameter, parameter), None),
            "allowed": (None, tuple(cores)),
            "cores": (None, cores),
            "radii": (0.05, 0.0),
        })
    capsule = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    return {
        "name": name,
        "family": family,
        "gap_m": -0.01,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 1,
        "length_m": _length((capsule, 0.05), (shape, 0.0)),
        "sites": tuple(sites),
    }


def cb_overlap():
    box = np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1))), dtype=float)
    box[:, 2] += 0.14
    face = np.isclose(box[:, 2], 0.04)
    return _capsule_on_face("CB-overlap", "CB", box, (box[face], np.arange(200, 208)[face]))


def ch_overlap():
    ring = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0.0), (0.3, 0.1), (-0.5, 0.1))
    prism = np.array([(x, y, z + 0.14) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    face = np.isclose(prism[:, 2], 0.04)
    return _capsule_on_face("CH-overlap", "CH", prism, (prism[face], np.arange(200, 210)[face]))


def cc_sep():
    sites = []
    for parameter, end in ((0.0, -0.5), (1.0, 0.5)):
        sites.append({
            "weight": 0.5,
            "surfaces": (np.array([end, 0.0, 0.05]), np.array([end, 0.0, 0.07])),
            "ids": ((100, 101), (200, 201)),
            "bary": ((1.0 - parameter, parameter), (1.0 - parameter, parameter)),
        })
    capsule = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    return {
        "name": "CC-sep",
        "family": "CC",
        "gap_m": 0.02,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 1,
        "length_m": _length((capsule, 0.05), (capsule + (0.0, 0.0, 0.12), 0.05)),
        "sites": tuple(sites),
    }


def _separated_capsule_on_face(name, family, shape, face):
    cores = {int(node): np.asarray(point, dtype=float) for node, point in zip(face[1], face[0], strict=True)}
    sites = []
    for parameter, end in ((0.0, -0.5), (1.0, 0.5)):
        sites.append({
            "weight": 0.5,
            "surfaces": (np.array([end, 0.0, 0.05]), np.array([end, 0.0, 0.07])),
            "ids": ((100, 101), None),
            "bary": ((1.0 - parameter, parameter), None),
            "allowed": (None, tuple(cores)),
            "cores": (None, cores),
            "radii": (0.05, 0.0),
        })
    capsule = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    return {
        "name": name,
        "family": family,
        "gap_m": 0.02,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 1,
        "length_m": _length((capsule, 0.05), (shape, 0.0)),
        "sites": tuple(sites),
    }


def cb_sep():
    box = np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1))), dtype=float)
    box[:, 2] += 0.17
    face = np.isclose(box[:, 2], 0.07)
    return _separated_capsule_on_face("CB-sep", "CB", box, (box[face], np.arange(200, 208)[face]))


def ch_sep():
    ring = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0.0), (0.3, 0.1), (-0.5, 0.1))
    prism = np.array([(x, y, z + 0.17) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    face = np.isclose(prism[:, 2], 0.07)
    return _separated_capsule_on_face("CH-sep", "CH", prism, (prism[face], np.arange(200, 210)[face]))


def cc_touch():
    sites = []
    for parameter, end in ((0.0, -0.5), (1.0, 0.5)):
        sites.append({
            "weight": 0.5,
            "surfaces": (np.array([end, 0.0, 0.05]), np.array([end, 0.0, 0.05])),
            "ids": ((100, 101), (200, 201)),
            "bary": ((1.0 - parameter, parameter), (1.0 - parameter, parameter)),
        })
    capsule = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    return {
        "name": "CC-touch",
        "family": "CC",
        "gap_m": 0.0,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 1,
        "length_m": _length((capsule, 0.05), (capsule + (0.0, 0.0, 0.10), 0.05)),
        "sites": tuple(sites),
    }


def _touch_capsule_on_face(name, family, shape, face):
    cores = {int(node): np.asarray(point, dtype=float) for node, point in zip(face[1], face[0], strict=True)}
    sites = []
    for parameter, end in ((0.0, -0.5), (1.0, 0.5)):
        sites.append({
            "weight": 0.5,
            "surfaces": (np.array([end, 0.0, 0.05]), np.array([end, 0.0, 0.05])),
            "ids": ((100, 101), None),
            "bary": ((1.0 - parameter, parameter), None),
            "allowed": (None, tuple(cores)),
            "cores": (None, cores),
            "radii": (0.05, 0.0),
        })
    capsule = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    return {
        "name": name,
        "family": family,
        "gap_m": 0.0,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 1,
        "length_m": _length((capsule, 0.05), (shape, 0.0)),
        "sites": tuple(sites),
    }


def cb_touch():
    box = np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1))), dtype=float)
    box[:, 2] += 0.15
    face = np.isclose(box[:, 2], 0.05)
    return _touch_capsule_on_face("CB-touch", "CB", box, (box[face], np.arange(200, 208)[face]))


def ch_touch():
    ring = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0.0), (0.3, 0.1), (-0.5, 0.1))
    prism = np.array([(x, y, z + 0.15) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    face = np.isclose(prism[:, 2], 0.05)
    return _touch_capsule_on_face("CH-touch", "CH", prism, (prism[face], np.arange(200, 210)[face]))


def _fan_weights(points):
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    areas = np.array([
        np.linalg.norm(np.cross(points[i] - center, points[(i + 1) % len(points)] - center))
        for i in range(len(points))
    ])
    return (areas + np.roll(areas, 1)) / (3.0 * areas.sum()) + 1.0 / (3.0 * len(points))


def _pentagon():
    return np.array(((-0.5, -0.1), (0.5, -0.1), (0.6, 0.0), (0.3, 0.1), (-0.5, 0.1)), dtype=float)


def bh_overlap():
    ring = _pentagon()
    weights = _fan_weights(np.column_stack((ring, np.zeros(len(ring)))))
    box = np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1))), dtype=float)
    top = np.isclose(box[:, 2], 0.1)
    cores = {int(node): point.copy() for node, point in zip(np.arange(100, 108)[top], box[top], strict=True)}
    prism = np.array([(x, y, z + 0.19) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    sites = []
    for index, (xy, weight) in enumerate(zip(ring, weights, strict=True)):
        sites.append({
            "weight": float(weight),
            "surfaces": (np.array([xy[0], xy[1], 0.1]), np.array([xy[0], xy[1], 0.09])),
            "ids": (None, (200 + 2 * index,)),
            "bary": (None, (1.0,)),
            "allowed": (tuple(cores), None),
            "cores": (cores, None),
            "radii": (0.0, 0.0),
        })
    return {
        "name": "BH-overlap",
        "family": "BH",
        "gap_m": -0.01,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 2,
        "length_m": _length((box, 0.0), (prism, 0.0)),
        "sites": tuple(sites),
    }


def hh_overlap():
    ring = _pentagon()
    weights = _fan_weights(np.column_stack((ring, np.full(len(ring), 0.1))))
    prism = np.array([(x, y, z) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    shifted = prism + (0.0, 0.0, 0.19)
    sites = []
    for index, (xy, weight) in enumerate(zip(ring, weights, strict=True)):
        sites.append({
            "weight": float(weight),
            "surfaces": (np.array([xy[0], xy[1], 0.1]), np.array([xy[0], xy[1], 0.09])),
            "ids": ((101 + 2 * index,), (200 + 2 * index,)),
            "bary": ((1.0,), (1.0,)),
        })
    return {
        "name": "HH-overlap",
        "family": "HH",
        "gap_m": -0.01,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 2,
        "length_m": _length((prism, 0.0), (shifted, 0.0)),
        "sites": tuple(sites),
    }


def bb_sep():
    normal = np.array([0.0, 0.0, 1.0])
    corners = ((-0.6, -0.1), (0.6, -0.1), (0.6, 0.1), (-0.6, 0.1))
    first_ids = {(-0.6, -0.1): 101, (-0.6, 0.1): 103, (0.6, -0.1): 105, (0.6, 0.1): 107}
    second_ids = {(-0.6, -0.1): 200, (-0.6, 0.1): 202, (0.6, -0.1): 204, (0.6, 0.1): 206}
    box = np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1))), dtype=float)
    sites = []
    for x, y in corners:
        sites.append({
            "weight": 0.25,
            "surfaces": (np.array([x, y, 0.1]), np.array([x, y, 0.12])),
            "ids": ((first_ids[(x, y)],), (second_ids[(x, y)],)),
            "bary": ((1.0,), (1.0,)),
        })
    return {
        "name": "BB-sep",
        "family": "BB",
        "gap_m": 0.02,
        "normal": normal,
        "dimension": 2,
        "length_m": _length((box, 0.0), (box + (0.0, 0.0, 0.22), 0.0)),
        "sites": tuple(sites),
    }


def bh_sep():
    ring = _pentagon()
    weights = _fan_weights(np.column_stack((ring, np.zeros(len(ring)))))
    box = np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1))), dtype=float)
    top = np.isclose(box[:, 2], 0.1)
    cores = {int(node): point.copy() for node, point in zip(np.arange(100, 108)[top], box[top], strict=True)}
    prism = np.array([(x, y, z + 0.22) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    sites = []
    for index, (xy, weight) in enumerate(zip(ring, weights, strict=True)):
        sites.append({
            "weight": float(weight),
            "surfaces": (np.array([xy[0], xy[1], 0.1]), np.array([xy[0], xy[1], 0.12])),
            "ids": (None, (200 + 2 * index,)),
            "bary": (None, (1.0,)),
            "allowed": (tuple(cores), None),
            "cores": (cores, None),
            "radii": (0.0, 0.0),
        })
    return {
        "name": "BH-sep",
        "family": "BH",
        "gap_m": 0.02,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 2,
        "length_m": _length((box, 0.0), (prism, 0.0)),
        "sites": tuple(sites),
    }


def hh_sep():
    ring = _pentagon()
    weights = _fan_weights(np.column_stack((ring, np.full(len(ring), 0.1))))
    prism = np.array([(x, y, z) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    shifted = prism + (0.0, 0.0, 0.22)
    sites = []
    for index, (xy, weight) in enumerate(zip(ring, weights, strict=True)):
        sites.append({
            "weight": float(weight),
            "surfaces": (np.array([xy[0], xy[1], 0.1]), np.array([xy[0], xy[1], 0.12])),
            "ids": ((101 + 2 * index,), (200 + 2 * index,)),
            "bary": ((1.0,), (1.0,)),
        })
    return {
        "name": "HH-sep",
        "family": "HH",
        "gap_m": 0.02,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 2,
        "length_m": _length((prism, 0.0), (shifted, 0.0)),
        "sites": tuple(sites),
    }


def bb_touch():
    normal = np.array([0.0, 0.0, 1.0])
    corners = ((-0.6, -0.1), (0.6, -0.1), (0.6, 0.1), (-0.6, 0.1))
    first_ids = {(-0.6, -0.1): 101, (-0.6, 0.1): 103, (0.6, -0.1): 105, (0.6, 0.1): 107}
    second_ids = {(-0.6, -0.1): 200, (-0.6, 0.1): 202, (0.6, -0.1): 204, (0.6, 0.1): 206}
    box = np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1))), dtype=float)
    sites = []
    for x, y in corners:
        sites.append({
            "weight": 0.25,
            "surfaces": (np.array([x, y, 0.1]), np.array([x, y, 0.1])),
            "ids": ((first_ids[(x, y)],), (second_ids[(x, y)],)),
            "bary": ((1.0,), (1.0,)),
        })
    return {
        "name": "BB-touch",
        "family": "BB",
        "gap_m": 0.0,
        "normal": normal,
        "dimension": 2,
        "length_m": _length((box, 0.0), (box + (0.0, 0.0, 0.20), 0.0)),
        "sites": tuple(sites),
    }


def bh_touch():
    ring = _pentagon()
    weights = _fan_weights(np.column_stack((ring, np.zeros(len(ring)))))
    box = np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1))), dtype=float)
    top = np.isclose(box[:, 2], 0.1)
    cores = {int(node): point.copy() for node, point in zip(np.arange(100, 108)[top], box[top], strict=True)}
    prism = np.array([(x, y, z + 0.20) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    sites = []
    for index, (xy, weight) in enumerate(zip(ring, weights, strict=True)):
        sites.append({
            "weight": float(weight),
            "surfaces": (np.array([xy[0], xy[1], 0.1]), np.array([xy[0], xy[1], 0.1])),
            "ids": (None, (200 + 2 * index,)),
            "bary": (None, (1.0,)),
            "allowed": (tuple(cores), None),
            "cores": (cores, None),
            "radii": (0.0, 0.0),
        })
    return {
        "name": "BH-touch",
        "family": "BH",
        "gap_m": 0.0,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 2,
        "length_m": _length((box, 0.0), (prism, 0.0)),
        "sites": tuple(sites),
    }


def hh_touch():
    ring = _pentagon()
    weights = _fan_weights(np.column_stack((ring, np.full(len(ring), 0.1))))
    prism = np.array([(x, y, z) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    shifted = prism + (0.0, 0.0, 0.20)
    sites = []
    for index, (xy, weight) in enumerate(zip(ring, weights, strict=True)):
        sites.append({
            "weight": float(weight),
            "surfaces": (np.array([xy[0], xy[1], 0.1]), np.array([xy[0], xy[1], 0.1])),
            "ids": ((101 + 2 * index,), (200 + 2 * index,)),
            "bary": ((1.0,), (1.0,)),
        })
    return {
        "name": "HH-touch",
        "family": "HH",
        "gap_m": 0.0,
        "normal": np.array([0.0, 0.0, 1.0]),
        "dimension": 2,
        "length_m": _length((prism, 0.0), (shifted, 0.0)),
        "sites": tuple(sites),
    }


def cap_side():
    normal = np.array([0.0, 0.0, 1.0])
    capsule = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    sphere = np.array([[0.0, 0.0, 0.09]])
    return {
        "name": "cap_side",
        "case": "cap_side",
        "family": "CS",
        "gap_m": -0.01,
        "normal": normal,
        "dimension": 0,
        "length_m": _length((capsule, 0.05), (sphere, 0.05)),
        "sites": ({
            "weight": 1.0,
            "surfaces": (np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.04])),
            "ids": ((100, 101), (200,)),
            "bary": ((0.5, 0.5), (1.0,)),
        },),
    }


def cap_end():
    normal = np.array([1.0, 0.0, 0.0])
    capsule = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    sphere = np.array([[0.59, 0.0, 0.0]])
    return {
        "name": "cap_end",
        "case": "cap_end",
        "family": "CS",
        "gap_m": -0.01,
        "normal": normal,
        "dimension": 0,
        "length_m": _length((capsule, 0.05), (sphere, 0.05)),
        "sites": ({
            "weight": 1.0,
            "surfaces": (np.array([0.55, 0.0, 0.0]), np.array([0.54, 0.0, 0.0])),
            "ids": ((101,), (200,)),
            "bary": ((1.0,), (1.0,)),
        },),
    }


def cap_oblique():
    normal = np.array([0.6, 0.0, 0.8])
    capsule = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    sphere = np.array([[0.575, 0.0, 0.1]])
    return {
        "name": "cap_oblique",
        "case": "cap_oblique",
        "family": "CS",
        "gap_m": 0.025,
        "normal": normal,
        "dimension": 0,
        "length_m": _length((capsule, 0.05), (sphere, 0.05)),
        "sites": ({
            "weight": 1.0,
            "surfaces": (np.array([0.53, 0.0, 0.04]), np.array([0.545, 0.0, 0.06])),
            "ids": ((101,), (200,)),
            "bary": ((1.0,), (1.0,)),
        },),
    }


def h_vertex():
    normal = np.array([0.8, 0.0, 0.6])
    vertex = np.array([0.6, 0.0, 0.1])
    center = np.array([0.68, 0.0, 0.16])
    ring = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0.0), (0.3, 0.1), (-0.5, 0.1))
    prism = np.array([(x, y, z) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    return {
        "name": "h_vertex",
        "case": "h_vertex",
        "family": "HS",
        "gap_m": 0.05,
        "normal": normal,
        "dimension": 0,
        "length_m": _length((prism, 0.0), (center.reshape(1, 3), 0.05)),
        "sites": ({
            "weight": 1.0,
            "surfaces": (vertex, center - 0.05 * normal),
            "ids": ((105,), (200,)),
            "bary": ((1.0,), (1.0,)),
        },),
    }


def h_slant():
    normal = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    face = np.array([0.55, -0.05, 0.0])
    center = np.array([0.63, -0.13, 0.0])
    ring = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0.0), (0.3, 0.1), (-0.5, 0.1))
    prism = np.array([(x, y, z) for x, y in ring for z in (-0.1, 0.1)], dtype=float)
    cores = {
        102: np.array([0.5, -0.1, -0.1]),
        103: np.array([0.5, -0.1, 0.1]),
        104: np.array([0.6, 0.0, -0.1]),
        105: np.array([0.6, 0.0, 0.1]),
    }
    return {
        "name": "h_slant",
        "case": "h_slant",
        "family": "HS",
        "gap_m": 0.08 * np.sqrt(2.0) - 0.05,
        "normal": normal,
        "dimension": 0,
        "length_m": _length((prism, 0.0), (center.reshape(1, 3), 0.05)),
        "sites": ({
            "weight": 1.0,
            "surfaces": (face, center - 0.05 * normal),
            "ids": (None, (200,)),
            "bary": (None, (1.0,)),
            "allowed": (tuple(cores), None),
            "cores": (cores, None),
            "radii": (0.0, 0.05),
        },),
    }


def end_box():
    normal = np.array([-5.0, 0.0, 2.0]) / np.sqrt(29.0)
    core = np.array([0.7, 0.0, 0.0])
    face = np.array([0.6, 0.0, 0.04])
    capsule = np.array([[0.7, 0.0, 0.0], [1.7, 0.0, 0.0]])
    box = np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1))), dtype=float)
    box[:, 2] += 0.14
    return {
        "name": "end_box",
        "case": "end_box",
        "family": "CB",
        "gap_m": float(np.sqrt(29.0) / 50.0 - 0.05),
        "normal": normal,
        "dimension": 0,
        "length_m": _length((capsule, 0.05), (box, 0.0)),
        "sites": ({
            "weight": 1.0,
            "surfaces": (core + 0.05 * normal, face),
            "ids": ((100,), (204, 206)),
            "bary": ((1.0,), (0.5, 0.5)),
        },),
    }


def cc_skew():
    normal = np.array([0.0, 0.0, 1.0])
    first = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    second = np.array([[0.0, -0.5, 0.09], [0.0, 0.5, 0.09]])
    return {
        "name": "cc_skew",
        "case": "cc_skew",
        "family": "CC",
        "gap_m": -0.01,
        "normal": normal,
        "dimension": 0,
        "length_m": _length((first, 0.05), (second, 0.05)),
        "sites": ({
            "weight": 1.0,
            "surfaces": (np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.04])),
            "ids": ((100, 101), (200, 201)),
            "bary": ((0.5, 0.5), (0.5, 0.5)),
        },),
    }


def cc_end_end():
    normal = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    first = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    second = np.array([[0.6, 0.1, 0.0], [0.6, 1.1, 0.0]])
    core_a = np.array([0.5, 0.0, 0.0])
    core_b = np.array([0.6, 0.1, 0.0])
    return {
        "name": "cc_end_end",
        "case": "cc_end_end",
        "family": "CC",
        "gap_m": float(np.sqrt(2.0) / 10.0 - 0.1),
        "normal": normal,
        "dimension": 0,
        "length_m": _length((first, 0.05), (second, 0.05)),
        "sites": ({
            "weight": 1.0,
            "surfaces": (core_a + 0.05 * normal, core_b - 0.05 * normal),
            "ids": ((101,), (200,)),
            "bary": ((1.0,), (1.0,)),
        },),
    }


def reversed_view(spec):
    sites = []
    for site in spec["sites"]:
        swapped = {
            "weight": site["weight"],
            "surfaces": (site["surfaces"][1], site["surfaces"][0]),
            "ids": (site["ids"][1], site["ids"][0]),
            "bary": (site["bary"][1], site["bary"][0]),
        }
        for key in ("allowed", "cores", "radii"):
            if key in site:
                swapped[key] = (site[key][1], site[key][0])
        sites.append(swapped)
    viewed = dict(spec)
    viewed["name"] = spec["name"] + "-reverse"
    viewed["normal"] = -np.asarray(spec["normal"], dtype=float)
    viewed["sites"] = tuple(sites)
    return viewed


def cycled(spec):
    cycle = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))

    def turn(point):
        return cycle @ np.asarray(point, dtype=float)

    sites = []
    for site in spec["sites"]:
        updated = dict(site)
        updated["surfaces"] = tuple(turn(point) for point in site["surfaces"])
        if "cores" in site:
            updated["cores"] = tuple(
                None if cores is None else {int(node): turn(point) for node, point in cores.items()}
                for cores in site["cores"]
            )
        sites.append(updated)
    viewed = dict(spec)
    viewed["name"] = spec["name"] + "-cycle"
    viewed["normal"] = turn(spec["normal"])
    viewed["sites"] = tuple(sites)
    return viewed


def tilted(spec):
    tilt = np.array(((0.6, 0.0, 0.8), (0.0, 1.0, 0.0), (-0.8, 0.0, 0.6)))

    def turn(point):
        return tilt @ np.asarray(point, dtype=float)

    sites = []
    for site in spec["sites"]:
        updated = dict(site)
        updated["surfaces"] = tuple(turn(point) for point in site["surfaces"])
        if "cores" in site:
            updated["cores"] = tuple(
                None if cores is None else {int(node): turn(point) for node, point in cores.items()}
                for cores in site["cores"]
            )
        sites.append(updated)
    viewed = dict(spec)
    viewed["name"] = spec["name"] + "-tilt"
    viewed["normal"] = turn(spec["normal"])
    viewed["sites"] = tuple(sites)
    return viewed


def permuted(spec):
    sites = []
    for site in spec["sites"]:
        updated = dict(site)
        updated["ids"] = tuple(None if ids is None else tuple(reversed(ids)) for ids in site["ids"])
        updated["bary"] = tuple(None if weights is None else tuple(reversed(weights)) for weights in site["bary"])
        sites.append(updated)
    viewed = dict(spec)
    viewed["name"] = spec["name"] + "-permute"
    viewed["sites"] = tuple(sites)
    return viewed


def cycle_reverse_ids(spec):
    turned = cycled(spec)

    def relabel(side, node):
        index = int(node) - (100 if side == 0 else 200)
        return 2**31 - 1 - 101 * (side + 1) - 7 * index

    sites = []
    for site in turned["sites"]:
        remapped = dict(site)
        remapped["ids"] = tuple(
            None if ids is None else tuple(relabel(side, node) for node in ids)
            for side, ids in enumerate(site["ids"])
        )
        if "cores" in site:
            remapped["cores"] = tuple(
                None if cores is None else {relabel(side, node): point for node, point in cores.items()}
                for side, cores in enumerate(site["cores"])
            )
        if "allowed" in site:
            remapped["allowed"] = tuple(
                None if allowed is None else tuple(relabel(side, node) for node in allowed)
                for side, allowed in enumerate(site["allowed"])
            )
        swapped = dict(remapped)
        swapped["surfaces"] = (remapped["surfaces"][1], remapped["surfaces"][0])
        swapped["ids"] = (remapped["ids"][1], remapped["ids"][0])
        swapped["bary"] = (remapped["bary"][1], remapped["bary"][0])
        for key in ("allowed", "cores", "radii"):
            if key in remapped:
                swapped[key] = (remapped[key][1], remapped[key][0])
        sites.append(swapped)
    viewed = dict(turned)
    viewed["name"] = spec["name"] + "-cycle-reverse-ids"
    viewed["normal"] = -np.asarray(turned["normal"], dtype=float)
    viewed["sites"] = tuple(sites)
    return viewed


def moment(spec):
    total = np.zeros(3)
    for site in spec["sites"]:
        midpoint = 0.5 * (site["surfaces"][0] + site["surfaces"][1])
        total += site["weight"] * midpoint
    return total


# B1 qualification extends the exact pre-existing K1 reference bytes.
# Legacy SHA256 e1020644b5cfc1dea816f508db9616f57f7063872f2986f9ac73af2db8de1e6a.
# No production geometry, hull routine, or device output enters these oracles.
def polygon_quadrature(polygon):
    polygon = np.asarray(polygon, dtype=float)
    center = polygon.mean(axis=0)
    following = np.roll(polygon, -1, axis=0)
    delta, delta_next = polygon - center, following - center
    twice = delta[:, 0] * delta_next[:, 1] - delta[:, 1] * delta_next[:, 0]
    assert np.all(twice >= 0) and float(twice.sum()) > 0
    weights = (twice + np.roll(twice, 1)) / (3 * twice.sum()) + 1 / (3 * len(polygon))
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    area = float(cross.sum()) / 2
    centroid = ((polygon + following) * cross[:, None]).sum(axis=0) / (6 * area)
    assert np.allclose(weights @ polygon, centroid, atol=1e-13, rtol=0)
    return weights, area, centroid


def polygon_case(name, variant="identity"):
    rectangle = [(-1., -.4), (1., -.4), (1., .4), (-1., .4)]
    diamond = [(0., -1.), (1., 0.), (0., 1.), (-1., 0.)]
    if name in ("trapezoid", "trapezoid_refined"):
        a = [(0., 0.), (2., 0.), (1., 1.), (0., 1.)]
        if name.endswith("refined"):
            a = [point for i, p in enumerate(a) for point in
                 (p, tuple((np.asarray(p) + a[(i + 1) % 4]) / 2))]
        b, intersection = list(a), list(a)
    elif name == "partial_bb":
        a, b = rectangle, [(-.4, -1.), (.4, -1.), (.4, 1.), (-.4, 1.)]
        intersection = [(-.4, -.4), (.4, -.4), (.4, .4), (-.4, .4)]
    elif name == "partial_bh":
        a = [(-1., -.5), (1., -.5), (1., .5), (-1., .5)]
        b = diamond
        intersection = [(-1., 0.), (-.5, -.5), (.5, -.5), (1., 0.), (.5, .5), (-.5, .5)]
    elif name == "partial_hh":
        a, b = diamond, [(-.8, -.8), (.8, -.8), (.8, .8), (-.8, .8)]
        intersection = [(-.8, -.2), (-.2, -.8), (.2, -.8), (.8, -.2),
                        (.8, .2), (.2, .8), (-.2, .8), (-.8, .2)]
    else:
        raise ValueError(name)
    core0 = np.asarray([(x, y, z) for x, y in a for z in (-.1, .1)])
    core1 = np.asarray([(x, y, z) for x, y in b for z in (.09, .29)])
    surface0 = np.asarray([(x, y, .1) for x, y in intersection])
    surface1 = np.asarray([(x, y, .09) for x, y in intersection])
    weights, area, centroid = polygon_quadrature(intersection)
    rotation, quaternion = np.eye(3), (0., 0., 0., 1.)
    anchor = np.zeros(3)
    if variant == "cycle":
        rotation = np.asarray([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        quaternion = (.5, .5, .5, .5)
        anchor = np.asarray([4., -2., 1.])
    elif variant == "tilt":
        rotation = np.asarray([[.6, 0., .8], [0., 1., 0.], [-.8, 0., .6]])
        quaternion = (0., 5 ** -.5, 0., 2 * 5 ** -.5)
    elif variant not in ("identity", "reverse", "permute"):
        raise ValueError(variant)
    ids0 = tuple(100 + i for i in range(len(core0)))
    ids1 = tuple(300 + i for i in range(len(core1)))
    world0, world1 = core0 @ rotation.T + anchor, core1 @ rotation.T + anchor
    if variant == "permute":
        order0 = np.arange(len(core0))[::-1]
        order1 = np.roll(np.arange(len(core1)), 3)
        core0, world0 = core0[order0], world0[order0]
        core1, world1 = core1[order1], world1[order1]
        ids0 = tuple(ids0[i] for i in order0)
        ids1 = tuple(ids1[i] for i in order1)
    case = dict(
        name=f"{name}-{variant}", keys=((17, 2, 11), (17, 3, 21)), kinds=("H", "H"),
        ids=(ids0, ids1), radii=(0., 0.), local=(core0, core1),
        points=(world0, world1), positions=(anchor, anchor),
        registration_reverse=variant == "permute", quaternion=quaternion,
    )
    spec = dict(
        name=case["name"], dimension=2, gap_m=-.01,
        normal=rotation @ np.array([0., 0., 1.]),
        length_m=_length((world0, 0.), (world1, 0.)), area_m2=area,
        centroid=rotation @ np.array([*centroid, .095]),
        sites=tuple(dict(weight=float(w), surfaces=(s0 @ rotation.T, s1 @ rotation.T))
                    for w, s0, s1 in zip(weights, surface0, surface1, strict=True)),
    )
    return case, spec
