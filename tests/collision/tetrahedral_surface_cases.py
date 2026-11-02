"""Register volume cells by their original stable node and cell identities."""

import numpy as np
from case_catalog import geometry_for

import phymeo as ph

CASES = {
    "RD04": ("base", "reindexed", "clear", "internal_face_trap", "lower_holdout"),
    "DD04": ("base", "thick", "lateral_clear", "reversed_thick", "moving_tet_holdout"),
}


def scene(family, name, world=0, reverse=False):
    _, v, g = geometry_for(family, name)
    solid, other = g["A"], g["B"]
    start = np.asarray(solid["vertices"], dtype=float)
    end = np.asarray(solid.get("vertices_end", start), dtype=float)
    order = np.asarray(v.get("node_order", range(len(start))))
    rows = np.argsort(order)[np.asarray(solid["tets"])]
    shapes, paths = [], {}
    for i in v.get("cell_order", range(len(rows))):
        key = ph.CollisionKey(world, 2 if reverse else 0, i)
        row = rows[i]
        shapes.append(ph.CollisionShape(key, "tetrahedron", tuple(order[row])))
        paths[key] = ph.CollisionMotion(start[order][row], end[order][row])
    key = ph.CollisionKey(world, 1, 0)
    if family == "RD04":
        shapes.append(ph.CollisionShape(key, "sphere", (0,), other["radius"]))
        paths[key] = ph.CollisionMotion([other["center_start"]], [other["center_end"]])
    else:
        shapes.append(
            ph.CollisionShape(
                key, "triangle", tuple(v.get("order_B", range(3))), v.get("half_thickness_m", 0.0)
            )
        )
        paths[key] = ph.CollisionMotion(other["vertices_start"], other["vertices_end"])
    return ph.CollisionModel(shapes), paths, key


def partners(out, key):
    return [row for row in out.pairs if key in (row.first, row.second)]
