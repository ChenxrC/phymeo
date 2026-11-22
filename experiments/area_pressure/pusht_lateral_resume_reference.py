"""Independent exact half-space clipping and CPU FP64 first-step reference.

Only immutable material declarations are shared with the experiment. This file
never calls I1 intersection, I0 quadrature, the local publication path or CUDA.
Exact affine points retain differences before the final world-coordinate rounding.
"""

from fractions import Fraction as F

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError("lateral independent reference: " + message)


def _r(value):
    return F(float(value))


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), F(0))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _inverse(matrix):
    n = len(matrix)
    rows = [list(row) + [F(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if rows[i][j]), None)
        require(pivot is not None, "nonsingular pressure tetrahedron")
        rows[j], rows[pivot] = rows[pivot], rows[j]
        factor = rows[j][j]
        rows[j] = [x / factor for x in rows[j]]
        for i in range(n):
            if i != j:
                factor = rows[i][j]
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[j], strict=True)]
    return tuple(tuple(row[n:]) for row in rows)


def _transform(vertices, transform):
    rotation, position = transform
    rotation = [tuple(map(_r, row)) for row in rotation]
    position = tuple(map(_r, position))
    return [
        tuple(
            _dot(row, tuple(map(_r, vertex))) + shift
            for row, shift in zip(rotation, position, strict=True)
        )
        for vertex in vertices
    ]


def _clip(polygon, planes):
    for plane in planes:
        if not polygon:
            return []
        result = []
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            da, db = _dot(plane, (*a, F(1))), _dot(plane, (*b, F(1)))
            if da >= 0:
                result.append(a)
            if (da < 0 < db) or (db < 0 < da):
                u = da / (da - db)
                result.append(tuple(x + u * (y - x) for x, y in zip(a, b, strict=True)))
        polygon = list(dict.fromkeys(result))
    while len(polygon) >= 3:
        remove = [
            i
            for i in range(len(polygon))
            if not any(
                _cross(
                    _sub(polygon[i], polygon[i - 1]),
                    _sub(polygon[(i + 1) % len(polygon)], polygon[i]),
                )
            )
        ]
        if not remove:
            break
        polygon = [p for i, p in enumerate(polygon) if i not in remove]
    if len(polygon) < 3:
        return []
    i = min(range(len(polygon)), key=polygon.__getitem__)
    return polygon[i:] + polygon[:i]


def reference_quadrature(
    pressure_vertices,
    cells,
    nodal_pressure,
    rigid_vertices,
    faces,
    *,
    pressure_transform,
    rigid_transform,
    cell_ids=None,
    face_ids=None,
):
    """Own exact affine clipping and degree-two triangle rule, world-axis outputs."""
    soft, rigid = (
        _transform(pressure_vertices, pressure_transform),
        _transform(rigid_vertices, rigid_transform),
    )
    cell_ids = tuple(str(i) for i in range(len(cells))) if cell_ids is None else tuple(cell_ids)
    face_ids = tuple(str(i) for i in range(len(faces))) if face_ids is None else tuple(face_ids)
    points, normals, areas, pressures, gradients, sources, polygons = [], [], [], [], [], [], {}
    fields, seen = {}, {}
    for ci in sorted(range(len(cells)), key=lambda i: cell_ids[i]):
        tetra = [soft[int(i)] for i in cells[ci]]
        inverse = _inverse([(*p, F(1)) for p in tetra])
        planes = [tuple(inverse[j][i] for j in range(4)) for i in range(4)]
        values = [_r(nodal_pressure[int(i)]) for i in cells[ci]]
        field = tuple(_dot(row, values) for row in inverse)
        fields[cell_ids[ci]] = field
        for fi in sorted(range(len(faces)), key=lambda i: face_ids[i]):
            face = [rigid[int(i)] for i in faces[fi]]
            if any(
                max(p[axis] for p in face) < min(p[axis] for p in tetra)
                or min(p[axis] for p in face) > max(p[axis] for p in tetra)
                for axis in range(3)
            ):
                continue
            poly = _clip(face, planes)
            if not poly:
                continue
            outward = _cross(_sub(face[1], face[0]), _sub(face[2], face[0]))
            require(_dot(field[:3], outward) > 0, "positive effective material gradient")
            key = (face_ids[fi], tuple(sorted(poly)))
            if key in seen:
                require(seen[key] == field[:3], "shared face has one pressure gradient")
                continue
            seen[key] = field[:3]
            source = (cell_ids[ci], face_ids[fi])
            polygons[source] = poly
            for j in range(1, len(poly) - 1):
                triangle = (poly[0], poly[j], poly[j + 1])
                cross = _cross(_sub(triangle[1], triangle[0]), _sub(triangle[2], triangle[0]))
                require(_dot(cross, outward) > 0, "positive finite triangle winding")
                vector = np.asarray([float(x) for x in cross])
                length = float(np.linalg.norm(vector))
                require(np.isfinite(length) and length > 0, "representable independent area")
                normal = vector / length
                effective = float(np.asarray([float(x) for x in field[:3]]) @ normal)
                for slot in range(3):
                    weights = [F(2, 3) if i == slot else F(1, 6) for i in range(3)]
                    point = tuple(
                        sum((w * p[a] for w, p in zip(weights, triangle, strict=True)), F(0))
                        for a in range(3)
                    )
                    pressure = _dot(field, (*point, F(1)))
                    require(pressure >= 0, "nonnegative exact pressure")
                    points.append([float(x) for x in point])
                    normals.append(normal)
                    areas.append(length / 6)
                    pressures.append(float(pressure))
                    gradients.append(effective)
                    sources.append(source)
    return dict(
        points=np.asarray(points).reshape(-1, 3),
        normals=np.asarray(normals).reshape(-1, 3),
        dA=np.asarray(areas),
        pressure=np.asarray(pressures),
        gradient=np.asarray(gradients),
        source_ids=sources,
        polygons=polygons,
        fields=fields,
    )


def _close(actual, expected, name, *, rtol=3e-4, atol=5e-5):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.shape == expected.shape
        and np.isfinite(actual).all()
        and np.isfinite(expected).all(),
        name + " finite shape",
    )
    require(np.allclose(actual, expected, rtol=rtol, atol=atol), name + " independent value")
    return float(np.max(abs(actual - expected), initial=0))


def _native(receipt, source):
    candidates = [c for c in source["contacts"] if c["retained"]]
    used, mapping = set(), {}
    for ci, contact in enumerate(receipt["contacts"]):
        options = [
            c
            for c in candidates
            if list(c["geometry_ids"]) == contact["geometry_ids"] and c["contact_index"] not in used
        ]
        require(bool(options), "complete native geometry identity")
        native = min(options, key=lambda c: np.linalg.norm(c["position"] - contact["position"]))
        _close(contact["position"], native["position"], "native contact position")
        _close(contact["frame"], native["frame"], "native contact frame")
        require(
            contact["dimension"] == 3 and contact["active"] == (native["efc_address"] >= 0),
            "native activation/dimension",
        )
        used.add(native["contact_index"])
        for row in receipt["constraints"]:
            if row["constraint_type"] == 6 and row["source_id"] == ci:
                mapping[native["efc_address"] + row["side_or_edge"]] = row
    for row in receipt["constraints"]:
        if row["constraint_type"] == 3:
            owner = next(
                o
                for o in source["owners"]
                if o["kind"] == "limit"
                and o["joint_id"] == row["source_id"]
                and o["side"] == row["side_or_edge"]
            )
            mapping[owner["source_row"]] = row
    require(
        used == {c["contact_index"] for c in candidates}
        and set(mapping) == set(source["retained_indices"])
        and len(receipt["constraints"]) == len(mapping),
        "native row/contact bijection",
    )
    for i, key in enumerate(source["retained_indices"]):
        row = mapping[int(key)]
        for field, source_field in (
            ("jacobian", "jacobian"),
            ("regularization", "regularization"),
            ("reference_acceleration", "reference_acceleration"),
        ):
            _close(row[field], source[source_field][i], "native " + field)
        require(row["force"] >= 0, "nonnegative native edge/limit force")
    return np.array([mapping[int(i)]["force"] for i in source["retained_indices"]])


def _point_jacobians(mj, model, data, points, bodies):
    first, second = (np.zeros((len(points), 3, model.nv)) for _ in range(2))
    for i, point in enumerate(points):
        mj.mj_jac(model, data, first[i], None, point, bodies[0])
        mj.mj_jac(model, data, second[i], None, point, bodies[1])
    return first, second


def _bind_first_packet(packet, row):
    import hashlib
    import json

    require(
        packet["schema"] == "phymeo-pusht-lateral-resume-first-step-v1"
        and packet["input_state"] == row["input_state"]
        and packet["output_state"] == row["output_state"]
        and packet["measurement"] == row["measurement"],
        "first-step physical publication binding",
    )
    saved = row.get("submitted_control", row.get("control"))
    keys = ("tick", "output_tick", "ctrl", "qfrc_applied", "xfrc_applied")
    require(
        set(packet["control"]) == set(keys)
        and set(saved) == set(keys) | {"channels_sha256"}
        and packet["control"] == {key: saved[key] for key in keys},
        "first-step shared control fields",
    )
    channels = {key: saved[key] for key in keys[2:]}
    digest = hashlib.sha256(
        json.dumps(channels, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    require(saved["channels_sha256"] == digest, "first-step derived control channel digest")
    for key in keys[2:]:
        require(
            np.asarray(packet["control"][key], np.float32).tobytes()
            == np.asarray(saved[key], np.float32).tobytes(),
            "first-step shared control words",
        )


def audit_first_step(packet, row):
    """Compare the saved first real receipt to independently assembled coefficients."""
    import mujoco as mj

    from phymeo.training.manipulation_tasks import manipulation_scenes

    from .pusht_entry_rows import assemble_entry
    from .pusht_entry_working_set import _reference
    from .pusht_public_scene import build_side_asset

    _bind_first_packet(packet, row)

    scene = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    model = mj.MjModel.from_xml_string(scene.mjcf)
    before, after, command, receipt = (
        packet[k] for k in ("input_state", "output_state", "control", "receipt")
    )
    for channel in ("ctrl", "qfrc_applied", "xfrc_applied"):
        require(receipt["stored_control"][channel] == command[channel], "first-step stored control")
    require(
        receipt["input_q"] == before["q"]
        and receipt["input_v"] == before["v"]
        and receipt["input_time"] == before["time"]
        and receipt["output_time"] == after["time"]
        and receipt["tick"] == after["local_tick"],
        "first receipt phase",
    )
    source = assemble_entry(
        model,
        dict(
            q=before["q"],
            v=before["v"],
            time=before["time"],
            **{k: command[k] for k in ("ctrl", "qfrc_applied", "xfrc_applied")},
        ),
    )
    data = source["data"]
    asset = build_side_asset(model, side_coverage="all-lateral")
    bodies = (int(model.body("tool").id), int(model.body("object").id))
    quad = reference_quadrature(
        asset.pressure_local.world_vertices,
        asset.pressure_local.cells,
        asset.pressure_local.nodal_pressure,
        asset.rigid_local.world_vertices,
        asset.rigid_local.faces,
        pressure_transform=(data.xmat[bodies[0]].reshape(3, 3), data.xpos[bodies[0]]),
        rigid_transform=(data.xmat[bodies[1]].reshape(3, 3), data.xpos[bodies[1]]),
        cell_ids=asset.pressure_local.cell_ids,
        face_ids=asset.rigid_local.face_ids,
    )
    (pair,) = receipt["raw_contact_fields"]["area_friction"]
    require(
        pair["body_ids"] == list(bodies)
        and pair["mu"] == float(np.float32(0.7))
        and pair["slip_speed_m_s"] == float(np.float32(0.01)),
        "material body order/friction law",
    )
    observed = pair["source_quadrature"]
    points = np.asarray(observed["points"], float).reshape(-1, 3)
    dA, pressure, gradient = (
        np.asarray(observed[k], float) for k in ("dA", "pressure", "gradient")
    )
    normals = np.asarray(observed["normals"], float).reshape(-1, 3)
    source_ids = list(map(tuple, observed["source_ids"]))
    require(
        len(points) == len(dA) == len(pressure) == len(gradient) == len(source_ids)
        and np.all(dA > 0)
        and np.all(gradient > 0),
        "actual finite source quadrature",
    )
    require(
        set(source_ids) == set(quad["source_ids"]),
        "independent positive-area source cell/face coverage",
    )
    # Each production point is checked as data. The independent polygon areas
    # and moments do not depend on its chosen fan diagonal.
    maximum_plane_error = 0.0
    for source_id in set(source_ids):
        ids = np.array([i for i, key in enumerate(source_ids) if key == source_id])
        expected_ids = np.array([i for i, key in enumerate(quad["source_ids"]) if key == source_id])
        polygon = quad["polygons"][source_id]
        field = quad["fields"][source_id[0]]
        origin = polygon[0]
        face_normal = np.cross(
            np.array([float(x) for x in _sub(polygon[1], origin)]),
            np.array([float(x) for x in _sub(polygon[2], origin)]),
        )
        face_normal /= np.linalg.norm(face_normal)
        _close(normals[ids], np.tile(face_normal, (len(ids), 1)), "point world normal")
        expected_pressure = np.array(
            [float(_dot(field, (*tuple(map(_r, p)), F(1)))) for p in points[ids]]
        )
        _close(pressure[ids], expected_pressure, "point pressure", rtol=2e-3, atol=0.05)
        effective = np.asarray([float(x) for x in field[:3]]) @ face_normal
        _close(
            gradient[ids],
            np.full(len(ids), effective),
            "point effective gradient",
            rtol=2e-3,
            atol=0.05,
        )
        local = points[ids] - np.array([float(x) for x in origin])
        plane_error = float(np.max(abs(local @ face_normal), initial=0))
        maximum_plane_error = max(maximum_plane_error, plane_error)
        require(plane_error <= 5e-8, "point lies on independent finite face")
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            edge = np.array([float(x) for x in _sub(b, a)])
            inward = np.cross(face_normal, edge)
            distance = (points[ids] - np.array([float(x) for x in a])) @ inward
            require(
                np.all(distance >= -5e-8 * np.linalg.norm(inward)),
                "point lies within independent clipped polygon",
            )
        _close(
            dA[ids].sum(),
            quad["dA"][expected_ids].sum(),
            "source clipped area",
            rtol=2e-3,
            atol=2e-8,
        )
        for power in (1, 2):
            actual_moment = np.einsum("s,si->i", dA[ids], points[ids] ** power)
            expected_moment = np.einsum(
                "s,si->i", quad["dA"][expected_ids], quad["points"][expected_ids] ** power
            )
            _close(actual_moment, expected_moment, "source quadrature moment", rtol=2e-3, atol=2e-8)
    triangles = packet["geometry"]["surface"]["triangles"]
    require(3 * len(triangles) == len(points), "triangle/quadrature cardinality")
    bary = np.full((3, 3), 1 / 6) + np.eye(3) * 0.5
    for triangle in triangles:
        ids = [
            i
            for i, sample in enumerate(observed["sample_ids"])
            if sample[0] == triangle["triangle_id"]
        ]
        require(
            len(ids) == 3 and [observed["sample_ids"][i][1] for i in ids] == [0, 1, 2],
            "point/fan/slot identity",
        )
        local, origin = np.asarray(triangle["local_vertices"]), np.asarray(triangle["origin_world"])
        _close(
            origin,
            receipt["dynamics"]["body_position"][bodies[1]],
            "real rigid publication origin",
            rtol=0,
            atol=0,
        )
        _close(
            triangle["vertices"], local + origin, "world triangle reconstruction", rtol=0, atol=0
        )
        _close(
            points[ids], bary @ local + origin, "actual world degree-two points", rtol=0, atol=2e-15
        )
        vector = np.cross(local[1] - local[0], local[2] - local[0])
        _close(
            dA[ids],
            np.full(3, np.linalg.norm(vector) / 6),
            "actual degree-two weights",
            rtol=1e-10,
            atol=1e-20,
        )
    _close(pair["points"], points.astype(np.float32), "device world points", rtol=0, atol=0)
    ja, jb = _point_jacobians(mj, model, data, np.asarray(pair["points"]), bodies)
    _close(pair["jacobian_a"], ja, "tool point Jacobian")
    _close(pair["jacobian_b"], jb, "object point Jacobian")
    force_a, force_b = np.asarray(pair["force_on_a"]), np.asarray(pair["force_on_b"])
    require(np.array_equal(force_b, -force_a), "opposite actual point forces")
    relative = (ja - jb) @ np.asarray(after["v"])
    vn = np.einsum("si,si->s", relative, normals)
    vt = relative - normals * vn[:, None]
    fn = np.maximum(0, dA * gradient * (pressure / gradient - float(receipt["h"]) * vn))
    law_force = fn[:, None] * (
        normals - 0.7 * vt / np.sqrt(np.sum(vt * vt, axis=1) + 0.01**2)[:, None]
    )
    _close(pair["normal_force"], fn, "point normal law", rtol=0.003, atol=0.002)
    _close(force_a, law_force, "point two-sided constitutive force", rtol=0.003, atol=0.002)
    _close(pair["relative_velocity"], relative, "point relative velocity")
    actual_native = _native(receipt, source)
    jref_a, jref_b = _point_jacobians(mj, model, data, quad["points"], bodies)
    h = float(receipt["h"])
    response = source["mass"] + h * np.diag(model.dof_damping)
    expected_v, expected_native, expected_force = _reference(
        source,
        dict(
            jac=jref_a - jref_b,
            normals=quad["normals"],
            stiffness=quad["dA"] * quad["gradient"],
            initial_gap=-quad["pressure"] / quad["gradient"],
        ),
        np.asarray(before["v"]),
        response,
        h,
    )
    expected_a = np.linalg.solve(source["mass"], response @ (expected_v - before["v"])) / h
    velocity_error = _close(
        after["v"], expected_v, "independent common-root velocity", rtol=5e-4, atol=5e-5
    )
    _close(
        h * np.asarray(receipt["qacc"]),
        h * expected_a,
        "physical acceleration",
        rtol=5e-4,
        atol=5e-5,
    )
    _close(actual_native, expected_native, "native coupled forces", rtol=0.003, atol=0.002)
    _close(receipt["dynamics"]["mass_matrix"], source["mass"], "FP64 physical mass")
    actual_area = np.einsum("sij,si->j", ja - jb, force_a)
    actual_total = source["jacobian"].T @ actual_native + actual_area
    generalized_error = _close(
        receipt["qfrc_constraint"],
        actual_total,
        "actual source generalized force",
        rtol=0.003,
        atol=0.002,
    )
    expected_area = np.einsum("sij,si->j", jref_a - jref_b, expected_force)
    _close(actual_area, expected_area, "independent area generalized force", rtol=0.003, atol=0.002)
    _close(
        source["mass"] @ receipt["qacc"],
        source["base_load"] + actual_total,
        "physical momentum",
        rtol=0.003,
        atol=0.002,
    )
    _close(
        response @ receipt["integration_acceleration"],
        source["base_load"] + actual_total,
        "damped response",
        rtol=0.003,
        atol=0.002,
    )
    actual_moment = np.cross(
        np.asarray(pair["points"]) - np.asarray(receipt["dynamics"]["body_com"])[bodies[1]], force_b
    ).sum(axis=0)
    expected_moment = np.cross(quad["points"] - data.xipos[bodies[1]], -expected_force).sum(axis=0)
    _close(
        force_a.sum(axis=0),
        expected_force.sum(axis=0),
        "independent total tool force",
        rtol=0.003,
        atol=0.002,
    )
    moment_error = _close(
        actual_moment, expected_moment, "independent object COM moment", rtol=0.003, atol=2e-4
    )
    _close(
        packet["measurement"]["group_moment_com_world"],
        actual_moment,
        "published object COM moment",
        rtol=3e-4,
        atol=2e-6,
    )
    _close(
        packet["measurement"]["group_force_world"],
        force_b.sum(axis=0),
        "published object force",
        rtol=3e-4,
        atol=2e-6,
    )
    q = np.asarray(before["q"], dtype=float).copy()
    mj.mj_integratePos(model, q, expected_v, h)
    _close(after["q"], q, "independent integrated configuration", rtol=5e-6, atol=2e-7)
    return dict(
        verified=True,
        reference="independent Fraction affine clipping, degree-two integration and FP64 common root",
        samples=len(points),
        reference_samples=len(quad["points"]),
        native_rows=len(actual_native),
        area_m2=float(dA.sum()),
        normal_force_n=float(np.sum(pair["normal_force"])),
        tool_force_world_n=force_a.sum(axis=0).tolist(),
        object_force_world_n=force_b.sum(axis=0).tolist(),
        object_moment_com_world_nm=actual_moment.tolist(),
        reference_object_moment_com_world_nm=expected_moment.tolist(),
        max_velocity_error=velocity_error,
        max_generalized_force_error=generalized_error,
        max_com_moment_error=moment_error,
        max_point_plane_error_m=maximum_plane_error,
        cpu_step_called=False,
        production_geometry_or_quadrature_called=False,
    )
