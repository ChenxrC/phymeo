"""Independent full-interval audit of published N2 common curves on the finite SC1 chart."""

from fractions import Fraction as Q
from functools import lru_cache
from math import comb, inf, lcm, nextafter

import numpy as np
from p65_common_motion import DEGREE, GRID, unpack

from phymeo.validation.bending_bounds import Interval, cross, dot, sqrt_interval


class Rejected(ValueError):
    pass


def test(ok, reason):
    if not ok:
        raise Rejected(reason)


ZERO = (Q(0),)
LEAVES = 64


def poly(coeff):
    a = list(coeff)
    while len(a) > 1 and not a[-1]:
        a.pop()
    return tuple(a) or ZERO


def add(p, q):
    a = list(p) + [Q(0)] * max(0, len(q) - len(p))
    for i, c in enumerate(q):
        a[i] += c
    return poly(a)


def scale(p, s):
    return poly([c * s for c in p])


def sub(p, q):
    return add(p, scale(q, -1))


def times(p, q):
    a = [Q(0)] * (len(p) + len(q) - 1)
    for i, v in enumerate(p):
        if v:
            for j, z in enumerate(q):
                a[i + j] += v * z
    return poly(a)


def integral(p, initial=0):
    return poly([Q(initial)] + [Q(c, i + 1) for i, c in enumerate(p)])


def derive(p):
    return poly([i * c for i, c in enumerate(p) if i])


def one_integral(p):
    return sum((Q(c, i + 1) for i, c in enumerate(p)), Q(0))


@lru_cache(maxsize=1)
def basis():
    out = [(1,), (-1, 2)]
    for _ in range(2, DEGREE + 1):
        out.append(sub(times((-2, 4), out[-1]), out[-2]))
    return out


def power(words):
    return poly(
        [
            Q(sum(int(words[j]) * t[k] for j, t in enumerate(basis()) if k < len(t)), GRID)
            for k in range(DEGREE + 1)
        ]
    )


def round_box(lo, hi):
    d = 1 << 96
    return Interval(Q((lo * d).__floor__(), d), Q((hi * d).__ceil__(), d))


@lru_cache(maxsize=1024)
def bounds(p):
    # Exact Bernstein subdivision with one common denominator per cell.
    d = len(p) - 1
    if d == 0:
        return (round_box(p[0], p[0]),) * LEAVES
    bc = [sum((Q(comb(i, j), comb(d, j)) * p[j] for j in range(i + 1)), Q(0)) for i in range(d + 1)]
    den = lcm(*(v.denominator for v in bc))
    nodes = [[int(v * den) for v in bc]]
    for _ in range(6):
        children = []
        for row in nodes:
            left = []
            right = []
            for k in range(d + 1):
                left.append(row[0] << (d - k))
                right.append(row[-1] << (d - k))
                row = [a + b for a, b in zip(row, row[1:])]
            children.extend((left, right[::-1]))
        nodes = children
        den <<= d
    return tuple(round_box(Q(min(row), den), Q(max(row), den)) for row in nodes)


def xyz_sub(a, b):
    return [sub(x, y) for x, y in zip(a, b, strict=True)]


def cross_poly(a, b):
    i, j, k = 0, 1, 2

    def term(i, j):
        return sub(times(a[i], b[j]), times(a[j], b[i]))

    return [term(j, k), term(k, i), term(i, j)]


def edge_boxes(edge, leaf):
    return [bounds(p)[leaf] for p in edge]


def length_box(vector):
    return sqrt_interval(sum(z.square() for z in vector), 96)


def material_box(e, u, v, rest):
    # Original unit-compliance edges (a,b),(a,c),(a,d); oriented hinge (a,b,c,d).
    edge = [e, u, v]
    force = [[Interval.point(0) for _ in range(3)] for _ in range(4)]
    energy = Interval.point(0)
    for i, vec in enumerate(edge, 1):
        length = length_box(vec)
        test(length.low > 0, "distance domain")
        ext = length - 1
        energy += ext.square() / 2
        for j in range(3):
            f = (ext / length) * vec[j]
            force[0][j] += f
            force[i][j] -= f
    edge_length = length_box(e)
    e2 = dot(e, e)
    nl = cross(e, u)
    nr = cross(v, e)
    l2 = sum(t.square() for t in nl)
    r2 = sum(t.square() for t in nr)
    cosine = dot(e, u) * dot(e, v) - e2 * dot(u, v)
    test(cosine.low > 0 and min(l2.low, r2.low) > Q(1, 4), "hinge orientation or area unsupported")
    z = ((-edge_length * dot(e, cross(u, v))) / cosine).rounded(96)
    # atan(z) = z with error at most |z|^3/3 on this supported principal branch.
    rem = z.magnitude**3 / 3
    angle = z + Interval(-rem, rem)
    test(angle.magnitude + abs(rest) < 1, "hinge branch")
    bend = angle - rest
    energy += bend.square() / 2
    gc = [-edge_length * t / l2 for t in nl]
    gd = [-edge_length * t / r2 for t in nr]
    tc = dot(u, e) / e2
    td = dot(v, e) / e2
    for j in range(3):
        g = [(tc - 1) * gc[j] + (td - 1) * gd[j], -tc * gc[j] - td * gd[j], gc[j], gd[j]]
        for i in range(4):
            force[i][j] -= bend * g[i]
    return force, energy


def up(x):
    den = 1 << 96
    return Q((x * den).__ceil__(), den)


def outward(x):
    return [nextafter(float(x.low), -inf), nextafter(float(x.high), inf)]


def verify_curve(claim, in_packet, out_packet, model, original, reference=None):
    # Only shared low-level history decoding is used; no producer force, collocation or endpoint routine.
    q0, v0, t0 = unpack(in_packet, model.signature)
    q1, v1, t1 = unpack(out_packet, model.signature)
    test(
        np.array_equal(claim["incoming_position"], np.asarray(q0, float))
        and np.array_equal(claim["incoming_velocity"], np.asarray(v0, float)),
        "published inlet velocity or position",
    )
    test(
        claim["schema"] == "p65-common-curve-v1"
        and claim["binding"] == original
        and tuple(claim["owners"]) == tuple(model.owners),
        "original input or owner binding",
    )
    h = Q(float(claim["h"]))
    test(h > 0 and t1 == t0 + h and claim["grid"] == GRID, "time interval or grid")
    mat = np.asarray(claim["material"])
    rows = np.asarray(claim["rows"])
    po = np.asarray(claim["pressure"])
    test(
        mat.shape == (21, 13)
        and rows.shape == (21, 6, 13)
        and po.shape == (21, 6)
        and all(a.dtype == np.dtype("int64") for a in (mat, rows, po)),
        "field schema",
    )
    test(
        np.array_equal(claim["mass"], model.mass)
        and np.array_equal(claim["external"], model.external),
        "original masses and loads",
    )
    test(
        claim["owner_charges"] == 486
        and claim["force_calls"] == 81
        and len(claim["iterations"]) == 4,
        "producer resource ledger",
    )
    mass = [Q(float(x)) for x in model.mass]
    ext = [Q(float(x)) for x in model.external]
    material = [power(mat[:, i]) for i in range(13)]
    pressure = [power(po[:, a]) for a in range(6)]
    mapped = [[power(rows[:, a, i]) for i in range(13)] for a in range(6)]
    q = []
    v = []
    node_moments = []
    for i, m in enumerate(mass):
        f = add((ext[i],), material[i])
        for row in mapped:
            f = add(f, row[i])
        vp = integral(scale(f, h / m), v0[i])
        qp = integral(scale(vp, h), q0[i])
        test(
            sum(qp) == q1[i] and sum(vp) == v1[i],
            "per-coordinate endpoint moment or history mismatch",
        )
        test(
            vp[0] == v0[i] and qp[0] == q0[i] and derive(qp) == scale(vp, h),
            "non-authoritative velocity",
        )
        q.append(qp)
        v.append(vp)
        node_moments.append(
            dict(
                material_J=float(h * one_integral(material[i])),
                material_B=float(h * h * one_integral(times((1, -1), material[i]))),
            )
        )
    x = [q[i : i + 3] for i in range(0, 12, 3)]
    edges = [xyz_sub(x[i], x[0]) for i in (1, 2, 3)]
    faces = model.faces
    normals = [cross_poly(xyz_sub(x[f[1]], x[f[0]]), xyz_sub(x[f[2]], x[f[0]])) for f in faces]
    rho = Q(float(model.rho))
    width = Q(1, 500)
    stiffness = Q(500)
    damp = Q(1000)
    area = Q(1, 6)
    patch = model.patch_vertices
    test(
        {tuple(Q(float(a)) for a in pt) for pt in patch}
        == {(Q(i), Q(j), -rho) for i in (-2, 2) for j in (-2, 2)},
        "unproved finite patch chart",
    )
    gaps = [sub(q[3 * i + 2], q[-1]) for f in faces for i in f]
    rates = [scale(derive(p), 1 / h) for p in gaps]
    gfac = []
    for p in gaps:
        k = 0
        while k + 1 < len(p) and not p[k]:
            k += 1
        gfac.append((k, bounds(p[k:])))
    nominal = []
    dn = []
    for a in range(6):
        comp = sub((width,), gaps[a])
        factor = sub((1,), scale(rates[a], damp))
        nominal.append(sub(pressure[a], scale(times(comp, factor), area * stiffness)))
        dn.append(scale(times(comp, times(rates[a], rates[a])), area * stiffness * damp))
    Js = [h * one_integral(p) for p in pressure]
    Bs = [h * h * one_integral(times((1, -1), p)) for p in pressure]
    JE = [Q(0)] * 6
    BE = [Q(0)] * 6
    mat_j = [Q(0)] * 13
    mat_b = [Q(0)] * 13
    map_j = [Q(0)] * 13
    map_b = [Q(0)] * 13
    loss_err = Q(0)
    angle_arm = Q(0)
    min_orient = None
    min_rate_factor = None
    min_pressure = None
    min_margin = None
    map_res = []
    for a, f in enumerate(faces.repeat(3, axis=0)):
        idx = int(f[a % 3])
        base = 3 * idx + 2
        map_res.append(
            [
                sub(mapped[a][i], pressure[a])
                if i == base
                else add(mapped[a][i], pressure[a])
                if i == 12
                else mapped[a][i]
                for i in range(13)
            ]
        )
    for leaf in range(LEAVES):
        span = h / LEAVES
        back = h * (1 - Q(leaf, LEAVES))
        force_box, _ = material_box(*(edge_boxes(f, leaf) for f in edges), model.rest)
        for i in range(13):
            expected = force_box[i // 3][i % 3] if i < 12 else Interval.point(0)
            err = up((bounds(material[i])[leaf] - expected).magnitude)
            mat_j[i] += span * err
            mat_b[i] += span * back * err
        for f, ids in enumerate(faces):
            c = [bounds(z)[leaf] for z in normals[f]]
            cp = [bounds(derive(z))[leaf] for z in normals[f]]
            margin = (
                2 - rho - max(bounds(x[i][axis])[leaf].magnitude for i in ids for axis in (0, 1))
            )
            test(c[2].low > 0 and margin > 0, "finite support or changing orientation")
            min_orient = c[2].low if min_orient is None else min(min_orient, c[2].low)
            min_margin = margin if min_margin is None else min(min_margin, margin)
            r = [c[j] / c[2] for j in (0, 1)]
            rp = [(cp[j] - r[j] * cp[2]) / c[2] for j in (0, 1)]
            ss = r[0].square() + r[1].square()
            root = sqrt_interval(1 + ss, 96)
            shift = (rho * ss / (root * (1 + root))).rounded(96)
            shift_rate = (rho * dot(r, rp) / ((1 + ss) * root * h)).rounded(96)
            arm = rho * (r[0].magnitude + r[1].magnitude)
            edge_max = max(
                sum(bounds(sub(x[i][j], x[k][j]))[leaf].magnitude for j in range(3))
                for i, k in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0]))
            )
            dmap = arm * edge_max * (1 + max(z.magnitude for z in r)) / c[2].low
            for corner, node in enumerate(ids):
                a = 3 * f + corner
                test(gfac[a][1][leaf].low >= 0, "negative or unresolved full-path hard gap")
                b = bounds(gaps[a])[leaf]
                u = bounds(rates[a])[leaf]
                comp = width - b - shift
                fac = 1 - damp * (u + shift_rate)
                pc = bounds(pressure[a])[leaf]
                test(comp.low > 0 and fac.low > 0, "layer or damping branch unresolved")
                test(pc.low >= 0, "negative represented pressure")
                min_rate_factor = (
                    fac.low if min_rate_factor is None else min(min_rate_factor, fac.low)
                )
                min_pressure = pc.low if min_pressure is None else min(min_pressure, pc.low)
                residual = bounds(nominal[a])[leaf] + area * stiffness * (
                    shift * (1 - damp * u) + comp * damp * shift_rate
                )
                err = up(residual.magnitude)
                JE[a] += span * err
                BE[a] += span * back * err
                tilt_error = shift.magnitude * (u.magnitude + shift_rate.magnitude) ** 2 + (
                    width - b
                ).magnitude * shift_rate.magnitude * (2 * u.magnitude + shift_rate.magnitude)
                loss_err += span * up(area * stiffness * damp * tilt_error)
                angle_arm += span * up(arm * pc.magnitude)
                for i in range(13):
                    offset = dmap if i < 12 and i // 3 in ids else Q(0)
                    bary = int(i == 3 * int(node) + 2 or i == 12)
                    mismatch = (
                        bounds(map_res[a][i])[leaf].magnitude
                        + offset * pc.magnitude
                        + (offset + bary) * err
                    )
                    mismatch = up(mismatch)
                    map_j[i] += span * mismatch
                    map_b[i] += span * back * mismatch
    loss0 = h * sum(one_integral(z) for z in dn)
    loss = Interval(max(Q(0), loss0 - loss_err), loss0 + loss_err)

    def energies(pos):
        xx = [pos[i : i + 3] for i in range(0, 12, 3)]
        ee = [[Interval.point(a - b) for a, b in zip(xx[i], xx[0], strict=True)] for i in (1, 2, 3)]
        _, em = material_box(*ee, model.rest)
        layer = Interval.point(0)
        point_gaps = []
        for ids in faces:
            u = [Interval.point(a - b) for a, b in zip(xx[ids[1]], xx[ids[0]], strict=True)]
            vv = [Interval.point(a - b) for a, b in zip(xx[ids[2]], xx[ids[0]], strict=True)]
            c = cross(u, vv)
            ss = (c[0] / c[2]).square() + (c[1] / c[2]).square()
            rt = sqrt_interval(1 + ss, 96)
            tilt = rho * ss / (rt * (1 + rt))
            for node in ids:
                gap = xx[node][2] - pos[-1] + tilt
                layer += area * stiffness * (width - gap).square() / 2
                point_gaps.append(gap)
        return em, layer, point_gaps

    m0, l0, _ = energies(q0)
    m1, l1, endgap = energies(q1)
    kinetic = sum(m * (b * b - a * a) / 2 for m, b, a in zip(mass, v1, v0, strict=True))
    original_work = sum(f * (b - a) for f, b, a in zip(ext, q1, q0, strict=True))
    energy = kinetic + (m1 - m0) + (l1 - l0) + loss - original_work
    moment = [Q(0)] * 3
    original_moment = [Q(0)] * 3
    external_int = [Q(0)] * 3
    for i in range(4):
        left = q0[3 * i : 3 * i + 3]
        right = q1[3 * i : 3 * i + 3]
        mi = mass[3 * i]
        eb = ext[3 * i : 3 * i + 3]
        for j, delta in enumerate(
            subs_cross(
                right,
                [mi * z for z in v1[3 * i : 3 * i + 3]],
                left,
                [mi * z for z in v0[3 * i : 3 * i + 3]],
            )
        ):
            moment[j] += delta
            original_moment[j] += delta - h * cross(left, eb)[j]
        for j, p in enumerate(cross_poly(x[i], [(f,) for f in eb])):
            external_int[j] += h * one_integral(p)
    for a, node in enumerate(faces.ravel()):
        for k, p in enumerate(cross_poly(x[node], [ZERO, ZERO, pressure[a]])):
            value = h * one_integral(p)
            moment[k] -= value
            original_moment[k] -= value
    ang = sum(abs(m - e) for m, e in zip(moment, external_int, strict=True)) + angle_arm
    old_ang = sum(map(abs, original_moment)) + angle_arm
    lin = [Q(0)] * 3
    for i in range(13):
        dim = i % 3 if i < 12 else 2
        lin[dim] += mass[i] * (v1[i] - v0[i]) - h * ext[i]
    vel_error = max(3 * (a + b) / m for a, b, m in zip(mat_j, map_j, mass, strict=True))
    pos_error = max(3 * (a + b) / m for a, b, m in zip(mat_b, map_b, mass, strict=True))
    actual_ref_error = None
    ref_velocity_error = None
    if reference is not None:
        actual_ref_error = max(
            max(abs(g.low - Q(float(v))), abs(g.high - Q(float(v))))
            for g, v in zip(endgap, reference["final_gaps_m"], strict=True)
        )
        ref_velocity_error = model.endpoint_rate_error(q1, v1, reference["final_rates_m_s"])
    checks = dict(
        nonnegative_pressure=True,
        hard_path=True,
        constitutive_impulse=max(JE) <= Q("1e-10"),
        constitutive_time_moment=max(BE) / min(mass) <= Q("1e-10"),
        force_position_defect=pos_error <= Q("1e-8"),
        force_velocity_defect=vel_error <= Q("2e-6"),
        original_energy=energy.high <= Q("1e-8"),
        paired_angular=max(ang, old_ang) <= Q("1e-8"),
        linear=sum(map(abs, lin)) <= Q("1e-10"),
        reference_gap=actual_ref_error is None or actual_ref_error <= Q("1e-8"),
        reference_speed=ref_velocity_error is None or ref_velocity_error <= 2e-6,
    )
    return dict(
        accepted=all(checks.values()),
        checks=checks,
        proof_leaves=LEAVES,
        owner_proof_enclosures=6 * LEAVES,
        minimum_oriented_area_z=float(min_orient),
        patch_margin_m=float(min_margin),
        minimum_pressure_n=float(min_pressure),
        minimum_active_factor=float(min_rate_factor),
        hard_witness=[
            dict(factor_at_inlet=k, minimum_factor_m=float(min(iv.low for iv in boxes)))
            for k, boxes in gfac
        ],
        j_ns=list(map(float, Js)),
        b_ns2=list(map(float, Bs)),
        j_defect_upper_ns=list(map(float, JE)),
        b_defect_upper_per_mass_m=[float(v / min(mass)) for v in BE],
        material_velocity_defect_upper_m_s=float(
            max(3 * a / m for a, m in zip(mat_j, mass, strict=True))
        ),
        material_position_defect_upper_m=float(
            max(3 * a / m for a, m in zip(mat_b, mass, strict=True))
        ),
        node_force_velocity_defect_upper_m_s=float(vel_error),
        node_force_position_defect_upper_m=float(pos_error),
        material_moments=node_moments,
        original_work_j=float(original_work),
        kinetic_change_j=float(kinetic),
        material_energy_j=[outward(m0), outward(m1)],
        layer_energy_j=[outward(l0), outward(l1)],
        dissipation_j=outward(loss),
        signed_energy_defect_j=outward(energy),
        paired_angular_upper_nms=float(ang),
        initial_convention_angular_upper_nms=float(old_ang),
        net_momentum_residual_ns=list(map(float, lin)),
        endpoint_gaps_m=[outward(g) for g in endgap],
        reference_gap_error_m=None if actual_ref_error is None else float(actual_ref_error),
        reference_speed_error_m_s=ref_velocity_error,
    )


def subs_cross(a, b, c, d):
    return [x - y for x, y in zip(cross(a, b), cross(c, d), strict=True)]
