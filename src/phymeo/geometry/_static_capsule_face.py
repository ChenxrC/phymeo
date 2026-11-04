"""Exact closest material interval of a closed segment and a filled triangle at one pose."""

from fractions import Fraction as F
from typing import NamedTuple

import numpy as np

from ..core.capabilities import SolverFailure
from ._capsule_triangle_sweep import capsule_face_query, segment_witness
from ._triangle_pair_sweep import _EDGES, _on_edge


class _MaterialTriangle(NamedTuple):
    ids: tuple
    p: np.ndarray


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _unit_parameter(num, den):
    return 0 if num <= 0 else 1 if num >= den else F(num, den)


def _plane_minimum(start, finish, points):
    a, b, c = points
    u, v = _sub(b, a), _sub(c, a)
    normal = _cross(u, v)
    den = _dot(normal, normal)
    if not den:
        raise SolverFailure("triangle_degenerate_on_path")
    uu, uv, vv = _dot(u,u), _dot(u,v), _dot(v,v)

    def project(point):
        rel = _sub(point, a)
        x, y = _dot(rel, u), _dot(rel, v)
        j, k = vv*x - uv*y, uu*y - uv*x
        return _dot(normal, rel), (den-j-k, j, k)

    h0, w0 = project(start)
    h1, w1 = project(finish)
    if h0 == h1:
        # Parallel: exactly clip the entire closed projected shaft by each
        # barycentric half-space. A singleton intersection is a valid tie.
        low, high = 0, 1
        for left, right in zip(w0, w1, strict=True):
            if left < 0 and right < 0:
                return None
            if left < 0:
                low = max(low, F(-left, right-left))
            elif right < 0:
                high = min(high, F(left, left-right))
        if low > high:
            return None
    else:
        # Unique minimum of the squared affine plane height on [0,1].
        step = h0 - h1
        low = high = _unit_parameter(h0 if step > 0 else -h0, abs(step))

    def material(s):
        num, denom = s.numerator, s.denominator
        return tuple(F(left*(denom-num)+right*num, den*denom) for left, right in zip(w0, w1, strict=True))

    first = material(low)
    if min(first) < 0:
        return None
    last = first if low == high else material(high)
    h_num = h0*(low.denominator-low.numerator) + h1*low.numerator
    return F(h_num*h_num, den*low.denominator**2), (low,first), (high,last)


def _edge_parameters(uu, uv, vv, uw, vw):
    # Extremes on all four box sides, then the unique interior stationary point.
    # When the directions are parallel the side minima include both ends of any tie.
    yield 0, _unit_parameter(vw, vv)
    if not uu:
        return
    yield 1, _unit_parameter(vw + uv, vv)
    yield _unit_parameter(-uw, uu), 0
    yield _unit_parameter(uv - uw, uu), 1
    det = uu*vv - uv*uv
    sn, tn = uv*vw - vv*uw, uu*vw - uv*uw
    if det and 0 <= sn <= det and 0 <= tn <= det:
        yield F(sn, det), F(tn, det)


def _boundary_minimum(start, axis, points):
    uu = _dot(axis, axis)
    best_num, best_den = None, 1
    low = high = None
    for i, j in _EDGES:
        edge = _sub(points[j], points[i])
        delta = _sub(start, points[i])
        uv, vv = _dot(axis,edge), _dot(edge,edge)
        uw, vw = _dot(axis,delta), _dot(edge,delta)
        for s, t in _edge_parameters(uu,uv,vv,uw,vw):
            sd, td = s.denominator, t.denominator
            common = sd*td
            sn, tn = s.numerator*td, t.numerator*sd
            dx = common*delta[0] + sn*axis[0] - tn*edge[0]
            dy = common*delta[1] + sn*axis[1] - tn*edge[1]
            dz = common*delta[2] + sn*axis[2] - tn*edge[2]
            num, den = dx*dx + dy*dy + dz*dz, common*common
            order = -1 if best_num is None else num*best_den - best_num*den
            if order <= 0:
                address = (s, _on_edge(i,j,t))
                if order < 0:
                    best_num, best_den, low, high = num, den, address, address
                else:
                    low, high = min(low,address), max(high,address)
    if not uu:
        low, high = (0, low[1]), (1, low[1])
    return F(best_num,best_den), low, high


def static_capsule_face_sample(a, b, ma, mb):
    """Return the original time-zero witness and exact sign without constructing an affine path.

    A non-static input retains the original whole-path validation and query semantics.
    Motion inputs own validated finite FP64 arrays. No geometry is shared between calls.
    """
    if {a.kind,b.kind} != {"capsule","triangle"}:
        return None
    face_first = a.kind == "triangle"
    cap, face, mc, mf = (b,a,mb,ma) if face_first else (a,b,ma,mb)
    if face.radius:
        return None
    if ma.moving or mb.moving or len(mc.start) != 2 or len(mf.start) != 3:
        query = capsule_face_query(a,b,ma,mb)
        return None if query is None else query.sample(0)

    # FP64 coordinates are dyadic. Keep the feature predicates on one exact
    # integer lattice and do divisions only for candidate material parameters.
    ratios = [tuple(float(v).as_integer_ratio() for v in point) for point in (*mc.start,*mf.start)]
    scale = max(d for row in ratios for _,d in row)
    lattice = [tuple(n*(scale//d) for n,d in row) for row in ratios]
    start, finish = lattice[:2]
    points = lattice[2:]
    axis = _sub(finish,start)
    nearest = _plane_minimum(start,finish,points)
    if nearest is None:
        # An interior face minimum would attain the supporting-plane lower bound;
        # when that bound is unattainable, a minimum lies on the triangle boundary.
        nearest = _boundary_minimum(start,axis,points)
    d2, low, high = nearest
    material = _MaterialTriangle(face.vertex_ids, np.array([[F(x,scale) for x in p] for p in points],dtype=object))
    p = np.array([F(x,scale) for x in start],dtype=object)
    u = np.array([F(x,scale) for x in axis],dtype=object)
    return segment_witness(cap,face_first,p,u,material,(d2/(scale*scale),low,high),"host-affine-capsule-face")
