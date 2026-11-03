"""First intersection of the actual boundaries of two closed meshes on one relative rotation axis."""

from dataclasses import dataclass
from fractions import Fraction as F
from math import ceil, floor, ulp
from numbers import Integral

import numpy as np

from ..core.capabilities import SolverFailure
from ._closed_mesh import _Facet
from ._mesh_mesh import _intersection
from ._rotation_angle import _PI_LO, _AngleChart, axis_length_bounds, rotate_enclosures
from ._round_sweep import _GRID, _translation, _unit, _vector, _xyz
from ._sweep_polynomial import Roots, _gcd, add, dot, first_zero_entry, mul, sub, value
from ._triangle_pair_sweep import _EDGES, _TrianglePair, _cross3, _edge_entries
from .collision_witness import CollisionWitness
from .rotating_mesh import MeshArcFacePair, mesh_arc_face_candidates

_MAX_PAIR_CHARTS = 512
_SPATIAL_ERROR_M = F(1, 10**6)


@dataclass(frozen=True)
class MeshArcSupport:
    faces: MeshArcFacePair
    witness: CollisionWitness
    first_outward_normal: tuple[float, float, float]
    second_outward_normal: tuple[float, float, float]


@dataclass(frozen=True)
class MeshArcBoundaryEvent:
    interval: tuple[float, float]
    witness_time: float
    supports: tuple[MeshArcSupport, ...]


@dataclass(frozen=True)
class MeshArcSurfaceResult:
    status: str
    reason: str
    event: MeshArcBoundaryEvent | None = None


def _relative(shapes, motions):
    """Bring a common translational drift and, when present, a common rotation into the frame."""
    if any(_translation(motions[0]) != _translation(motions[1])):
        return None
    turning = [i for i,path in enumerate(motions) if path.turn and path.turn.angle]
    if not turning:
        return 0, _vector((0,0,0)), _vector((0,0,1)), F(0)
    # Reordering the two arguments keeps the same relative chart and tie order.
    m = min(turning, key=lambda i: shapes[i].key)
    turn = motions[m].turn
    pivot = _vector(turn.pivot_start)
    axis = _vector(turn.axis)
    angle = F(turn.angle)
    if len(turning)==2:
        fixed = motions[1-m].turn
        other_axis = _vector(fixed.axis)
        if any(_cross3(axis, other_axis)) or any(_cross3(axis, _vector(fixed.pivot_start)-pivot)):
            return None
        angle -= F(fixed.angle) if axis @ other_axis > 0 else -F(fixed.angle)
    return m, pivot, axis, angle


def _constants(v):
    return tuple((x,) for x in v)


def _scaled(point, den):
    return tuple(mul((x,), den) for x in point)


class _HalfRotation:
    """Homogeneous coordinates for a signed half-turn with a positive common denominator."""

    def __init__(self, axis):
        self.axis = axis
        w, s = (F(1), F(-1)), (F(0), F(1))
        self.ss = mul(s,s)
        self.twows = mul((2,),mul(w,s))
        ww, aa = mul(w,w), mul((axis@axis,),self.ss)
        self.den, self.diagonal = add(ww,aa),sub(ww,aa)

    def vertex(self, p, *, inverse=False):
        axial = 2*self.axis*(self.axis@p)
        cross = _cross3(self.axis,p)
        if inverse:
            cross = -cross
        return tuple(add(add(mul((x,),self.diagonal),mul((a,),self.ss)),mul((c,),self.twows))
                     for x,a,c in zip(p,axial,cross,strict=True))


def _at_face(vertex, den, face):
    a,b,c = face
    u,v = b-a,c-a
    n = _cross3(u,v)
    relative = tuple(sub(x,y) for x,y in zip(vertex,_scaled(a,den),strict=True))
    height = dot(relative,_constants(n))
    w1 = dot(relative,_constants(_cross3(v,n)))
    w2 = dot(relative,_constants(_cross3(n,u)))
    return height, (sub(sub(mul((n@n,),den),w1),w2),w1,w2)


def _root_feature(equation, limits):
    bounds = first_zero_entry(equation,limits,None)
    return None if bounds is None else (bounds, equation, limits)


def _closed_count(poly, a, b):
    if not poly:
        return 0
    return int(value(poly,a)==0) + Roots(poly).count(a,b)


def _same_root(features):
    """A coarse cell may contain several real roots. Only publish ties proven to share a root."""
    (u,v) = features[0][0]
    if u==v:
        return
    lo,hi = F(u,_GRID), F(v,_GRID)
    common = None
    for _,equation,limits in features:
        defining = (equation,) if equation else limits
        product = (F(1),)
        for p in defining:
            if _closed_count(p,lo,hi):
                product = mul(product,p)
        if _closed_count(product,lo,hi) != 1:
            raise SolverFailure("mesh_arc_feature_root_ambiguity")
        squarefree = Roots(product).chain[0]
        common = squarefree if common is None else _gcd(common,squarefree)
        if len(common)<2 or not _closed_count(common,lo,hi):
            raise SolverFailure("mesh_arc_competing_first_roots")


def _chart_first(moving, stationary, face_pairs, half):
    """Reuse exact vertex/face and projected edge/edge material gates on rational polynomials."""
    forward = {i:half.vertex(p) for i,p in moving.items()}
    inverse = {i:half.vertex(p,inverse=True) for i,p in stationary.items()}
    fixed = {i:_scaled(p,half.den) for i,p in stationary.items()}
    vf,ee = {},{}
    active = []
    earliest = None
    for face_ids, m, s in face_pairs:
        features = []
        for side,source,face,points,coords in (
            (0,m,s,forward,stationary), (1,s,m,inverse,moving)
        ):
            for node in source:
                key = side,node,face
                if key not in vf:
                    eq,limits = _at_face(points[node],half.den,[coords[k] for k in face])
                    vf[key] = _root_feature(eq,limits)
                if vf[key] is not None:
                    features.append(vf[key])
        for i,j in _EDGES:
            e = tuple(sorted((m[i],m[j])))
            for k,l in _EDGES:
                h = tuple(sorted((s[k],s[l])))
                if (e,h) not in ee:
                    items = _edge_entries(forward[e[0]],forward[e[1]],fixed[h[0]],fixed[h[1]])
                    ee[e,h] = [root for f in items if (root:=_root_feature(f.equation,f.limits)) is not None]
                features.extend(ee[e,h])
        for feature in features:
            bounds=feature[0]
            if earliest is None or bounds<earliest:
                earliest,active=bounds,[]
            if bounds==earliest:
                active.append((face_ids,feature))
    if earliest is None:
        return None
    _same_root([f for _,f in active])
    return earliest, tuple(sorted({pair for pair,_ in active},key=lambda x:(x.first,x.second)))


def _face_poses(shapes, motions, pairs, time):
    sides=[]
    for side,(shape,path) in enumerate(zip(shapes,motions,strict=True)):
        faces = {p.first if side==0 else p.second for p in pairs}
        needed = {node for f in faces for node in f}
        points={i:_vector(p) for i,p in zip(shape.vertex_ids,path.start,strict=True) if i in needed}
        drift=time*_translation(path)
        error=F(0)
        if time and path.turn and path.turn.angle:
            pivot=_vector(path.turn.pivot_start)
            axis=_vector(path.turn.axis)
            rotated=rotate_enclosures([p-pivot for p in points.values()],axis,axis_length_bounds(axis),F(path.turn.angle)*time)
            points={i:p+pivot+drift for i,(p,_) in zip(points,rotated,strict=True)}
            error=max(e for _,e in rotated)
        else:
            points={i:p+drift for i,p in points.items()}
        facets={f:_Facet(shape,points,f,_vector((0,0,0))) for f in faces}
        sides.append((facets,error))
    return sides


def _event(shapes,motions,pairs,bracket,*,at_start=False):
    if not pairs:
        return None
    time = (bracket[0]+bracket[1])/2
    (first,ea),(second,eb) = _face_poses(shapes,motions,pairs,F(time))
    supports=[]
    for pair in pairs:
        a,b=first[pair.first],second[pair.second]
        intersections,_,_=_intersection(a,b,F(0))
        if intersections:
            center=sum(intersections)/len(intersections)
            wa,wb=a.weights(center),b.weights(center)
        elif at_start:
            continue
        else:
            # Canonical ordering also selects the same material pair after owner exchange.
            reverse = a.shape.key>b.shape.key
            l,r=(b,a) if reverse else (a,b)
            w,_=_TrianglePair(l.shape,r.shape,l.stationary,r.stationary,sweep=False).sample(F(0))
            wa,wb=(w.second_weights,w.first_weights) if reverse else (w.first_weights,w.second_weights)
        if wa is None or wb is None:
            raise SolverFailure("mesh_arc_surface_address_unresolved")
        wa=tuple((i,float(w)) for i,w in wa)
        wb=tuple((i,float(w)) for i,w in wb)
        pa=sum(F(w)*a.points[i] for i,w in wa)
        pb=sum(F(w)*b.points[i] for i,w in wb)
        direction,gap=_unit(pb-pa)
        xyz_a,xyz_b=_xyz(pa),_xyz(pb)
        rounding=sum(ulp(x) for x in (*xyz_a,*xyz_b))
        if ea+eb+F(rounding)+F(gap)>_SPATIAL_ERROR_M:
            raise SolverFailure("mesh_arc_spatial_witness_resolution")
        # A normal across sub-resolution gaps need not be the normal at a non-smooth touch.
        n=None if direction is None or F(gap)<=ea+eb+F(rounding) else _xyz(direction)
        w=CollisionWitness(gap,n,xyz_a,xyz_b,wa,wb,"host-mesh-surface-angle-chart")
        supports.append(MeshArcSupport(pair,w,_xyz(_unit(a.face.normal)[0]),_xyz(_unit(b.face.normal)[0])))
    if not supports:
        return None
    return MeshArcBoundaryEvent(bracket,time,tuple(supports))


def mesh_arc_first_surface(first,second,first_motion,second_motion,*,max_pair_charts=_MAX_PAIR_CHARTS):
    """First actual surface intersection on two coaxial rigid motions with equal linear pivot drift.

    A single rotor can face a static or equally translating mesh. With two rotors the
    axes must be the same world line. The returned sample lies in the certified first
    time interval; both labelled material points are evaluated at that exact sample.
    A surface_clear result concerns the boundaries, including when one closed solid
    remains strictly inside the other. All contacts and root calculations run on HOST.
    """
    if not isinstance(max_pair_charts,Integral) or isinstance(max_pair_charts,(bool,np.bool_)) or not 0<=max_pair_charts<=_MAX_PAIR_CHARTS:
        raise ValueError(f"max_pair_charts must be an integer in [0,{_MAX_PAIR_CHARTS}]")
    shapes,motions = (first,second),(first_motion,second_motion)
    if first.key.body==second.key.body:
        return MeshArcSurfaceResult("unsupported","mesh_arc_requires_independent_bodies")
    relative=_relative(shapes,motions)
    if relative is None:
        return MeshArcSurfaceResult("unsupported","mesh_arc_requires_one_relative_axis")
    m,pivot,axis,angle=relative
    try:
        candidates=mesh_arc_face_candidates(*shapes,*motions)
        if not candidates.candidate_count:
            return MeshArcSurfaceResult("surface_clear","no_swept_surface_face_pairs")
        start_pairs=candidates.windows[0].pairs
        if len(start_pairs)>max_pair_charts:
            raise SolverFailure("mesh_arc_pair_chart_capacity")
        event=_event(shapes,motions,start_pairs,(0.0,0.0),at_start=True)
        if event:
            return MeshArcSurfaceResult("hit","initial_surface_intersection",event)
        if not angle:
            return MeshArcSurfaceResult("surface_clear","fixed_relative_boundaries")
        if angle<0:
            axis,angle=-axis,-angle
        mesh_points=[{i:_vector(p)-pivot for i,p in zip(s.vertex_ids,path.start,strict=True)}
                     for s,path in zip(shapes,motions,strict=True)]
        moving,stationary=mesh_points[m],mesh_points[1-m]
        half=_HalfRotation(axis)
        length=axis_length_bounds(axis)
        used=0
        for turn in range(ceil(angle/_PI_LO)):
            chart=_AngleChart(turn,length,angle)
            t0,t1=chart.time(0,False),chart.time(_GRID,True)
            pairs={p for w in candidates.windows if w.interval[0]<=t1 and t0<=w.interval[1] for p in w.pairs}
            used+=len(pairs)
            if used>max_pair_charts:
                raise SolverFailure("mesh_arc_pair_chart_capacity")
            rows=[(p,p.first,p.second) if m==0 else (p,p.second,p.first) for p in pairs]
            hit=_chart_first(moving,stationary,rows,half)
            if hit is not None:
                bounds,incident=hit
                lo,hi=chart.time(bounds[0],False),chart.time(bounds[1],True)
                if lo>1:
                    break
                if hi>1 or hi-lo>F(1,10**7):
                    raise SolverFailure("mesh_arc_time_resolution")
                bracket=floor(lo*_GRID)/_GRID,ceil(hi*_GRID)/_GRID
                event=_event(shapes,motions,incident,bracket)
                return MeshArcSurfaceResult("hit","mesh_surface_first_intersection",event)
            moving={i:2*axis*((axis@v)/(axis@axis))-v for i,v in moving.items()}
        return MeshArcSurfaceResult("surface_clear","certified_surface_sweep")
    except (SolverFailure,OverflowError,FloatingPointError) as exc:
        return MeshArcSurfaceResult("unresolved",str(exc))
