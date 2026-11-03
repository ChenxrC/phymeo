"""Material sections and closest boundary witnesses of a closed rigid mesh and a filled triangle."""

from dataclasses import dataclass,replace
from fractions import Fraction as F
from itertools import combinations
from math import ulp

import numpy as np

from ..core.capabilities import SolverFailure
from ._closed_mesh import _Facet
from ._mesh_mesh import _intersection
from ._prism_clip import edges,hull,inside
from ._round_sweep import _translation,_unit,_vector,_xyz
from ._tetrahedral_surface import _TriangleMotion
from ._triangle_pair_sweep import _FacePath,_TrianglePair
from .collision_witness import CollisionSweep,CollisionWitness

_ZERO=F(0)
_ONE=F(1)
_ORIGIN=_vector((0,0,0))
_MAX_SECTION_CELLS=4096


@dataclass(frozen=True)
class MeshTrianglePatch:
    mesh_face: tuple[int,int,int]
    dimension: int
    points: tuple[CollisionWitness,...]


@dataclass(frozen=True)
class MeshTrianglePose:
    relation: str
    boundary_gap_m: float
    inside_fraction: float
    boundary_fraction: float
    outside_fraction: float
    boundary: tuple[MeshTrianglePatch,...]
    closest: tuple[MeshTrianglePatch,...]
    witness: CollisionWitness


@dataclass(frozen=True)
class MeshTriangleEvent:
    interval: tuple[float,float]
    witness_time: float
    pose: MeshTrianglePose


@dataclass(frozen=True)
class MeshTriangleTrace:
    start: MeshTrianglePose
    end: MeshTrianglePose
    first_contact: MeshTriangleEvent|None


def _cuts(segments,vertices):
    xs={_ZERO,_ONE,*(p[0] for p in vertices)}
    # Proper crossings can split vertical order. Collinear intervals already
    # contribute all endpoints and an isolated point contributes its own x.
    for ((p,q),(r,s)) in combinations(segments,2):
        if max(p[0],r[0])>=min(q[0],s[0]):
            continue
        dx,dy=q[0]-p[0],q[1]-p[1]
        ex,ey=s[0]-r[0],s[1]-r[1]
        det=dx*ey-dy*ex
        if not det:
            continue
        wx,wy=r[0]-p[0],r[1]-p[1]
        t=(wx*ey-wy*ex)/det
        u=(wx*dy-wy*dx)/det
        if 0<t<1 and 0<u<1:
            xs.add(p[0]+t*dx)
    return sorted(xs)


def _fractions(mesh,triangle,polygons):
    vertices={p for polygon in polygons for p in polygon}
    segments={tuple(sorted((p,q))) for polygon in polygons for p,q in edges(polygon)}
    coplanar=[poly for poly in polygons if len(poly)>2]
    xs=_cuts(segments,vertices)
    contained=on_surface=_ZERO
    tested=0
    for left,right in zip(xs,xs[1:]):
        x=(left+right)/2
        ys={_ZERO,1-x}
        for p,q in segments:
            if p[0]<x<q[0]:
                ys.add(p[1]+(x-p[0])*(q[1]-p[1])/(q[0]-p[0]))
        ys=sorted(ys)
        for low,high in zip(ys,ys[1:]):
            tested+=1
            if tested>_MAX_SECTION_CELLS:
                raise SolverFailure("mesh_triangle_section_capacity")
            y=(low+high)/2
            # Bounds are linear on each strip. Midpoint width gives its exact
            # area; factor two converts to a fraction of the reference triangle.
            fraction=2*(right-left)*(high-low)
            if any(inside((x,y),polygon) for polygon in coplanar):
                on_surface+=fraction
            elif mesh.inside((1-x-y)*triangle[0]+x*triangle[1]+y*triangle[2]):
                contained+=fraction
    return contained,on_surface,1-contained-on_surface


class MeshTriangleQuery:
    features=()

    def __init__(self,mesh,cloth,motion,mesh_first):
        self.mesh,self.cloth,self.mesh_first=mesh,cloth,mesh_first
        a=np.array([_vector(p) for p in motion.start])
        if motion.end is not None:
            b=np.array([_vector(p) for p in motion.end])
        else:
            b=a+_translation(motion)
        b=b-mesh.step
        self.motion=_TriangleMotion(a,b,not np.array_equal(a,b))
        self.path=_FacePath(cloth,self.motion)

    def _witness(self,facet,cloth,wm,wc,shift,normal,gap):
        pm=_xyz(sum(F(w)*facet.points[i] for i,w in wm)+shift)
        pc=_xyz(sum(F(w)*cloth.points[i] for i,w in wc)+shift)
        if any(ulp(x)>1e-7 for x in (*pm,*pc)):
            raise SolverFailure("mesh_triangle_world_resolution")
        wm=tuple(sorted((i,float(w)) for i,w in wm if w))
        wc=tuple(sorted((i,float(w)) for i,w in wc if w))
        if self.mesh_first:
            return CollisionWitness(gap,normal,pm,pc,wm,wc,"host-closed-mesh-triangle")
        normal=None if normal is None else tuple(-v for v in normal)
        return CollisionWitness(gap,normal,pc,pm,wc,wm,"host-closed-mesh-triangle")

    def _closest(self,cloth,shift,is_inside):
        order=[]
        for i,f in enumerate(self.mesh.facets):
            delta=np.maximum(0,np.maximum(f.lo-cloth.hi,cloth.lo-f.hi))
            order.append((delta@delta,i))
        best=None
        patches=[]
        for lo,i in sorted(order):
            if best is not None and lo>best:
                break
            f=self.mesh.facets[i]
            leaf=_TrianglePair(f.shape,self.cloth,f.stationary,cloth.stationary,sweep=False,proximity=True)
            w,d2=leaf.sample(_ZERO)
            if best is not None and d2>best:
                continue
            if best is None or d2<best:
                best,patches=d2,[]
            patch=w.proximity_patch
            gap,n=w.gap_m,w.normal
            if is_inside:
                gap=-gap
                n=tuple(-x for x in n)
            points=tuple(self._witness(f,cloth,p.first_weights,p.second_weights,shift,n,gap) for p in patch.points)
            patches.append(MeshTrianglePatch(f.ids,patch.dimension,points))
        if not patches:
            raise SolverFailure("mesh_triangle_nearest_unresolved")
        return tuple(sorted(patches,key=lambda p:p.mesh_face))

    def sample(self,t):
        if not self.path.valid:
            raise SolverFailure("triangle_degenerate_on_path")
        xyz=self.path.start+t*self.path.step
        ids=self.cloth.vertex_ids
        cloth=_Facet(self.cloth,dict(zip(ids,xyz,strict=True)),ids,_ORIGIN)
        shift=t*self.mesh.step
        raw=[]
        polygons=[]
        for f in self.mesh.facets:
            if any(f.hi<cloth.lo) or any(cloth.hi<f.lo):
                continue
            pts,area,_=_intersection(f,cloth,_ZERO)
            if not len(pts):
                continue
            entries=[]
            domain=[]
            for p in pts:
                wm,wc=f.weights(p),cloth.weights(p)
                if wm is None or wc is None:
                    raise SolverFailure("mesh_triangle_boundary_material_unresolved")
                entries.append((wm,wc))
                wc=dict(wc)
                domain.append((wc.get(ids[1],_ZERO),wc.get(ids[2],_ZERO)))
            raw.append((f,2 if area else min(1,len(pts)-1),entries))
            polygons.append(hull(domain))
        inside_area,boundary_area,outside_area=_fractions(self.mesh,xyz,polygons)
        if min(inside_area,boundary_area,outside_area)<0:
            raise SolverFailure("mesh_triangle_area_unresolved")
        if not raw:
            relation="contained" if inside_area else "separated"
            boundary=()
            closest=self._closest(cloth,shift,bool(inside_area))
            patches=closest
        else:
            relation=("crossing" if inside_area and outside_area else
                      "contained_touching" if inside_area else "touching")
            pieces=[]
            for f,dimension,addresses in raw:
                normal=_xyz(_unit(f.face.normal)[0]) if relation=="touching" else None
                points=tuple(self._witness(f,cloth,wm,wc,shift,normal,0.0) for wm,wc in addresses)
                pieces.append(MeshTrianglePatch(f.ids,dimension,points))
            boundary=tuple(pieces)
            closest=()
            patches=boundary
        witness=patches[0].points[0]
        if len({p.normal for patch in patches for p in patch.points})!=1:
            witness=replace(witness,normal=None)
        return MeshTrianglePose(relation,abs(witness.gap_m),float(inside_area),float(boundary_area),
                                float(outside_area),boundary,closest,witness)

    def result(self,moving):
        if moving:
            return "unsupported","mesh_triangle_motion",None,None,None
        pose=self.sample(_ZERO)
        status=pose.relation if pose.relation in ("separated","touching") else "overlap"
        return status,pose.relation,pose.witness,None,MeshTriangleTrace(pose,pose,None)
