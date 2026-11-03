"""Exact dyadic triangle distances and a symbolically perturbed scanline sign for regular nodes."""

from bisect import bisect_right
from fractions import Fraction as F
from functools import lru_cache
from math import inf,isfinite,nextafter,prod

import numpy as np

from ..constraints.collision_shapes import CollisionMotion,CollisionShape
from ._closed_mesh import ClosedMesh
from ._prism_pair_exact import sqrt_bounds
from ._sheet_pair_bounds import _cross,_dot,_sub

_MAX_FACE_NODES=8_000_000


class _BakeFace:
    def __init__(self,points):
        a,b,c=points
        self.a=a
        u,v=_sub(b,a),_sub(c,a)
        self.n=_cross(u,v)
        self.den=_dot(self.n,self.n)
        self.plane=_dot(self.n,a)
        self.w1,self.w2=_cross(v,self.n),_cross(self.n,u)
        self.lo=tuple(map(min,zip(*points,strict=True)))
        self.hi=tuple(map(max,zip(*points,strict=True)))
        self.edges=[]
        for p,q in ((a,b),(b,c),(c,a)):
            d=_sub(q,p)
            self.edges.append((p,q,d,_dot(d,d)))

    def distance2(self,p):
        u=_sub(p,self.a)
        w1,w2=_dot(u,self.w1),_dot(u,self.w2)
        if w1>=0 and w2>=0 and w1+w2<=self.den:
            h=_dot(u,self.n)
            return h*h,self.den
        best_n,best_d=None,1
        for a,b,d,dd in self.edges:
            v=_sub(p,a)
            t=_dot(v,d)
            if t<=0:
                n,ds=_dot(v,v),1
            elif t>=dd:
                w=_sub(p,b)
                n,ds=_dot(w,w),1
            else:
                n,ds=_dot(v,v)*dd-t*t,dd
            if best_n is None or n*best_d<best_n*ds:
                best_n,best_d=n,ds
        return best_n,best_d

    def crossing_x(self,y,z):
        # Cast along +X at (y+epsilon,z+epsilon²). A shared edge has one
        # owner, and rays through projected silhouette edges have even parity.
        # If a grid node coincides with a crossing, its geometric distance
        # is zero and its sign is handled without relying on this ray.
        nx,ny,nz=self.n
        if not nx or not(self.lo[1]<=y<=self.hi[1] and self.lo[2]<=z<=self.hi[2]):
            return None
        for p,_,d,_ in self.edges:
            orientation=d[1]*(z-p[2])-d[2]*(y-p[1])
            leading=orientation or -d[2] or d[1]
            if leading*nx<=0:
                return None
        return F(self.plane-ny*y-nz*z,nx)


def bake_nodes(shape,vertices,shape_xyz,origin,spacing):
    if not isinstance(shape,CollisionShape) or shape.kind!="mesh" or shape.radius:
        raise ValueError("SDF baking requires an unrounded closed mesh")
    size=prod(shape_xyz)
    work=size*len(shape.triangles)
    if work>_MAX_FACE_NODES:
        raise ValueError(f"SDF bake capacity exceeded: {work} face-node pairs")
    motion=CollisionMotion(vertices)
    if len(motion.start)!=len(shape.vertex_ids):
        raise ValueError("baked mesh requires one local position per material vertex")
    mesh=ClosedMesh(shape,motion)
    origin=tuple(map(F,origin))
    spacing=tuple(map(F,spacing))
    scale=max(x.denominator for p in (*mesh.points.values(),origin,spacing) for x in p)
    def integer(v):
        return tuple(x.numerator*(scale//x.denominator) for x in v)
    points={i:integer(p) for i,p in mesh.points.items()}
    faces=[_BakeFace(tuple(points[i] for i in f.ids)) for f in mesh.facets]
    start,step=integer(origin),integer(spacing)
    nodes=[range(o,o+n*h,h) for o,n,h in zip(start,shape_xyz,step,strict=True)]
    values=np.empty(shape_xyz[::-1],dtype=np.float64)
    error=F(0)
    tested=0
    @lru_cache(maxsize=4096)
    def length(n,d):
        lo,hi=sqrt_bounds(F(n,d*scale*scale),96)
        mid=float((lo+hi)/2)
        if not isfinite(mid):
            raise ValueError("baked signed distance outside FP64")
        return mid,max(abs(F(mid)-lo),abs(hi-F(mid)))
    for iz,z in enumerate(nodes[2]):
        for iy,y in enumerate(nodes[1]):
            hits=sorted(h for f in faces if (h:=f.crossing_x(y,z)) is not None)
            for ix,x in enumerate(nodes[0]):
                p=x,y,z
                best_n,best_d=None,1
                for f in faces:
                    if best_n is not None:
                        dx=max(0,f.lo[0]-x,x-f.hi[0])
                        dy=max(0,f.lo[1]-y,y-f.hi[1])
                        dz=max(0,f.lo[2]-z,z-f.hi[2])
                        if (dx*dx+dy*dy+dz*dz)*best_d>=best_n:
                            continue
                    n,d=f.distance2(p)
                    tested+=1
                    if best_n is None or n*best_d<best_n*d:
                        best_n,best_d=n,d
                distance,bound=length(best_n,best_d)
                error=max(error,bound)
                odd=(len(hits)-bisect_right(hits,x))%2
                values[iz,iy,ix]=-distance if distance and odd else distance
    published_error=float(error)
    if F(published_error)<error:
        published_error=nextafter(published_error,inf)
    stats=dict(validated_meshes=1,face_count=len(faces),node_count=size,
               exact_face_distance_tests=tested,scanlines=shape_xyz[1]*shape_xyz[2],
               node_roundoff_bound_m=published_error)
    return values,error,stats
