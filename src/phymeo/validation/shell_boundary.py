"""Independent high-level audit from original inputs. No producer imports/calls."""
from dataclasses import fields, is_dataclass
from fractions import Fraction as F
from itertools import combinations

import numpy as np

from ..constraints.conforming_shell import ConformingShellReference, ShellUnresolved
from ..constraints.shell_boundary import (
    BoundaryContactStratum, BoundaryFanWitness, BoundaryFieldWitness, BoundaryLaw,
    BoundaryPairWitness, BoundaryPlaneEvidence, BoundaryPointBasis, BoundaryProofLimits,
    BoundarySupportEvidence, BoundaryTopology, BoundaryWitnessHint,
    ConvexClusterBoundaryWitness, ExactBoundaryAddress, MatchingIntersectionWitness,
    ShellBoundaryAdmission, ShellBoundarySweep, ShellExactBoundaryPoint, digest,
)
from ..geometry._shell_contact_field import FieldArithmetic, Meter
from ..geometry._shell_intervals import Interval, Publication


def _reject(detail):
    raise ShellUnresolved("invalid_certificate",detail)


def _rotation_cycle(seq):
    seq=tuple(seq)
    return min(seq[i:]+seq[:i] for i in range(len(seq)))


def _sign_order(seq):
    inversions=0
    for i in range(len(seq)):
        for j in range(i):
            inversions+=seq[j]>seq[i]
    return (-1,1)[inversions%2==0]


def _graph(nodes,edges):
    nodes=set(nodes)
    if not nodes: return False
    reach={min(nodes)}
    while True:
        extended=reach | {b for a,b in edges if a in reach} | {a for a,b in edges if b in reach}
        if extended==reach: return reach==nodes
        reach=extended


def _triangle_manifold(triangles,boundary_expected):
    triangles=tuple(triangles)
    vertices=sorted({v for f in triangles for v in f})
    face_keys=[tuple(sorted(f)) for f in triangles]
    if not vertices or len(set(face_keys))!=len(face_keys) or any(len(set(f))!=3 for f in triangles):
        _reject("surface face multiplicity")
    incidence={}
    for tri in triangles:
        for i in range(3):
            edge=(tri[i],tri[(i+1)%3])
            incidence.setdefault(tuple(sorted(edge)),[]).append(edge)
    if not _graph(vertices,incidence): _reject("surface disconnected")
    boundary=[]
    for occurrences in incidence.values():
        if len(occurrences)==1: boundary.append(occurrences[0])
        elif len(occurrences)!=2 or occurrences[0][0]!=occurrences[1][1] or occurrences[0][1]!=occurrences[1][0]:
            _reject("surface edge incidence/orientation")
    if len(vertices)-len(incidence)+len(triangles)!=(1 if boundary_expected else 2):
        _reject("surface Euler number")
    if bool(boundary)!=boundary_expected: _reject("surface boundary type")
    ordered=()
    if boundary:
        if len({a for a,_ in boundary})!=len(boundary) or len({b for _,b in boundary})!=len(boundary):
            _reject("boundary branch")
        successor=dict(boundary)
        walk=[min(successor)]
        for _ in range(len(boundary)-1):
            nxt=successor.get(walk[-1])
            if nxt is None or nxt in walk: _reject("boundary gap/repetition")
            walk.append(nxt)
        if successor.get(walk[-1])!=walk[0]: _reject("incomplete boundary cycle")
        ordered=tuple(walk)
    boundary_nodes={v for edge in boundary for v in edge}
    links=[]
    for v in vertices:
        edges=tuple(tuple(u for u in tri if u!=v) for tri in triangles if v in tri)
        nodes={u for e in edges for u in e}
        degree={u:sum(u in e for e in edges) for u in nodes}
        endpoints=sum(d==1 for d in degree.values())
        if not _graph(nodes,edges): _reject("surface link disconnected")
        if v in boundary_nodes:
            if endpoints!=2 or any(d not in (1,2) for d in degree.values()): _reject("boundary vertex link")
        elif any(d!=2 for d in degree.values()):
            _reject("interior vertex link")
        links.append((v,tuple(sorted(tuple(sorted(e)) for e in edges))))
    return ordered,tuple(links)


def _solid_links(tetrahedra):
    incidences={}
    for tid,vertices in sorted(tetrahedra.items()):
        for slot in range(4):
            f=tuple(vertices[i] for i in range(4) if i!=slot)
            if slot in (1,3): f=f[0],f[2],f[1]
            incidences.setdefault(tuple(sorted(f)),[]).append((_rotation_cycle(f),tid))
    boundary=[]
    adjacency=[]
    for _,rows in sorted(incidences.items()):
        if len(rows)==1: boundary.append(rows[0][0])
        elif len(rows)==2 and _sign_order(rows[0][0])!=_sign_order(rows[1][0]):
            adjacency.append((rows[0][1],rows[1][1]))
        else: _reject("solid face incidence")
    if not _graph(tetrahedra,adjacency): _reject("solid connected interior")
    _,boundary_links=_triangle_manifold(boundary,False)
    exterior={tuple(sorted(s)) for f in boundary for d in (1,2) for s in combinations(f,d)}
    links=[("boundary",boundary_links)]
    vertices=sorted({v for t in tetrahedra.values() for v in t})
    for v in vertices:
        triangles=[]
        for _,t in sorted(tetrahedra.items()):
            if v in t:
                slot=t.index(v)
                f=tuple(x for x in t if x!=v)
                if slot%2: f=f[0],f[2],f[1]
                triangles.append(f)
        cycle,detail=_triangle_manifold(triangles,(v,) in exterior)
        links.append(("vertex",v,cycle,detail))
    edges=sorted({tuple(sorted(e)) for t in tetrahedra.values() for e in combinations(t,2)})
    for e in edges:
        opposite=tuple(tuple(v for v in t if v not in e)
                       for _,t in sorted(tetrahedra.items()) if set(e)<=set(t))
        nodes={v for q in opposite for v in q}
        ds=[sum(v in q for q in opposite) for v in nodes]
        if not _graph(nodes,opposite): _reject("solid edge link disconnected")
        if e in exterior:
            if ds.count(1)!=2 or any(d not in (1,2) for d in ds): _reject("solid boundary edge link")
        elif any(d!=2 for d in ds):
            _reject("solid interior edge link")
        links.append(("edge",e,tuple(sorted(tuple(sorted(q)) for q in opposite))))
    return tuple(boundary),tuple(links)


def _same(claim,expected,meter):
    """Exact values/coverage, independent interval containment, actual proposals."""
    if type(expected) is Interval:
        if type(claim) is not Interval: _reject("interval type")
        meter.interval(claim)
        if meter.qsub(expected.lo,claim.lo)<0 or meter.qsub(claim.hi,expected.hi)<0:
            _reject("claimed interval does not contain independent enclosure")
        return
    if type(expected) is Publication:
        if type(claim) is not Publication or claim.shape!=expected.shape:
            _reject("publication type/shape")
        if (type(claim.values) is not np.ndarray or claim.values.dtype!=np.dtype("float64")
                or claim.values.shape!=expected.shape or claim.values.flags.writeable
                or len(claim.intervals)!=len(expected.intervals)
                or len(claim.errors)!=len(expected.intervals)):
            _reject("publication ownership/coverage")
        for interval,independent,error,stored in zip(claim.intervals,expected.intervals,
                                                     claim.errors,claim.values.flat,strict=True):
            _same(interval,independent,meter)
            meter.check(error)
            midpoint=meter.qdiv(meter.qadd(interval.lo,interval.hi),F(2))
            proposed=meter.input_float(stored)
            if proposed!=meter.input_float(float(midpoint)): _reject("RN64 rational midpoint")
            observed=max(meter.qabs(meter.qsub(proposed,interval.lo)),
                         meter.qabs(meter.qsub(proposed,interval.hi)))
            if meter.qsub(error,observed)<0 or error>=F(1,2**40):
                _reject("actual proposal storage error")
        return
    if type(expected) is np.ndarray:
        if type(claim) is not np.ndarray or claim.dtype!=expected.dtype or claim.shape!=expected.shape or claim.tobytes()!=expected.tobytes():
            _reject("original array bytes")
        return
    if is_dataclass(expected):
        if type(claim) is not type(expected): _reject("record type")
        for f in fields(expected): _same(getattr(claim,f.name),getattr(expected,f.name),meter)
        return
    if type(expected) is tuple:
        if type(claim) is not tuple or len(claim)!=len(expected): _reject("complete tuple coverage")
        for a,b in zip(claim,expected,strict=True): _same(a,b,meter)
        return
    if type(claim) is not type(expected) or claim!=expected:
        _reject("exact evidence mismatch")


class Audit:
    def __init__(self,reference,positions,limits,meter):
        if type(reference) is not ConformingShellReference or type(limits) is not BoundaryProofLimits:
            _reject("reference/policy types")
        limits.__post_init__()
        self.ref,self.limits,self.m=reference,limits,meter
        r=reference
        if type(r.node_ids) is not np.ndarray or type(r.face_ids) is not np.ndarray:
            _reject("original ID arrays")
        if r.node_ids.ndim!=1 or r.face_ids.ndim!=1:
            _reject("original ID ranks")
        n,f=len(r.node_ids),len(r.face_ids)
        cap=limits.strict_limits
        if not 3<=n<=cap.max_nodes or not 1<=f<=cap.max_faces or 3*f>cap.max_tetrahedra:
            _reject("original shape cap")
        for array,dtype,shape in (
            (r.node_ids,"int64",(n,)),(r.face_ids,"int64",(f,)),
            (r.oriented_nodes,"int64",(f,3)),(r.reference_positions,"float64",(n,3)),
            (r.half_thickness_m,"float64",(n,)),(positions,"float64",(n,3))):
            if type(array) is not np.ndarray or array.dtype!=np.dtype(dtype) or array.shape!=shape:
                _reject("original array dtype/shape")
            if not np.isfinite(array).all(): _reject("original finite values")
        if type(r.domain) is not str or not r.domain or len(r.domain.encode())>256:
            _reject("reference domain")
        order=tuple(map(int,r.node_ids))
        self.nodes=tuple(sorted(order))
        fids=tuple(map(int,r.face_ids))
        if len(set(order))!=n or len(set(fids))!=f or min(order)<0 or min(fids)<0:
            _reject("unique original IDs")
        self.faces=tuple(sorted((i,tuple(map(int,v))) for i,v in zip(fids,r.oriented_nodes,strict=True)))
        if len({tuple(sorted(v)) for _,v in self.faces})!=f:
            _reject("unoriented duplicate faces")
        for _,v in self.faces:
            if len(set(v))!=3 or not set(v)<=set(order): _reject("face node identities")
        if r.law.kind!="raw-area-star-director-v1" or r.law.projected_ratio_min!=F(1,4) or r.law.shape_policy!="invariant-1e-10-v1":
            _reject("original director law")
        identity=digest(("conforming-shell-reference-v1",r.domain,
            sorted((i,r.reference_positions[k].tobytes().hex(),r.half_thickness_m[k:k+1].tobytes().hex())
                   for k,i in enumerate(order)),
            sorted((i,_rotation_cycle(v)) for i,v in self.faces),
            r.law.kind,(1,4),r.law.shape_policy))
        if identity!=r.reference_id: _reject("reference identity reconstructed")
        self.stars={i:tuple(fid for fid,face in self.faces if i in face) for i in self.nodes}
        if any(not s or len(s)>cap.max_star_faces for s in self.stars.values()):
            _reject("full reference star bounds")
        unseen=set(self.nodes)
        components=[]
        while unseen:
            component={min(unseen)}
            while True:
                grown=component|{v for _,f in self.faces if set(f)&component for v in f}
                if grown==component: break
                component=grown
            unseen-=component
            components.append(tuple(sorted(component)))
        self.components=tuple(components)
        self.cycles=tuple(_triangle_manifold(tuple(face for _,face in self.faces if face[0] in c),True)[0]
                          for c in self.components)
        self.tets={}
        for fid,face in self.faces:
            i,j,k=sorted(face)
            lifted=((i,-1),(j,-1),(k,-1),(i,1),(j,1),(k,1))
            for stair,indices in enumerate(((0,1,2,5),(0,1,4,5),(0,3,4,5))):
                vertices=tuple(lifted[index] for index in indices)
                if _sign_order(face)*(1,-1,1)[stair]<0:
                    vertices=vertices[0],vertices[1],vertices[3],vertices[2]
                self.tets[fid,stair]=vertices
        if len(self.tets)*(len(self.tets)-1)//2>cap.max_pairs: _reject("all pair budget")
        boundary=[]
        for c in self.components:
            b,_=_solid_links({k:t for k,t in self.tets.items() if t[0][0] in c})
            boundary.extend(b)
        self.boundary=tuple(sorted(boundary))
        self.strata={}
        for tid,t in self.tets.items():
            for count in range(1,5):
                for s in combinations(sorted(t),count):
                    self.strata.setdefault(s,[]).append(tid)
        if len(self.strata)>limits.max_material_strata: _reject("material stratum bound")
        self.boundary_strata={tuple(sorted(s)) for f in self.boundary for count in range(1,4)
                              for s in combinations(f,count)}
        self.x={i:tuple(meter.input_float(v) for v in positions[k]) for k,i in enumerate(order)}
        self.rest={i:tuple(meter.input_float(v) for v in r.reference_positions[k]) for k,i in enumerate(order)}
        self.rho={i:meter.input_float(r.half_thickness_m[k]) for k,i in enumerate(order)}
        if any(q<=0 for q in self.rho.values()): _reject("positive thickness")

    def law(self,x):
        m=self.m
        epsilon=m.input_float(1e-10)
        squared=m.qmul(epsilon,epsilon)
        areas,edges,face_gates={},{},[]
        for fid,v in self.faces:
            p,q,r=(x[i] for i in v)
            ab=m.qvsub(q,p)
            ac=m.qvsub(r,p)
            bc=m.qvsub(r,q)
            normal=m.qcross(ab,ac)
            area=m.qdot(normal,normal)
            if area<=0: _reject("nonzero face area")
            sizes=tuple(m.qdot(e,e) for e in (ab,ac,bc))
            margins=tuple(m.qsub(area,m.qmul(squared,m.qmul(s,s))) for s in sizes)
            if any(v<=0 for v in margins): _reject("face quality")
            areas[fid]=(normal,area)
            edges[fid]=(ab,ac)
            face_gates.append(("face",fid,margins))
        stars={}
        norms={}
        star_gates=[]
        for node in self.nodes:
            value=(F(0),F(0),F(0))
            total=F(0)
            for fid in self.stars[node]:
                value=m.qvadd(value,areas[fid][0])
                total=m.qadd(total,areas[fid][1])
            norm=m.qdot(value,value)
            margin=m.qsub(norm,m.qmul(squared,m.qmul(F(len(self.stars[node])),total)))
            if norm<=0 or margin<=0: _reject("nonzero/quality raw area star")
            stars[node],norms[node]=value,norm
            star_gates.append(("star",node,margin))
        projections=[]
        for fid,face in self.faces:
            for node in face:
                dot=m.qdot(stars[node],areas[fid][0])
                margin=m.qsub(m.qmul(dot,dot),m.qmul(F(1,16),m.qmul(norms[node],areas[fid][1])))
                if dot<=0 or margin<0: _reject("projected ratio")
                projections.append(("projection",fid,node,dot,margin))
        return areas,edges,stars,norms,tuple(face_gates+star_gates+projections)

    def field(self,generators,norms):
        m=self.m
        algebra=FieldArithmetic(generators,m,self.limits)
        # Independent square-class admission, never a producer helper/tag.
        for index,radicand in enumerate(generators):
            for subset in range(2**index):
                ratio=m.qdiv(radicand,algebra.products[subset])
                if m.perfect_square(ratio) is not None:
                    _reject("dependent square class")
        dependencies=[]
        roots={}
        for node in sorted(norms):
            candidates=[]
            for mask in range(algebra.width):
                ratio=m.qdiv(norms[node],algebra.products[mask])
                rational_root=m.perfect_square(ratio)
                if rational_root is not None and rational_root>0:
                    candidates.append((rational_root,mask))
                    break
            if not candidates: _reject("unrepresented actual star radical")
            coefficient,mask=candidates[0]
            roots[node]=algebra.root_from_dependency(norms[node],coefficient,mask)
            dependencies.append((node,norms[node],coefficient,mask))
        return algebra,roots,tuple(dependencies)

    def lifted(self,x,stars,roots,a):
        normals={}
        points={}
        for node in self.nodes:
            reciprocal=a.inverse(roots[node])
            direction=tuple(a.scale(v,reciprocal) for v in stars[node])
            displacement=tuple(a.scale(self.rho[node],q) for q in direction)
            normals[node]=direction
            base=tuple(a.rational(v) for v in x[node])
            points[node,-1]=a.vsub(base,displacement)
            points[node,1]=a.vadd(base,displacement)
        return normals,points
    def fact(self,key,value,a):
        sign=a.sign(value)
        return sign,(key,value,a.enclose(value))

    def area(self,p,q,r,drop,a):
        axes=[i for i in range(3) if i!=drop]
        first=[a.sub(q[i],p[i]) for i in axes]
        second=[a.sub(r[i],p[i]) for i in axes]
        return a.sub(a.mul(first[0],second[1]),a.mul(first[1],second[0]))

    def cluster(self,cid,tet_ids,descriptors,points,a,reference=False):
        if type(tet_ids) is not tuple or not tet_ids or len(tet_ids)>self.limits.strict_limits.max_tetrahedra:
            _reject("cluster tetra bound")
        if len(set(tet_ids))!=len(tet_ids) or not set(tet_ids)<=set(self.tets):
            _reject("cluster exact membership")
        local={t:self.tets[t] for t in tet_ids}
        boundary,links=_solid_links(local)
        vertices=tuple(sorted({v for t in local.values() for v in t}))
        if not reference and (len(vertices)>self.limits.max_cluster_vertices or
                              len(boundary)>self.limits.max_cluster_boundary_triangles):
            _reject("current cluster cap")
        if not 1<=len(descriptors)<=256: _reject("face group bound")
        boundary_lookup={tuple(sorted(f)):f for f in boundary}
        used=[]
        zero_keys=set()
        planes=[]
        for descriptor in descriptors:
            keys=tuple(tuple(sorted(t)) for t in descriptor.triangles)
            if not keys or len(keys)>128 or len(set(keys))!=len(keys) or not set(keys)<=set(boundary_lookup):
                _reject("face group complete triangle membership")
            used.extend(keys)
            triangles=tuple(boundary_lookup[k] for k in keys)
            cycle,_=_triangle_manifold(triangles,True)
            if _rotation_cycle(cycle)!=_rotation_cycle(descriptor.cycle):
                _reject("face boundary cycle orientation")
            if not reference and len(cycle)>self.limits.max_face_polygon_vertices:
                _reject("face polygon bound")
            if len({points[v] for v in cycle})!=len(cycle):
                _reject("face distinct vertices")
            drop=descriptor.drop_axis
            if type(drop) is not int or drop not in (0,1,2): _reject("projection axis")
            v0,v1,v2=(points[v] for v in triangles[0])
            normal=a.cross(a.vsub(v1,v0),a.vsub(v2,v0))
            if normal[drop]==a.zero: _reject("nonzero projection")
            supports=[]
            zero=[]
            for vertex in vertices:
                sign,row=self.fact(vertex,a.dot(normal,a.vsub(points[vertex],v0)),a)
                if sign>0: _reject("full cluster support")
                if sign==0: zero.append(vertex)
                supports.append(row)
            face_vertices=tuple(sorted({v for t in triangles for v in t}))
            if tuple(zero)!=face_vertices or face_vertices in zero_keys:
                _reject("full exact zero-support face group")
            zero_keys.add(face_vertices)
            oriented=[]
            orientation=None
            for triangle in triangles:
                value=self.area(*(points[v] for v in triangle),drop,a)
                sign,row=self.fact(tuple(sorted(triangle)),value,a)
                if sign==0 or (orientation is not None and sign!=orientation):
                    _reject("positive face triangulation")
                orientation=sign
                oriented.append(row)
            tests=[]
            for index,v in enumerate(cycle):
                w=cycle[(index+1)%len(cycle)]
                strict=False
                for u in face_vertices:
                    sign,row=self.fact((v,w,u),self.area(points[v],points[w],points[u],drop,a),a)
                    if sign*orientation<0: _reject("convex face full vertex support")
                    strict=strict or sign!=0
                    tests.append(row)
                if not strict: _reject("nonzero monotone boundary edge")
            planes.append(BoundaryPlaneEvidence(face_vertices,tuple(sorted(keys)),cycle,drop,
                          v0,normal,tuple(supports),tuple(oriented),tuple(tests)))
        if len(used)!=len(boundary_lookup) or set(used)!=set(boundary_lookup):
            _reject("entire boundary triangle partition")
        determinants=[]
        for tid,t in sorted(local.items()):
            origin=points[t[0]]
            edges=tuple(a.vsub(points[v],origin) for v in t[1:])
            value=a.dot(edges[0],a.cross(edges[1],edges[2]))
            sign,row=self.fact(tid,value,a)
            if sign<=0: _reject("positive oriented actual tetrahedron")
            determinants.append(row)
        return ConvexClusterBoundaryWitness(cid,tuple(sorted(local)),boundary,
                   tuple(sorted(planes,key=lambda p:p.vertices)),tuple(determinants),links)

    def reference(self,claim,generators):
        if type(claim) is not tuple or len(claim)!=3: _reject("reference evidence shape")
        alleged,alleged_separation,alleged_dependencies=claim
        if len(alleged)!=len(self.components): _reject("all reference components")
        raw=self.law(self.rest)
        rational=True
        for value in raw[3].values():
            if self.m.perfect_square(value) is None:
                rational=False
                break
        a,roots,dependencies=self.field(() if rational else generators,raw[3])
        _,points=self.lifted(self.rest,raw[2],roots,a)
        proofs=[]
        for cid,component in enumerate(self.components):
            if len({self.rho[node] for node in component})!=1:
                _reject("constant reference thickness")
            if len(alleged[cid])!=3: _reject("reference component evidence")
            normal=raw[0][next(fid for fid,face in self.faces if face[0] in component)][0]
            drop=next((i for i,q in enumerate(normal) if q),None)
            if drop is None: _reject("reference chart nonzero normal")
            origin=self.rest[component[0]]
            plane=[]
            for node in component:
                residual=self.m.qdot(normal,self.m.qvsub(self.rest[node],origin))
                if residual: _reject("exact original planarity")
                plane.append((node,residual))
            axes=tuple(i for i in range(3) if i!=drop)
            edge_rows=[]
            cycle=self.cycles[cid]
            for i,u in enumerate(cycle):
                v=cycle[(i+1)%len(cycle)]
                edge=tuple(self.m.qsub(self.rest[v][j],self.rest[u][j]) for j in axes)
                values=[]
                for node in component:
                    delta=tuple(self.m.qsub(self.rest[node][j],self.rest[u][j]) for j in axes)
                    values.append(self.m.qcross2(edge,delta))
                if not any(values) or min(values)<0<max(values): _reject("convex original chart")
                edge_rows.append((u,v,tuple(zip(component,values,strict=True))))
            local=tuple(key for key,t in sorted(self.tets.items()) if t[0][0] in component)
            candidate=alleged[cid][2]
            if type(candidate) is not ConvexClusterBoundaryWitness: _reject("reference cluster type")
            independently=self.cluster(cid,local,candidate.planes,points,a,reference=True)
            proofs.append((tuple(plane),tuple(edge_rows),independently))
        if len(alleged_separation)!=len(self.components)*(len(self.components)-1)//2:
            _reject("reference component pair coverage")
        seen=set()
        separation=[]
        for entry in alleged_separation:
            if type(entry) is not tuple or len(entry)!=4: _reject("reference support tuple")
            source,target,index,rows=entry
            if any(type(i) is not int for i in (source,target,index)):
                _reject("reference support IDs")
            if source==target or not 0<=source<len(proofs) or not 0<=target<len(proofs):
                _reject("reference support owners")
            pair=tuple(sorted((source,target)))
            if pair in seen: _reject("reference support duplicate pair")
            seen.add(pair)
            if not 0<=index<len(proofs[source][2].planes): _reject("reference support plane index")
            p=proofs[source][2].planes[index]
            facts=[]
            for node in self.components[target]:
                for sign in (-1,1):
                    vertex=node,sign
                    side,row=self.fact(vertex,a.dot(p.normal,a.vsub(points[vertex],p.origin)),a)
                    if side<=0: _reject("strict original component separation")
                    facts.append(row)
            separation.append((source,target,index,tuple(facts)))
        expected=(tuple(proofs),tuple(separation),dependencies)
        _same(claim,expected,self.m)
        return expected

    def maps(self,x,raw,roots,normals,points,a,publish):
        # Direct vertex differentiation of each face, independently of producer H.
        m=self.m
        adjacency={node:tuple(sorted({j for _,f in self.faces if node in f for j in f}))
                   for node in self.nodes}
        h={i:{j:(F(0),F(0),F(0)) for j in adjacency[i]} for i in self.nodes}
        for _,(i,j,k) in self.faces:
            terms={i:m.qvsub(x[k],x[j]),j:m.qvsub(x[i],x[k]),k:m.qvsub(x[j],x[i])}
            for owner in (i,j,k):
                for node,term in terms.items():
                    h[owner][node]=m.qvadd(h[owner][node],term)
        result=[]
        for node in self.nodes:
            reciprocal=a.inverse(roots[node])
            s=raw[2][node]
            norm=raw[3][node]
            d=[]
            for row in range(3):
                entries=[]
                for column in range(3):
                    quotient=m.qdiv(m.qmul(s[row],s[column]),norm)
                    projection=m.qsub(F(row==column),quotient)
                    entries.append(a.scale(m.qmul(self.rho[node],projection),reciprocal))
                d.append(tuple(entries))
            derivatives={}
            for neighbor in adjacency[node]:
                hx,hy,hz=h[node][neighbor]
                skew=((F(0),m.qneg(hz),hy),(hz,F(0),m.qneg(hx)),(m.qneg(hy),hx,F(0)))
                matrix=[]
                for row in range(3):
                    columns=[]
                    for column in range(3):
                        total=a.zero
                        for inner in range(3):
                            total=a.add(total,a.scale(skew[inner][column],d[row][inner]))
                        columns.append(total)
                    matrix.append(tuple(columns))
                derivatives[neighbor]=tuple(matrix)
            for sign in (-1,1):
                vertex=node,sign
                matrices=[]
                for neighbor in adjacency[node]:
                    matrix=[]
                    for row in range(3):
                        values=[]
                        for column in range(3):
                            identity=a.rational(F(node==neighbor and row==column))
                            offset=derivatives[neighbor][row][column]
                            values.append(a.add(identity,offset) if sign==1 else a.sub(identity,offset))
                        matrix.append(tuple(values))
                    matrices.append(tuple(matrix))
                representatives=[]
                if publish:
                    for tid,t in sorted(self.tets.items()):
                        if vertex not in t: continue
                        weights=tuple(a.one if v==vertex else a.zero for v in t)
                        total=a.zero
                        for q in weights: total=a.add(total,q)
                        if total!=a.one: _reject("vertex representation normalization")
                        representatives.append((tid,weights))
                point_pub=a.publish(points[vertex],(3,)) if publish else None
                jac_pub=(a.publish(tuple(q for matrix in matrices for row in matrix for q in row),
                                   (len(matrices),3,3)) if publish else None)
                result.append(BoundaryPointBasis(vertex,adjacency[node],points[vertex],
                              tuple(matrices),point_pub,jac_pub,tuple(representatives)))
        return tuple(result)
    def matching(self,hint,clusters,points,a):
        left,right=hint.clusters
        if left>=right or left not in clusters or right not in clusters: _reject("pair owners")
        active_a=tuple(sorted({v for p in clusters[left].planes for v in p.vertices}))
        active_b=tuple(sorted({v for p in clusters[right].planes for v in p.vertices}))
        supports=[]
        axes=[]
        separated=False
        for candidate in hint.supports:
            normal=tuple(a.element(c) for c in candidate.normal)
            if all(c==a.zero for c in normal) or candidate.anchor not in active_a:
                _reject("support normal/anchor")
            if len(axes)==1 and all(c==a.zero for c in a.cross(axes[0],normal)):
                _reject("support descent rank")
            if len(axes)==2 and a.dot(axes[0],a.cross(axes[1],normal))==a.zero:
                _reject("third support rank")
            axes.append(normal)
            origin=points[candidate.anchor]
            evaluated=[]
            for active in (active_a,active_b):
                rows=tuple(self.fact(v,a.dot(normal,a.vsub(points[v],origin)),a) for v in active)
                evaluated.append(rows)
            rows_a,rows_b=evaluated
            if any(s>0 for s,_ in rows_a) or any(s<0 for s,_ in rows_b):
                _reject("full support inequalities")
            zero_a=tuple(v for v,(s,_) in zip(active_a,rows_a,strict=True) if not s)
            zero_b=tuple(v for v,(s,_) in zip(active_b,rows_b,strict=True) if not s)
            supports.append(BoundarySupportEvidence(normal,candidate.anchor,active_a,active_b,
                             tuple(row for _,row in rows_a),tuple(row for _,row in rows_b),zero_a,zero_b))
            if not zero_a or not zero_b:
                separated=True
                break
            active_a,active_b=zero_a,zero_b
        if separated:
            return MatchingIntersectionWitness((left,right),tuple(supports),(),(),"empty")
        mapping=dict(hint.matching)
        if len(mapping)!=len(hint.matching) or set(mapping)!=set(active_a) or len(set(mapping.values()))!=len(mapping) or set(mapping.values())!=set(active_b):
            _reject("full matching vertices")
        for v,w in sorted(mapping.items()):
            differences=tuple(a.sub(p,q) for p,q in zip(points[v],points[w],strict=True))
            if differences!=(a.zero,a.zero,a.zero): _reject("matching exact coordinates")
        complexes=[]
        for cid,active in ((left,active_a),(right,active_b)):
            simplices={tuple(sorted(s)) for face in clusters[cid].boundary_facets
                       for degree in range(1,4) for s in combinations(face,degree)
                       if set(s)<=set(active)}
            complexes.append(simplices)
        transformed={tuple(sorted(mapping[v] for v in s)) for s in complexes[0]}
        if transformed!=complexes[1]: _reject("complete matching face/subface complex")
        pieces=[]
        labels=set()
        for first in sorted(complexes[0]):
            second=tuple(sorted(mapping[v] for v in first))
            if all(mapping[v]==v for v in first):
                label="material_incidence"
            elif first in self.boundary_strata and second in self.boundary_strata:
                label="true_boundary_contact"
            else: _reject("global material boundary preimage")
            pieces.append((first,second,label))
            labels.add(label)
        label=next(iter(labels)) if len(labels)==1 else "incidence_and_boundary_contact"
        return MatchingIntersectionWitness((left,right),tuple(supports),tuple(sorted(mapping.items())),
                                           tuple(pieces),label)

    def fans(self,clusters,exterior,a):
        actual_boundary={}
        for f in self.boundary:
            key=tuple(sorted(f))
            for degree in range(1,4):
                for s in combinations(key,degree): actual_boundary.setdefault(s,[]).append(key)
        directions=dict(exterior)
        if len(directions)!=len(exterior) or set(directions)!=set(actual_boundary):
            _reject("complete ordinary boundary exterior directions")
        owner={t:cid for cid,c in clusters.items() for t in c.tetra_ids}
        output=[]
        for simplex,incident in sorted(self.strata.items()):
            cids=tuple(sorted({owner[t] for t in incident}))
            cones=[]
            for cid in cids:
                indices=tuple(i for i,face in enumerate(clusters[cid].planes) if set(simplex)<=set(face.vertices))
                if len(indices)>self.limits.max_fan_planes: _reject("active fan bound")
                cones.append((cid,indices))
            direction=()
            witnesses=[]
            if simplex in directions:
                direction=tuple(a.element(v) for v in directions[simplex])
                if not any(v!=a.zero for v in direction): _reject("nonzero exterior")
                for cid,active in cones:
                    success=False
                    for index in active:
                        value=a.dot(clusters[cid].planes[index].normal,direction)
                        sign,row=self.fact((cid,index),value,a)
                        witnesses.append(row)
                        if sign>0:
                            success=True
                            break
                    if not success: _reject("individual material star proper exterior")
            output.append(BoundaryFanWitness(simplex,tuple(sorted(incident)),
                          tuple(sorted(actual_boundary.get(simplex,()))),tuple(cones),
                          direction,tuple(witnesses)))
        return tuple(output)

    def coverage(self,clusters,pairs,fans,points):
        membership={t:cid for cid,c in clusters.items() for t in c.tetra_ids}
        lookup={p.clusters:p for p in pairs}
        all_pairs=[]
        for first,second in combinations(sorted(self.tets),2):
            shared=tuple(sorted(set(self.tets[first])&set(self.tets[second])))
            ia,ib=membership[first],membership[second]
            if ia==ib:
                cpair=(ia,ia)
                pieces=((shared,shared,"material_incidence"),) if shared else ()
                label="material_incidence" if shared else "empty"
            else:
                cpair=tuple(sorted((ia,ib)))
                rows=[]
                for s,t,classification in lookup[cpair].intersection_simplices:
                    left,right=(s,t) if ia<ib else (t,s)
                    if set(left)<=set(self.tets[first]) and set(right)<=set(self.tets[second]):
                        rows.append((left,right,classification))
                pieces=tuple(rows)
                classes={v[2] for v in pieces}
                label=("empty" if not classes else next(iter(classes)) if len(classes)==1
                       else "incidence_and_boundary_contact")
                if shared and (shared,shared,"material_incidence") not in pieces:
                    _reject("shared material cross-cluster coverage")
            all_pairs.append(BoundaryPairWitness((first,second),shared,cpair,pieces,label))
        by_simplex={v.material_stratum:v for v in fans}
        contacts=[]
        owners=set()
        for pair in pairs:
            mapping=dict(pair.matching)
            for first,second,classification in pair.intersection_simplices:
                if classification!="true_boundary_contact": continue
                key=(first,second)
                if key in owners: _reject("remote owner duplication")
                owners.add(key)
                subfaces=tuple((s,tuple(sorted(mapping[v] for v in s)))
                               for count in range(1,len(first)) for s in combinations(first,count))
                fa,fb=by_simplex[first],by_simplex[second]
                contacts.append(BoundaryContactStratum(key,tuple(points[v] for v in first),subfaces,
                               (fa.true_boundary_facets,fb.true_boundary_facets),pair.supports,
                               (fa.occupied_cones,fb.occupied_cones)))
        return tuple(all_pairs),tuple(sorted(contacts,key=lambda c:c.owners))


def _header(claim,expected_type,reference,positions,limits,witness,pending,kind):
    if type(claim) is not expected_type: _reject("certificate type")
    if claim.reference_id!=reference.reference_id or claim.policy!=limits:
        _reject("reference/policy binding")
    expected=digest((kind,reference.reference_id,reference.node_ids,reference.face_ids,
                     reference.oriented_nodes,positions,limits,witness))
    if claim.original_input_binding!=expected: _reject("complete original binding")
    if (type(claim.original_positions) is not np.ndarray or
            claim.original_positions.dtype!=positions.dtype or
            claim.original_positions.shape!=positions.shape or
            claim.original_positions.tobytes()!=positions.tobytes()):
        _reject("original position bytes")
    if claim.precision!="ideal-from-stored-fp64": _reject("precision kind")
    for number in (claim.operations,claim.producer_operations,claim.checker_operations,claim.peak_fraction_bits):
        if type(number) is not int or number<0: _reject("counter type")
    if claim.operations!=claim.producer_operations+claim.checker_operations:
        _reject("counter aggregate/reset")
    if not 0<claim.operations<=limits.strict_limits.max_scalar_operations:
        _reject("operation receipt range")
    if claim.producer_operations<=0 or (not pending and claim.checker_operations<=0):
        _reject("complete producer/inner accounting")
    if not 0<claim.peak_fraction_bits<=limits.strict_limits.max_fraction_bits:
        _reject("bit receipt range")


def _audit_static(claim,reference,positions,witness_hint,limits,meter,pending=False):
    _header(claim,ShellBoundaryAdmission,reference,positions,limits,witness_hint,pending,"static")
    if claim.schema!="shell-boundary-v1" or claim.status!="weak_material_geometry_certified" or claim.proof_kind!="convex-cluster-matching-v1":
        _reject("static schema/status/domain")
    if type(witness_hint) is not BoundaryWitnessHint: _reject("untrusted hint type")
    witness_hint.__post_init__()
    if len(witness_hint.clusters)>limits.max_clusters: _reject("tight cluster cap")
    audit=Audit(reference,positions,limits,meter)
    if type(claim.field) is not BoundaryFieldWitness: _reject("field witness type")
    meter.set_bits(claim.field.sqrt_bits)
    reference_facts=audit.reference(claim.topology.reference_facts,witness_hint.generators)
    raw=audit.law(audit.x)
    a,roots,dependencies=audit.field(witness_hint.generators,raw[3])
    normals,points=audit.lifted(audit.x,raw[2],roots,a)
    flat=tuple(t for c in witness_hint.clusters for t in c.tetra_ids)
    cids=tuple(c.cluster_id for c in witness_hint.clusters)
    if len(set(cids))!=len(cids) or len(flat)!=len(set(flat)) or set(flat)!=set(audit.tets):
        _reject("complete cluster partition")
    clusters={c.cluster_id:audit.cluster(c.cluster_id,c.tetra_ids,c.faces,points,a)
              for c in sorted(witness_hint.clusters,key=lambda c:c.cluster_id)}
    needed=set(combinations(sorted(clusters),2))
    if len(witness_hint.pairs)!=len(needed) or {p.clusters for p in witness_hint.pairs}!=needed:
        _reject("complete cluster pair hints")
    pairs=tuple(audit.matching(p,clusters,points,a) for p in sorted(witness_hint.pairs,key=lambda p:p.clusters))
    fans=audit.fans(clusters,witness_hint.exterior,a)
    all_pairs,contacts=audit.coverage(clusters,pairs,fans,points)
    basis=audit.maps(audit.x,raw,roots,normals,points,a,True)
    topology=BoundaryTopology(audit.nodes,audit.faces,tuple(sorted(audit.tets.items())),
               audit.boundary,audit.components,
               tuple((s,tuple(sorted(t))) for s,t in sorted(audit.strata.items())),reference_facts)
    law=BoundaryLaw(tuple(sorted(raw[0].items())),tuple(sorted(raw[2].items())),
                   tuple(sorted(raw[3].items())),tuple(sorted(normals.items())),raw[4])
    field=BoundaryFieldWitness(a.generators,tuple(sorted(raw[3].items())),dependencies,
                              tuple(range(a.width)),meter.bits)
    for actual,expected in (
        (claim.field,field),(claim.topology,topology),(claim.law,law),
        (claim.clusters,tuple(clusters.values())),(claim.cluster_pairs,pairs),
        (claim.all_pairs,all_pairs),(claim.full_strata_and_fans,fans),
        (claim.contacts,contacts),(claim.basis,basis)):
        _same(actual,expected,meter)
    return audit,raw,a,roots,normals,points,clusters,pairs,fans,basis


def _check_static(claim,reference,positions,witness_hint,limits,meter,*,pending=False):
    _audit_static(claim,reference,positions,witness_hint,limits,meter,pending)
    return True


def verify_shell_boundary(certificate,reference,positions,*,witness_hint,limits):
    try:
        return _check_static(certificate,reference,positions,witness_hint,limits,Meter(limits.strict_limits))
    except (ShellUnresolved,ValueError,TypeError,AttributeError,KeyError,IndexError,OverflowError):
        return False


def _check_point(claim,reference,positions,address,limits,meter,*,pending=False):
    _header(claim,ShellExactBoundaryPoint,reference,positions,limits,address,pending,"point")
    if claim.schema!="shell-exact-boundary-point-v1" or claim.status!="exact_boundary_map_only":
        _reject("map-only schema/status")
    if type(address) is not ExactBoundaryAddress: _reject("exact address type")
    address.__post_init__()
    if claim.address!=address or address.reference_id!=reference.reference_id:
        _reject("original exact address binding")
    audit=Audit(reference,positions,limits,meter)
    meter.set_bits(claim.field.sqrt_bits)
    audit.reference(claim.reference_facts,address.generators)
    raw=audit.law(audit.x)
    a,roots,dependencies=audit.field(address.generators,raw[3])
    normals,points=audit.lifted(audit.x,raw[2],roots,a)
    simplex=address.minimal_simplex
    if simplex!=tuple(sorted(set(simplex))) or simplex not in audit.boundary_strata:
        _reject("true boundary minimal material simplex")
    weights=tuple(a.element(w) for w in address.weights)
    total=a.zero
    for w in weights:
        if a.sign(w)<=0: _reject("positive exact minimal weights")
        total=a.add(total,w)
    if total!=a.one: _reject("exact material weight sum")
    reps=tuple((tid,tuple(weights[simplex.index(v)] if v in simplex else a.zero for v in t))
               for tid,t in sorted(audit.tets.items()) if set(simplex)<=set(t))
    if address.representatives!=reps: _reject("complete incident exact representations")
    for _,qs in reps:
        total=a.zero
        for q in qs: total=a.add(total,q)
        if total!=a.one: _reject("exact tetra representation")
    basis=audit.maps(audit.x,raw,roots,normals,points,a,False)
    lookup={b.vertex:b for b in basis}
    ids=tuple(sorted({n for v in simplex for n in lookup[v].node_ids}))
    point=[a.zero]*3
    blocks={i:[[a.zero]*3 for _ in range(3)] for i in ids}
    for v,weight in zip(simplex,weights,strict=True):
        b=lookup[v]
        for row in range(3): point[row]=a.add(point[row],a.mul(weight,b.exact_point[row]))
        for node,matrix in zip(b.node_ids,b.exact_jacobian,strict=True):
            for row in range(3):
                for column in range(3):
                    blocks[node][row][column]=a.add(blocks[node][row][column],a.mul(weight,matrix[row][column]))
    point=tuple(point)
    matrices=tuple(tuple(tuple(row) for row in blocks[i]) for i in ids)
    point_pub=a.publish(point,(3,))
    matrix_pub=a.publish(tuple(q for matrix in matrices for row in matrix for q in row),(len(ids),3,3))
    law=BoundaryLaw(tuple(sorted(raw[0].items())),tuple(sorted(raw[2].items())),
                   tuple(sorted(raw[3].items())),tuple(sorted(normals.items())),raw[4])
    field=BoundaryFieldWitness(a.generators,tuple(sorted(raw[3].items())),dependencies,
                               tuple(range(a.width)),meter.bits)
    for actual,expected in ((claim.node_ids,ids),(claim.exact_point,point),
            (claim.exact_jacobian,matrices),(claim.point,point_pub),(claim.jacobian,matrix_pub),
            (claim.law,law),(claim.field,field)):
        _same(actual,expected,meter)
    return True


def verify_shell_exact_boundary_point(receipt,reference,positions,address,*,limits):
    try:
        return _check_point(receipt,reference,positions,address,limits,Meter(limits.strict_limits))
    except (ShellUnresolved,ValueError,TypeError,AttributeError,KeyError,IndexError,OverflowError):
        return False