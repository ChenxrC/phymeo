"""Finite weak material-boundary geometry; never a physical contact acceptance."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import combinations

import numpy as np

from ..constraints.conforming_shell import ConformingShellReference, ShellUnresolved
from ..constraints.shell_boundary import (
    BoundaryContactStratum, BoundaryFaceHint, BoundaryFanWitness,
    BoundaryFieldWitness, BoundaryLaw, BoundaryPairWitness,
    BoundaryPlaneEvidence, BoundaryPointBasis, BoundaryProofLimits,
    BoundarySupportEvidence, BoundaryTopology, BoundaryWitnessHint,
    ClusterHint, ConvexClusterBoundaryWitness, ExactBoundaryAddress,
    MatchingIntersectionWitness, ShellBoundaryAdmission, ShellExactBoundaryPoint,
    digest,
)
from ._shell_contact_field import FieldArithmetic, Meter


def _fail(category,detail):
    raise ShellUnresolved(category,detail)


def _cycle(values):
    values=tuple(values)
    return min(values[i:]+values[:i] for i in range(len(values)))


def _orientation(values):
    return -1 if sum(a>b for i,a in enumerate(values) for b in values[i+1:])%2 else 1


def _connected(vertices,edges):
    if not vertices: return False
    graph={v:set() for v in vertices}
    for a,b in edges:
        graph[a].add(b)
        graph[b].add(a)
    visited,pending=set(),[min(vertices)]
    while pending:
        v=pending.pop()
        if v not in visited:
            visited.add(v)
            pending.extend(graph[v]-visited)
    return visited==set(vertices)


def _surface(triangles,kind):
    """Complete oriented disk/sphere combinatorics including every vertex link."""
    faces=tuple(triangles)
    keys=[tuple(sorted(f)) for f in faces]
    if not faces or len(set(keys))!=len(keys) or any(len(set(f))!=3 for f in faces):
        _fail("incomplete_material_cover","duplicate/degenerate surface face")
    edge_entries={}
    vertices=set()
    for f in faces:
        vertices.update(f)
        for a,b in zip(f,f[1:]+f[:1],strict=True):
            edge_entries.setdefault(tuple(sorted((a,b))),[]).append((a,b))
    if not _connected(vertices,edge_entries):
        _fail("incomplete_material_cover","disconnected surface")
    boundary=[]
    for entries in edge_entries.values():
        if len(entries)==1: boundary.append(entries[0])
        elif len(entries)!=2 or entries[0]!=entries[1][::-1]:
            _fail("incomplete_material_cover","surface orientation/manifold edge")
    expected=1 if kind=="disk" else 2
    if len(vertices)-len(edge_entries)+len(faces)!=expected:
        _fail("incomplete_material_cover","surface Euler characteristic")
    if (kind=="sphere" and boundary) or (kind=="disk" and not boundary):
        _fail("incomplete_material_cover","surface boundary kind")
    cycle=()
    if boundary:
        outgoing=dict(boundary)
        if len(outgoing)!=len(boundary) or len(set(outgoing.values()))!=len(boundary):
            _fail("incomplete_material_cover","branched surface boundary")
        ordered=[min(outgoing)]
        while outgoing.get(ordered[-1])!=ordered[0]:
            nxt=outgoing.get(ordered[-1])
            if nxt is None or nxt in ordered:
                _fail("incomplete_material_cover","open/repeated boundary")
            ordered.append(nxt)
        if len(ordered)!=len(outgoing):
            _fail("incomplete_material_cover","multiple boundary cycles")
        cycle=tuple(ordered)
    links=[]
    boundary_nodes={v for e in boundary for v in e}
    for v in sorted(vertices):
        edges=[tuple(u for u in f if u!=v) for f in faces if v in f]
        link_nodes={u for e in edges for u in e}
        degree={u:sum(u in e for e in edges) for u in link_nodes}
        expected_degree=sorted(degree.values())
        valid=(expected_degree.count(1)==2 and all(d in (1,2) for d in expected_degree)
               if v in boundary_nodes else all(d==2 for d in expected_degree))
        if not valid or not _connected(link_nodes,edges):
            _fail("incomplete_material_cover","surface vertex link")
        links.append((v,tuple(sorted(tuple(sorted(e)) for e in edges))))
    return cycle,tuple(links)


def _tet_complex(tets):
    facet_entries={}
    vertices={v for t in tets.values() for v in t}
    if not tets: _fail("incomplete_material_cover","empty tetra cluster")
    for key,t in tets.items():
        if len(set(t))!=4: _fail("invalid_reference","repeated tetra vertex")
        for omitted in range(4):
            f=tuple(v for i,v in enumerate(t) if i!=omitted)
            if omitted%2: f=(f[0],f[2],f[1])
            facet_entries.setdefault(tuple(sorted(f)),[]).append((_cycle(f),key))
    boundary=[]
    dual=[]
    for key,entries in sorted(facet_entries.items()):
        if len(entries)==1: boundary.append(entries[0][0])
        elif len(entries)==2 and _orientation(entries[0][0])!=_orientation(entries[1][0]):
            dual.append((entries[0][1],entries[1][1]))
        else: _fail("invalid_reference","tetra facets do not cancel")
    if not _connected(set(tets),dual):
        _fail("incomplete_material_cover","disconnected tetra interior")
    _,surface_links=_surface(tuple(boundary),"sphere")
    boundary_sets={tuple(sorted(s)) for f in boundary
                   for n in (1,2) for s in combinations(f,n)}
    links=[("boundary",surface_links)]
    for v in sorted(vertices):
        opposite=[]
        for t in tets.values():
            if v in t:
                i=t.index(v)
                f=tuple(u for u in t if u!=v)
                if i%2: f=(f[0],f[2],f[1])
                opposite.append(f)
        cycle,details=_surface(tuple(opposite),"disk" if (v,) in boundary_sets else "sphere")
        links.append(("vertex",v,cycle,details))
    for edge in sorted({tuple(sorted(e)) for t in tets.values() for e in combinations(t,2)}):
        opposite=[tuple(v for v in t if v not in edge) for t in tets.values() if set(edge)<=set(t)]
        points={v for e in opposite for v in e}
        degrees=[sum(v in e for e in opposite) for v in sorted(points)]
        valid=(degrees.count(1)==2 and all(d in (1,2) for d in degrees)
               if edge in boundary_sets else all(d==2 for d in degrees))
        if not valid or not _connected(points,opposite):
            _fail("incomplete_material_cover","tetra edge link")
        links.append(("edge",edge,tuple(sorted(tuple(sorted(e)) for e in opposite))))
    return tuple(boundary),tuple(links)


def _inputs(reference,positions,limits,meter):
    if type(reference) is not ConformingShellReference or type(limits) is not BoundaryProofLimits:
        raise TypeError("original reference/public policy required")
    limits.__post_init__()
    arrays=((reference.node_ids,"int64",1),(reference.face_ids,"int64",1),
            (reference.oriented_nodes,"int64",2),(reference.reference_positions,"float64",2),
            (reference.half_thickness_m,"float64",1))
    for value,dtype,ndim in arrays:
        if type(value) is not np.ndarray or value.dtype!=np.dtype(dtype) or value.ndim!=ndim:
            _fail("invalid_shape_or_identity","reference array type")
    n,f=len(reference.node_ids),len(reference.face_ids)
    strict=limits.strict_limits
    if not 3<=n<=strict.max_nodes or not 1<=f<=strict.max_faces or 3*f>strict.max_tetrahedra:
        _fail("shape_budget","reference dimensions")
    if reference.oriented_nodes.shape!=(f,3) or reference.reference_positions.shape!=(n,3) or reference.half_thickness_m.shape!=(n,):
        _fail("invalid_shape_or_identity","reference shapes")
    for value,_,_ in arrays:
        if not np.isfinite(value).all(): _fail("invalid_reference","nonfinite input")
    if type(reference.domain) is not str or not reference.domain or len(reference.domain.encode())>256:
        _fail("invalid_reference","domain")
    if type(positions) is not np.ndarray or positions.dtype!=np.dtype("float64") or positions.shape!=(n,3):
        _fail("invalid_shape_or_identity","FP64[N,3] positions")
    if not np.isfinite(positions).all(): _fail("invalid_shape_or_identity","nonfinite positions")
    ids=tuple(int(v) for v in reference.node_ids)
    face_ids=tuple(int(v) for v in reference.face_ids)
    if len(set(ids))!=n or len(set(face_ids))!=f or min(ids)<0 or min(face_ids)<0:
        _fail("invalid_reference","duplicate/negative IDs")
    nodes=tuple(sorted(ids))
    faces=tuple(sorted((fid,tuple(int(i) for i in row))
                       for fid,row in zip(face_ids,reference.oriented_nodes,strict=True)))
    edge_entries={}
    seen=set()
    stars={i:[] for i in nodes}
    for fid,face in faces:
        if len(set(face))!=3 or not set(face)<=set(nodes) or tuple(sorted(face)) in seen:
            _fail("invalid_reference","face identity")
        seen.add(tuple(sorted(face)))
        for i in face: stars[i].append(fid)
        for a,b in zip(face,face[1:]+face[:1],strict=True):
            edge_entries.setdefault(tuple(sorted((a,b))),[]).append((a,b))
    for entries in edge_entries.values():
        if len(entries)>2 or (len(entries)==2 and entries[0]!=entries[1][::-1]):
            _fail("invalid_reference","oriented middle edge")
    if any(not v or len(v)>strict.max_star_faces for v in stars.values()):
        _fail("shape_budget","complete material stars")
    unseen=set(nodes)
    components=[]
    while unseen:
        pending=[min(unseen)]
        component=set()
        while pending:
            v=pending.pop()
            if v not in component:
                component.add(v)
                pending.extend(u for e in edge_entries if v in e for u in e if u not in component)
        unseen-=component
        components.append(tuple(sorted(component)))
    cycles=[]
    for component in components:
        cycle,_=_surface(tuple(face for _,face in faces if face[0] in component),"disk")
        cycles.append(cycle)
    tets={}
    for fid,face in faces:
        i,j,k=sorted(face)
        triples=(((i,-1),(j,-1),(k,-1),(k,1)),
                 ((i,-1),(j,-1),(j,1),(k,1)),
                 ((i,-1),(i,1),(j,1),(k,1)))
        for stair,t in enumerate(triples):
            if _orientation(face)*(1,-1,1)[stair]<0:
                t=t[:2]+(t[3],t[2])
            tets[(fid,stair)]=t
    if len(tets)*(len(tets)-1)//2>strict.max_pairs:
        _fail("shape_budget","complete pair count")
    boundary=[]
    for component in components:
        local={key:t for key,t in tets.items() if t[0][0] in component}
        b,_=_tet_complex(local)
        boundary.extend(b)
    strata={}
    for key,t in tets.items():
        for size in range(1,5):
            for simplex in combinations(t,size):
                strata.setdefault(tuple(sorted(simplex)),[]).append(key)
    if len(strata)>limits.max_material_strata:
        _fail("shape_budget","all material strata")
    positions_by_id={i:tuple(meter.input_float(v) for v in positions[row]) for row,i in enumerate(ids)}
    rest={i:tuple(meter.input_float(v) for v in reference.reference_positions[row]) for row,i in enumerate(ids)}
    rho={i:meter.input_float(reference.half_thickness_m[row]) for row,i in enumerate(ids)}
    if any(v<=0 for v in rho.values()): _fail("invalid_reference","thickness")
    canonical_nodes=sorted((i,reference.reference_positions[row].tobytes().hex(),
                            reference.half_thickness_m[row:row+1].tobytes().hex())
                           for row,i in enumerate(ids))
    law=reference.law
    if law.kind!="raw-area-star-director-v1" or law.projected_ratio_min!=F(1,4) or law.shape_policy!="invariant-1e-10-v1":
        _fail("invalid_reference","director law")
    expected=digest(("conforming-shell-reference-v1",reference.domain,canonical_nodes,
                     sorted((fid,_cycle(face)) for fid,face in faces),
                     law.kind,(1,4),law.shape_policy))
    if expected!=reference.reference_id:
        _fail("stale_binding","reference identity")
    return dict(nodes=nodes,faces=faces,tets=tets,boundary=tuple(sorted(boundary)),
                components=tuple(components),cycles=tuple(cycles),strata=strata,
                positions=positions_by_id,rest=rest,rho=rho,stars=stars,limits=limits)


def _raw(original,positions,m):
    areas,edges={},{}
    stars={i:(F(0),)*3 for i in original["nodes"]}
    gates=[]
    epsilon=m.input_float(1e-10)
    eps2=m.qmul(epsilon,epsilon)
    for fid,(i,j,k) in original["faces"]:
        e1=m.qvsub(positions[j],positions[i])
        e2=m.qvsub(positions[k],positions[i])
        e3=m.qvsub(positions[k],positions[j])
        c=m.qcross(e1,e2)
        c2=m.qdot(c,c)
        if c2<=0: _fail("zero_star","zero current/reference face")
        lengths=tuple(m.qdot(e,e) for e in (e1,e2,e3))
        quality=tuple(m.qsub(c2,m.qmul(eps2,m.qmul(q,q))) for q in lengths)
        if min(quality)<=0: _fail("law_domain_unknown","face shape")
        areas[fid]=(c,c2)
        edges[fid]=(e1,e2)
        gates.append(("face",fid,quality))
        for node in (i,j,k): stars[node]=m.qvadd(stars[node],c)
    norms={}
    for node,s in stars.items():
        s2=m.qdot(s,s)
        if s2<=0: _fail("zero_star","zero area star")
        norms[node]=s2
        total=F(0)
        for fid in original["stars"][node]: total=m.qadd(total,areas[fid][1])
        bound=m.qmul(eps2,m.qmul(F(len(original["stars"][node])),total))
        margin=m.qsub(s2,bound)
        if margin<=0: _fail("law_domain_unknown","sufficient star shape")
        gates.append(("star",node,margin))
    for fid,face in original["faces"]:
        c,c2=areas[fid]
        for node in face:
            dot=m.qdot(stars[node],c)
            margin=m.qsub(m.qmul(dot,dot),m.qmul(F(1,16),m.qmul(norms[node],c2)))
            if dot<=0 or margin<0: _fail("law_domain_unknown","projected director")
            gates.append(("projection",fid,node,dot,margin))
    return areas,edges,stars,norms,tuple(gates)


def _field(generators,norms,meter,limits):
    a=FieldArithmetic(generators,meter,limits)
    for i,generator in enumerate(a.generators):
        for mask in range(1<<i):
            if meter.perfect_square(meter.qdiv(generator,a.products[mask])) is not None:
                _fail("unsupported_algebraic_rank","dependent square classes")
    roots,dependencies={},[]
    for key,radicand in sorted(norms.items()):
        found=None
        for mask,product in enumerate(a.products):
            root=meter.perfect_square(meter.qdiv(radicand,product))
            if root is not None and root>0:
                found=(root,mask)
                break
        if found is None: _fail("unsupported_algebraic_rank","missing star radical dependency")
        coefficient,mask=found
        roots[key]=a.root_from_dependency(radicand,coefficient,mask)
        dependencies.append((key,radicand,coefficient,mask))
    return a,roots,tuple(dependencies)


def _lift(original,positions,stars,roots,a):
    normals,points={},{}
    for node in original["nodes"]:
        inverse=a.inverse(roots[node])
        normal=tuple(a.scale(q,inverse) for q in stars[node])
        offset=tuple(a.scale(original["rho"][node],v) for v in normal)
        base=tuple(a.rational(q) for q in positions[node])
        normals[node]=normal
        points[node,-1]=a.vsub(base,offset)
        points[node,1]=a.vadd(base,offset)
    return normals,points

def _area(p,q,r,drop,a):
    axes=tuple(i for i in range(3) if i!=drop)
    u=tuple(a.sub(q[i],p[i]) for i in axes)
    v=tuple(a.sub(r[i],p[i]) for i in axes)
    return a.sub(a.mul(u[0],v[1]),a.mul(u[1],v[0]))


def _fact(key,value,a):
    sign,bound=a.signed(value)
    return sign,(key,value,bound)


def _cluster(hint,tets,points,a,*,reference=False):
    if len(set(hint.tetra_ids))!=len(hint.tetra_ids) or not set(hint.tetra_ids)<=set(tets):
        _fail("incomplete_material_cover","cluster tetra membership")
    local={key:tets[key] for key in hint.tetra_ids}
    boundary,links=_tet_complex(local)
    vertices=tuple(sorted({v for t in local.values() for v in t}))
    if not reference and (len(vertices)>a.limits.max_cluster_vertices or
                          len(boundary)>a.limits.max_cluster_boundary_triangles):
        _fail("shape_budget","current convex cluster")
    oriented={tuple(sorted(f)):f for f in boundary}
    assigned=[]
    planes=[]
    plane_keys=set()
    for face in hint.faces:
        face.__post_init__()
        keys=tuple(tuple(sorted(f)) for f in face.triangles)
        if len(set(keys))!=len(keys) or not set(keys)<=set(oriented):
            _fail("incomplete_material_cover","face triangle membership")
        assigned.extend(keys)
        triangles=tuple(oriented[key] for key in keys)
        cycle,_=_surface(triangles,"disk")
        if _cycle(face.cycle)!=_cycle(cycle):
            _fail("incomplete_material_cover","face boundary order")
        if not reference and len(cycle)>a.limits.max_face_polygon_vertices:
            _fail("shape_budget","face polygon")
        group_vertices=tuple(sorted({v for f in triangles for v in f}))
        if len(set(points[v] for v in cycle))!=len(cycle):
            _fail("incomplete_material_cover","coincident boundary vertices")
        p,q,r=(points[v] for v in triangles[0])
        normal=a.cross(a.vsub(q,p),a.vsub(r,p))
        if normal[face.drop_axis]==a.zero:
            _fail("unsupported_witness","singular face projection")
        supports=[]
        zeros=[]
        for v in vertices:
            sign,fact=_fact(v,a.dot(normal,a.vsub(points[v],p)),a)
            if sign>0: _fail("unsupported_witness","cluster is not supported by face")
            if sign==0: zeros.append(v)
            supports.append(fact)
        key=tuple(zeros)
        if key!=group_vertices or key in plane_keys:
            _fail("incomplete_material_cover","partial/duplicate coplanar zero set")
        plane_keys.add(key)
        orientations=[]
        target=None
        for triangle in triangles:
            sign,fact=_fact(tuple(sorted(triangle)),
                            _area(*(points[v] for v in triangle),face.drop_axis,a),a)
            if sign==0 or (target is not None and target!=sign):
                _fail("unsupported_witness","nonpositive oriented face disk")
            target=sign
            orientations.append(fact)
        edge_tests=[]
        for v,w in zip(cycle,cycle[1:]+cycle[:1],strict=True):
            strict=False
            for u in group_vertices:
                sign,fact=_fact((v,w,u),_area(points[v],points[w],points[u],face.drop_axis,a),a)
                if sign*target<0:
                    _fail("unsupported_witness","nonconvex face boundary")
                strict=strict or sign!=0
                edge_tests.append(fact)
            if not strict:
                _fail("unsupported_witness","degenerate/backtracking face edge")
        planes.append(BoundaryPlaneEvidence(group_vertices,tuple(sorted(keys)),cycle,
                      face.drop_axis,p,normal,tuple(supports),tuple(orientations),tuple(edge_tests)))
    if len(assigned)!=len(oriented) or set(assigned)!=set(oriented):
        _fail("incomplete_material_cover","whole cluster boundary")
    determinants=[]
    for key,t in sorted(local.items()):
        p,q,r,s=(points[v] for v in t)
        value=a.determinant(a.vsub(q,p),a.vsub(r,p),a.vsub(s,p))
        sign,fact=_fact(key,value,a)
        if sign<=0: _fail("inverted_or_degenerate_tetra","strict oriented volume")
        determinants.append(fact)
    planes.sort(key=lambda p:p.vertices)
    return ConvexClusterBoundaryWitness(hint.cluster_id,tuple(sorted(local)),boundary,
                                       tuple(planes),tuple(determinants),links)


def _reference(original,generators,meter,limits):
    raw=_raw(original,original["rest"],meter)
    areas,_,stars,norms,_=raw
    rational=all(meter.perfect_square(v) is not None for v in norms.values())
    a,roots,dependencies=_field(() if rational else generators,norms,meter,limits)
    _,points=_lift(original,original["rest"],stars,roots,a)
    proofs=[]
    for cid,(component,cycle) in enumerate(zip(original["components"],original["cycles"],strict=True)):
        if len({original["rho"][i] for i in component})!=1:
            _fail("unsupported_reference_chart","constant component thickness required")
        first=next(fid for fid,face in original["faces"] if face[0] in component)
        normal=areas[first][0]
        origin=original["rest"][component[0]]
        drop=next((i for i,v in enumerate(normal) if v),None)
        if drop is None: _fail("invalid_reference","zero chart plane")
        planar=[]
        for node in component:
            v=meter.qdot(normal,meter.qvsub(original["rest"][node],origin))
            if v: _fail("unsupported_reference_chart","reference not exactly planar")
            planar.append((node,v))
        local={key:t for key,t in original["tets"].items() if t[0][0] in component}
        boundary,_=_tet_complex(local)
        vertex_groups=[tuple((i,sign) for i in component) for sign in (-1,1)]
        chart_edges=[]
        axes=tuple(i for i in range(3) if i!=drop)
        for u,v in zip(cycle,cycle[1:]+cycle[:1],strict=True):
            edge=tuple(meter.qsub(original["rest"][v][i],original["rest"][u][i]) for i in axes)
            zero=[]
            signs=[]
            for node in component:
                relative=tuple(meter.qsub(original["rest"][node][i],original["rest"][u][i]) for i in axes)
                value=meter.qcross2(edge,relative)
                signs.append(value)
                if value==0: zero.append(node)
            if not any(signs) or (min(signs)<0<max(signs)):
                _fail("unsupported_reference_chart","reference boundary not convex")
            vertex_groups.append(tuple((i,sign) for i in zero for sign in (-1,1)))
            chart_edges.append((u,v,tuple(zip(component,signs,strict=True))))
        groups=[]
        used=set()
        for vertices in vertex_groups:
            key=tuple(sorted(vertices))
            if key in used: continue
            used.add(key)
            selected=tuple(f for f in boundary if set(f)<=set(vertices))
            face_cycle,_=_surface(selected,"disk")
            p,q,r=(points[v] for v in selected[0])
            n=a.cross(a.vsub(q,p),a.vsub(r,p))
            projection=next((i for i,value in enumerate(n) if value!=a.zero),None)
            if projection is None: _fail("invalid_reference","flat boundary face")
            groups.append(BoundaryFaceHint(tuple(tuple(sorted(f)) for f in selected),
                                           face_cycle,projection))
        proof=_cluster(ClusterHint(cid,tuple(sorted(local)),tuple(groups)),
                       original["tets"],points,a,reference=True)
        proofs.append((tuple(planar),tuple(chart_edges),proof))
    separation=[]
    for ia,ib in combinations(range(len(proofs)),2):
        found=None
        for source,target in ((ia,ib),(ib,ia)):
            target_vertices=tuple((i,sign) for i in original["components"][target] for sign in (-1,1))
            for plane_index,plane in enumerate(proofs[source][2].planes):
                facts=tuple(_fact(v,a.dot(plane.normal,a.vsub(points[v],plane.origin)),a)
                            for v in target_vertices)
                if all(sign>0 for sign,_ in facts):
                    found=(source,target,plane_index,tuple(fact for _,fact in facts))
                    break
            if found is not None: break
        if found is None:
            _fail("unsupported_reference_chart","strict component support not proved")
        separation.append(found)
    return (tuple(proofs),tuple(separation),dependencies)


def _basis(original,raw,roots,normals,points,a,*,publish=True):
    _,edges,_,_,_=raw
    m=a.meter
    h={i:{j:(F(0),)*3 for j in sorted({v for fid,face in original["faces"]
                                     if i in face for v in face})}
       for i in original["nodes"]}
    for fid,face in original["faces"]:
        e1,e2=edges[fid]
        derivatives=(m.qvsub(e2,e1),tuple(m.qneg(v) for v in e2),e1)
        for i in face:
            for j,v in zip(face,derivatives,strict=True):
                h[i][j]=m.qvadd(h[i][j],v)
    output=[]
    for i in original["nodes"]:
        factor=a.scale(original["rho"][i],a.inverse(roots[i]))
        projector=tuple(tuple(a.sub(a.rational(F(r==c)),
                                     a.mul(normals[i][r],normals[i][c])) for c in range(3))
                        for r in range(3))
        d=tuple(tuple(a.mul(factor,v) for v in row) for row in projector)
        offsets={}
        for j,v in h[i].items():
            x,y,z=v
            skew=((F(0),m.qneg(z),y),(z,F(0),m.qneg(x)),(m.qneg(y),x,F(0)))
            matrix=[]
            for r in range(3):
                row=[]
                for c in range(3):
                    total=a.zero
                    for k in range(3): total=a.add(total,a.scale(skew[k][c],d[r][k]))
                    row.append(total)
                matrix.append(tuple(row))
            offsets[j]=tuple(matrix)
        for sign in (-1,1):
            exact=[]
            for j in h[i]:
                matrix=tuple(tuple(
                    (a.add if sign==1 else a.sub)(a.rational(F(i==j and r==c)),offsets[j][r][c])
                    for c in range(3)) for r in range(3))
                exact.append(matrix)
            vertex=(i,sign)
            representatives=[]
            for key,t in (sorted(original["tets"].items()) if publish else ()):
                if vertex in t:
                    weights=tuple(a.one if v==vertex else a.zero for v in t)
                    total=a.zero
                    for q in weights: total=a.add(total,q)
                    if total!=a.one: _fail("incomplete_material_cover","vertex barycentric sum")
                    representatives.append((key,weights))
            publication=a.publish(points[vertex],(3,)) if publish else None
            jacobian=(a.publish(tuple(q for matrix in exact for row in matrix for q in row),
                                 (len(exact),3,3)) if publish else None)
            output.append(BoundaryPointBasis(vertex,tuple(h[i]),points[vertex],tuple(exact),
                                            publication,jacobian,tuple(representatives)))
    return tuple(output)

def _support_complex(cluster,active):
    allowed=set(active)
    return tuple(sorted({tuple(sorted(s)) for f in cluster.boundary_facets
                         for size in range(1,4) for s in combinations(f,size)
                         if set(s)<=allowed}))


def _matching_incidence(first,second,mapping,simplices):
    """Complete oriented simplicial chain map; all operations below are ID signs."""
    facets_a={tuple(sorted(f)):f for f in first.boundary_facets}
    facets_b={tuple(sorted(f)):f for f in second.boundary_facets}
    records=[]
    for simplex in simplices:
        target=tuple(sorted(mapping[v] for v in simplex))
        sign=_orientation(tuple(mapping[v] for v in simplex))
        boundary=[]
        if len(simplex)>1:
            for index in range(len(simplex)):
                face=simplex[:index]+simplex[index+1:]
                mapped=tuple(sorted(mapping[v] for v in face))
                image_sign=(1 if index%2==0 else -1)*_orientation(tuple(mapping[v] for v in face))
                omitted=target.index(mapping[simplex[index]])
                target_sign=sign*(1 if omitted%2==0 else -1)
                if image_sign!=target_sign:
                    _fail("unsupported_matching","oriented boundary chain identity")
                boundary.append((face,mapped,image_sign))
        oa=ob=0
        if len(simplex)==3:
            oa=_orientation(facets_a[simplex])
            ob=_orientation(facets_b[target])
            if oa*sign!=-ob:
                _fail("unsupported_matching","opposite occupied patch orientations")
        records.append((simplex,target,sign,tuple(boundary),oa,ob))
    return tuple(records)

def _pair(hint,clusters,points,boundary_strata,a):
    ia,ib=hint.clusters
    if len(hint.supports)>a.limits.max_support_descents:
        _fail("shape_budget","tight support descent bound")
    if ia>=ib or ia not in clusters or ib not in clusters:
        _fail("incomplete_material_cover","canonical cluster pair")
    ca,cb=clusters[ia],clusters[ib]
    va=tuple(sorted({v for plane in ca.planes for v in plane.vertices}))
    vb=tuple(sorted({v for plane in cb.planes for v in plane.vertices}))
    evidence=[]
    normals=[]
    empty=False
    for support in hint.supports:
        normal=tuple(a.element(v) for v in support.normal)
        if all(v==a.zero for v in normal) or support.anchor not in va:
            _fail("unsupported_witness","nonzero support and active owner anchor")
        if normals:
            if len(normals)==1:
                if all(v==a.zero for v in a.cross(normals[0],normal)):
                    _fail("unsupported_matching","dependent support descent")
            elif a.determinant(normals[0],normals[1],normal)==a.zero:
                _fail("unsupported_matching","dependent third descent")
        normals.append(normal)
        origin=points[support.anchor]
        facts_a=tuple(_fact(v,a.dot(normal,a.vsub(points[v],origin)),a) for v in va)
        facts_b=tuple(_fact(v,a.dot(normal,a.vsub(points[v],origin)),a) for v in vb)
        if any(sign>0 for sign,_ in facts_a) or any(sign<0 for sign,_ in facts_b):
            _fail("unsupported_matching","complete vertex support inequality failed")
        za=tuple(v for v,(sign,_) in zip(va,facts_a,strict=True) if sign==0)
        zb=tuple(v for v,(sign,_) in zip(vb,facts_b,strict=True) if sign==0)
        evidence.append(BoundarySupportEvidence(normal,support.anchor,va,vb,
                        tuple(v for _,v in facts_a),tuple(v for _,v in facts_b),za,zb))
        if not za or not zb:
            empty=True
            break
        va,vb=za,zb
    if empty:
        return MatchingIntersectionWitness((ia,ib),tuple(evidence),(),(),"empty")
    mapping=dict(hint.matching)
    if len(mapping)!=len(hint.matching) or set(mapping)!=set(va) or set(mapping.values())!=set(vb) or len(set(mapping.values()))!=len(mapping):
        _fail("unsupported_matching","complete vertex matching bijection")
    for v,w in sorted(mapping.items()):
        if any(a.sub(x,y)!=a.zero for x,y in zip(points[v],points[w],strict=True)):
            _fail("unsupported_matching","exact geometric equality")
    first,second=_support_complex(ca,va),_support_complex(cb,vb)
    mapped={tuple(sorted(mapping[v] for v in s)) for s in first}
    if mapped!=set(second):
        _fail("unsupported_matching","full face-complex incidence bijection")
    pieces=[]
    classes=set()
    for simplex in first:
        target=tuple(sorted(mapping[v] for v in simplex))
        if all(mapping[v]==v for v in simplex):
            classification="material_incidence"
        elif simplex in boundary_strata and target in boundary_strata:
            classification="true_boundary_contact"
        else:
            _fail("forbidden_interior_identification","entire matched simplex is not material boundary")
        pieces.append((simplex,target,classification))
        classes.add(classification)
    classification=("incidence_and_boundary_contact" if len(classes)>1 else next(iter(classes)))
    return MatchingIntersectionWitness((ia,ib),tuple(evidence),tuple(sorted(mapping.items())),
                                       tuple(pieces),classification,
                                       _matching_incidence(ca,cb,mapping,first))


def _fans(original,clusters,exterior,a):
    boundary_by_stratum={}
    for f in original["boundary"]:
        key=tuple(sorted(f))
        for size in range(1,4):
            for simplex in combinations(key,size):
                boundary_by_stratum.setdefault(simplex,[]).append(key)
    hints=dict(exterior)
    if len(hints)!=len(exterior) or set(hints)!=set(boundary_by_stratum):
        _fail("incomplete_material_cover","all ordinary boundary exterior witnesses")
    membership={t:cid for cid,cluster in clusters.items() for t in cluster.tetra_ids}
    fans=[]
    for simplex,incident in sorted(original["strata"].items()):
        cids=tuple(sorted({membership[t] for t in incident}))
        cones=[]
        for cid in cids:
            active=tuple(i for i,p in enumerate(clusters[cid].planes) if set(simplex)<=set(p.vertices))
            if len(active)>a.limits.max_fan_planes: _fail("shape_budget","active fan planes")
            cones.append((cid,active))
        if sum(len(indices) for _,indices in cones)>a.limits.max_fan_planes:
            _fail("shape_budget","total active stratum planes")
        facts=[]
        direction=()
        if simplex in hints:
            direction=tuple(a.element(v) for v in hints[simplex])
            if all(q==a.zero for q in direction):
                _fail("fan_occupancy_violation","zero exterior direction")
            for cid,active in cones:
                found=None
                for index in active:
                    sign,fact=_fact((cid,index),a.dot(clusters[cid].planes[index].normal,direction),a)
                    facts.append(fact)
                    if sign>0:
                        found=index
                        break
                if found is None:
                    _fail("fan_occupancy_violation","no strict exterior of an incident occupied cone")
        fans.append(BoundaryFanWitness(simplex,tuple(sorted(incident)),
                    tuple(sorted(boundary_by_stratum.get(simplex,()))),
                    tuple(cones),direction,tuple(facts)))
    return tuple(fans)


def _coverage(original,clusters,pairs,fans,points):
    by_tet={t:cid for cid,c in clusters.items() for t in c.tetra_ids}
    pair_lookup={p.clusters:p for p in pairs}
    result=[]
    for ta,tb in combinations(sorted(original["tets"]),2):
        shared=tuple(sorted(set(original["tets"][ta]) & set(original["tets"][tb])))
        ca,cb=by_tet[ta],by_tet[tb]
        if ca==cb:
            classification="material_incidence" if shared else "empty"
            pieces=((shared,shared,"material_incidence"),) if shared else ()
            cluster_pair=(ca,ca)
        else:
            cluster_pair=tuple(sorted((ca,cb)))
            pair=pair_lookup[cluster_pair]
            selected=[]
            for left,right,label in pair.intersection_simplices:
                sa,sb=(left,right) if ca<cb else (right,left)
                if set(sa)<=set(original["tets"][ta]) and set(sb)<=set(original["tets"][tb]):
                    selected.append((sa,sb,label))
            pieces=tuple(selected)
            labels={v[2] for v in pieces}
            classification=("empty" if not labels else
                            next(iter(labels)) if len(labels)==1 else "incidence_and_boundary_contact")
            if shared and not any(a==b==shared for a,b,_ in pieces):
                _fail("incomplete_material_cover","cross-cluster shared material simplex")
        result.append(BoundaryPairWitness((ta,tb),shared,cluster_pair,pieces,classification))
    fan_index={f.material_stratum:f for f in fans}
    contacts=[]
    seen=set()
    for pair in pairs:
        mapping=dict(pair.matching)
        for left,right,label in pair.intersection_simplices:
            if label!="true_boundary_contact": continue
            owner=(left,right)
            if owner in seen: _fail("incomplete_material_cover","duplicate contact owner pair")
            seen.add(owner)
            subfaces=tuple((s,tuple(sorted(mapping[v] for v in s))) for n in range(1,len(left))
                           for s in combinations(left,n))
            fa,fb=fan_index[left],fan_index[right]
            if len(contacts)>=original["limits"].max_material_strata:
                _fail("shape_budget","complete contact stratum cap")
            contacts.append(BoundaryContactStratum(owner,tuple(points[v] for v in left),
                           subfaces,(fa.true_boundary_facets,fb.true_boundary_facets),
                           pair.supports,(fa.occupied_cones,fb.occupied_cones)))
    return tuple(result),tuple(sorted(contacts,key=lambda c:c.owners))


def _binding(kind,reference,positions,limits,witness):
    return digest((kind,reference.reference_id,reference.node_ids,reference.face_ids,
                   reference.oriented_nodes,positions,limits,witness))


def _prepare(reference,positions,generators,limits,meter):
    original=_inputs(reference,positions,limits,meter)
    reference_facts=_reference(original,generators,meter,limits)
    raw=_raw(original,original["positions"],meter)
    a,roots,dependencies=_field(generators,raw[3],meter,limits)
    normals,points=_lift(original,original["positions"],raw[2],roots,a)
    return original,reference_facts,raw,a,roots,dependencies,normals,points


def _build_static(reference,positions,hint,limits,meter):
    if type(hint) is not BoundaryWitnessHint:
        _fail("unsupported_witness","bounded untrusted witness_hint required")
    hint.__post_init__()
    if len(hint.clusters)>limits.max_clusters or len(hint.generators)>limits.max_field_rank:
        _fail("shape_budget","tightened witness limits")
    original,reference_facts,raw,a,roots,dependencies,normals,points=_prepare(
        reference,positions,hint.generators,limits,meter)
    ids=[c.cluster_id for c in hint.clusters]
    flat=[t for c in hint.clusters for t in c.tetra_ids]
    if len(ids)!=len(set(ids)) or len(flat)!=len(set(flat)) or set(flat)!=set(original["tets"]):
        _fail("incomplete_material_cover","all tetrahedra partition exactly once")
    clusters={c.cluster_id:_cluster(c,original["tets"],points,a)
              for c in sorted(hint.clusters,key=lambda c:c.cluster_id)}
    required=set(combinations(sorted(clusters),2))
    if len(hint.pairs)!=len(required) or {p.clusters for p in hint.pairs}!=required:
        _fail("incomplete_material_cover","all cluster pairs")
    boundary_strata={tuple(sorted(s)) for f in original["boundary"]
                     for size in range(1,4) for s in combinations(f,size)}
    pairs=tuple(_pair(p,clusters,points,boundary_strata,a)
                for p in sorted(hint.pairs,key=lambda p:p.clusters))
    fans=_fans(original,clusters,hint.exterior,a)
    all_pairs,contacts=_coverage(original,clusters,pairs,fans,points)
    basis=_basis(original,raw,roots,normals,points,a)
    topology=BoundaryTopology(original["nodes"],original["faces"],
        tuple(sorted(original["tets"].items())),original["boundary"],original["components"],
        tuple((s,tuple(sorted(ts))) for s,ts in sorted(original["strata"].items())),reference_facts)
    law=BoundaryLaw(tuple(sorted(raw[0].items())),tuple(sorted(raw[2].items())),
                    tuple(sorted(raw[3].items())),tuple(sorted(normals.items())),raw[4])
    field=BoundaryFieldWitness(a.generators,tuple(sorted(raw[3].items())),dependencies,
                               tuple(range(a.width)),meter.bits)
    receipt=ShellBoundaryAdmission(reference.reference_id,
        _binding("static",reference,positions,limits,hint),limits,positions,field,topology,law,
        tuple(clusters.values()),pairs,all_pairs,fans,contacts,basis,meter.operations,
        meter.operations,0,meter.peak_fraction_bits)
    context=(original,raw,a,roots,normals,points)
    return receipt,context


def certify_shell_boundary(reference,positions,*,witness_hint,limits=BoundaryProofLimits()):
    meter=Meter(limits.strict_limits)
    receipt,_=_build_static(reference,positions,witness_hint,limits,meter)
    before=meter.operations
    from ..validation.shell_boundary import _check_static
    if not _check_static(receipt,reference,positions,witness_hint,limits,meter,pending=True):
        _fail("independent_verification_failed","complete same-counter inner check")
    return replace(receipt,operations=meter.operations,producer_operations=before,
                   checker_operations=meter.operations-before,peak_fraction_bits=meter.peak_fraction_bits)


def _address(original,reference,address,a):
    if type(address) is not ExactBoundaryAddress:
        _fail("invalid_shape_or_identity","separate ExactBoundaryAddress required")
    address.__post_init__()
    if address.reference_id!=reference.reference_id:
        _fail("stale_binding","exact material address")
    simplex=address.minimal_simplex
    if tuple(sorted(simplex))!=simplex or len(set(simplex))!=len(simplex):
        _fail("invalid_shape_or_identity","canonical minimal simplex")
    boundary={tuple(sorted(s)) for f in original["boundary"] for n in range(1,4)
              for s in combinations(f,n)}
    if simplex not in boundary:
        _fail("missing_boundary_owner","minimal simplex is not true global boundary")
    weights=tuple(a.element(q) for q in address.weights)
    total=a.zero
    for q in weights:
        if a.sign(q)<=0: _fail("invalid_shape_or_identity","positive minimal weights")
        total=a.add(total,q)
    if total!=a.one: _fail("invalid_shape_or_identity","exact barycentric normalization")
    expected=tuple((key,tuple(weights[simplex.index(v)] if v in simplex else a.zero for v in t))
                   for key,t in sorted(original["tets"].items()) if set(simplex)<=set(t))
    if address.representatives!=expected:
        _fail("incomplete_material_cover","all exact incident representations")
    for _,qs in expected:
        total=a.zero
        for q in qs: total=a.add(total,q)
        if total!=a.one: _fail("invalid_shape_or_identity","representative normalization")
    return simplex,weights


def _build_point(reference,positions,address,limits,meter):
    if type(address) is not ExactBoundaryAddress:
        _fail("invalid_shape_or_identity","exact material address required")
    original,reference_facts,raw,a,roots,dependencies,normals,points=_prepare(
        reference,positions,address.generators,limits,meter)
    simplex,weights=_address(original,reference,address,a)
    basis=_basis(original,raw,roots,normals,points,a,publish=False)
    lookup={b.vertex:b for b in basis}
    nodes=tuple(sorted({i for v in simplex for i in lookup[v].node_ids}))
    point=[a.zero]*3
    jacobian={i:[[a.zero]*3 for _ in range(3)] for i in nodes}
    for v,weight in zip(simplex,weights,strict=True):
        b=lookup[v]
        for r in range(3): point[r]=a.add(point[r],a.mul(weight,b.exact_point[r]))
        for node,matrix in zip(b.node_ids,b.exact_jacobian,strict=True):
            for r in range(3):
                for c in range(3):
                    jacobian[node][r][c]=a.add(jacobian[node][r][c],a.mul(weight,matrix[r][c]))
    exact=tuple(tuple(tuple(row) for row in jacobian[i]) for i in nodes)
    point=tuple(point)
    point_pub=a.publish(point,(3,))
    jac_pub=a.publish(tuple(q for matrix in exact for row in matrix for q in row),(len(nodes),3,3))
    field=BoundaryFieldWitness(a.generators,tuple(sorted(raw[3].items())),dependencies,
                               tuple(range(a.width)),meter.bits)
    return ShellExactBoundaryPoint(reference.reference_id,
        _binding("point",reference,positions,limits,address),limits,positions,address,field,
        nodes,point,exact,point_pub,jac_pub,reference_facts,
        BoundaryLaw(tuple(sorted(raw[0].items())),tuple(sorted(raw[2].items())),
                    tuple(sorted(raw[3].items())),tuple(sorted(normals.items())),raw[4]),
        meter.operations,meter.operations,0,meter.peak_fraction_bits)


def shell_exact_boundary_point(reference,positions,address,*,limits=BoundaryProofLimits()):
    meter=Meter(limits.strict_limits)
    receipt=_build_point(reference,positions,address,limits,meter)
    before=meter.operations
    from ..validation.shell_boundary import _check_point
    if not _check_point(receipt,reference,positions,address,limits,meter,pending=True):
        _fail("independent_verification_failed","complete map-only inner check")
    return replace(receipt,operations=meter.operations,producer_operations=before,
                   checker_operations=meter.operations-before,peak_fraction_bits=meter.peak_fraction_bits)