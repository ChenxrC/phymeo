"""Authored before sources; all collection and execution are deferred by grant."""
import dataclasses
import hashlib
import itertools
import json
from fractions import Fraction as F

import numpy as np
import pytest

from phymeo.constraints.conforming_shell import ConformingShellReference, ShellUnresolved
from phymeo.constraints.shell_boundary import (
    BoundaryFaceHint, BoundaryProofLimits, BoundaryWitnessHint,
    ClusterHint, ClusterPairHint, SupportHint,
)
from phymeo.geometry.shell_boundary import certify_shell_boundary
from phymeo.validation.shell_boundary import verify_shell_boundary

VARIANTS = ("identity", "rotation", "translation", "permutation")


def scalar(q, rank):
    return (F(q),) + (F(0),) * ((1 << rank) - 1)


def vec(values, rank):
    return tuple(scalar(v, rank) for v in values)


def parity(values):
    return -1 if sum(a > b for i, a in enumerate(values) for b in values[i + 1:]) % 2 else 1


def product_tets(face_ids, faces):
    result = {}
    for face_id, face in zip(face_ids, faces, strict=True):
        i, j, k = sorted(map(int, face))
        triples = (
            ((i, -1), (j, -1), (k, -1), (k, 1)),
            ((i, -1), (j, -1), (j, 1), (k, 1)),
            ((i, -1), (i, 1), (j, 1), (k, 1)),
        )
        for stair, vertices in enumerate(triples):
            if parity(tuple(map(int, face))) * (1, -1, 1)[stair] < 0:
                vertices = vertices[:2] + (vertices[3], vertices[2])
            result[(int(face_id), stair)] = vertices
    return result


def facets(tets):
    entries = {}
    for key, vertices in tets.items():
        for omitted in range(4):
            face = tuple(v for i, v in enumerate(vertices) if i != omitted)
            if omitted % 2:
                face = (face[0], face[2], face[1])
            entries.setdefault(tuple(sorted(face)), []).append((face, key))
    return tuple(v[0][0] for _, v in sorted(entries.items()) if len(v) == 1)


def face_hint(boundary, vertices, axis):
    selected = tuple(f for f in boundary if set(f) <= set(vertices))
    counts = {}
    for f in selected:
        for a, b in zip(f, f[1:] + f[:1], strict=True):
            counts.setdefault(tuple(sorted((a, b))), []).append((a, b))
    outgoing = {v[0][0]: v[0][1] for v in counts.values() if len(v) == 1}
    start = min(outgoing)
    cycle = [start]
    while outgoing[cycle[-1]] != start:
        cycle.append(outgoing[cycle[-1]])
        assert len(cycle) <= len(outgoing)
    return BoundaryFaceHint(
        tuple(sorted(tuple(sorted(f)) for f in selected)), tuple(cycle), axis
    )


def transform_vector(v, rotation):
    return tuple(
        tuple(sum(F(int(rotation[i, j])) * v[j][k] for j in range(3))
              for k in range(len(v[0])))
        for i in range(3)
    )


def control(name, variant="identity"):
    """Declared material data/hints, independent of every new geometry routine."""
    rotation = np.eye(3, dtype=np.float64)
    shift = np.zeros(3, dtype=np.float64)
    if variant == "rotation":
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    if variant == "translation":
        shift = np.array([4, -2, 1], dtype=np.float64)
    if name == "C1":
        s = (0, 1, 3, 5, 7, 8)
        columns = ((0, 0), (1, 0), (1, 2), (-1, 2), (-1, 0), (0, 0))
        rest = [(x, y, 0) for x in s for y in (-1, 1)]
        current = [(x, y, z) for x, z in columns for y in (-1, 1)]
        faces = []
        for j in range(5):
            a, b, c, d, m = 2*j, 2*j+2, 2*j+3, 2*j+1, 12+j
            faces.extend(((a, b, m), (b, c, m), (c, d, m), (d, a, m)))
            rest.append(((s[j] + s[j+1])/2, 0, 0))
            current.append(((columns[j][0]+columns[j+1][0])/2, 0,
                            (columns[j][1]+columns[j+1][1])/2))
        face_ids = np.arange(100, 120, dtype=np.int64)
        rank, generators = 2, (F(2), F(5))
        grouped = [tuple(range(100+4*j, 104+4*j)) for j in range(5)]
        panel_normals = ((0, 0, 1), (-1, 0, 0), (0, 0, -1), (1, 0, 0), (0, 0, 1))
        directors = (
            vec((0, 0, 1), 2),
            ((F(0), F(0), F(-2, 5), F(0)), scalar(0, 2),
             (F(0), F(0), F(1, 5), F(0))),
            ((F(0), F(-1, 2), F(0), F(0)), scalar(0, 2),
             (F(0), F(-1, 2), F(0), F(0))),
            ((F(0), F(1, 2), F(0), F(0)), scalar(0, 2),
             (F(0), F(-1, 2), F(0), F(0))),
            ((F(0), F(0), F(2, 5), F(0)), scalar(0, 2),
             (F(0), F(0), F(1, 5), F(0))),
            vec((0, 0, 1), 2),
        )
    else:
        rest = [(-2, -1, 0), (-1, -1, 0), (-1, 1, 0), (-2, 1, 0),
                (-7/8, 1, 0), (1/8, 1, 0), (1/8, 3, 0), (-7/8, 3, 0)]
        current = rest[:4] + [(-1, 1, 0), (0, 1, 0), (0, 3, 0), (-1, 3, 0)]
        faces = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)]
        face_ids = np.array([10, 20, 30, 40], dtype=np.int64)
        rank, generators = 0, ()
        grouped = [(10, 20), (30, 40)]
    node_ids = np.arange(len(rest), dtype=np.int64)
    faces = np.array(faces, dtype=np.int64)
    tets = product_tets(face_ids, faces)
    clusters = []
    for cid, group in enumerate(grouped):
        selected = {key: v for key, v in tets.items() if key[0] in group}
        boundary = facets(selected)
        if name == "C1":
            groups = []
            axis = 2 if cid in (0, 2, 4) else 0
            for face_id in group:
                f = faces[int(face_id)-100]
                for sign in (-1, 1):
                    groups.append((tuple((int(i), sign) for i in f), axis))
            a, b, c, d = 2*cid, 2*cid+2, 2*cid+3, 2*cid+1
            for edge, axis in (((a, b), 1), ((b, c), 0), ((c, d), 1), ((d, a), 0)):
                groups.append((tuple((i, sign) for i in edge for sign in (-1, 1)), axis))
        else:
            ids = tuple(range(4*cid, 4*cid+4))
            groups = [(tuple((i, sign) for i in ids), 2) for sign in (-1, 1)]
            for edge, axis in (((ids[0], ids[1]), 1), ((ids[1], ids[2]), 0),
                               ((ids[2], ids[3]), 1), ((ids[3], ids[0]), 0)):
                groups.append((tuple((i, sign) for i in edge for sign in (-1, 1)), axis))
        changed_groups = []
        for vertices, axis in groups:
            if variant == "rotation" and axis in (0, 1):
                axis = 1-axis
            changed_groups.append(face_hint(boundary, vertices, axis))
        clusters.append(ClusterHint(cid, tuple(sorted(selected)), tuple(changed_groups)))
    pairs = []
    if name == "C1":
        normals = {(0, 1):(1, 0, 2), (1, 2):(-1, 0, 1),
                   (2, 3):(-1, 0, -1), (3, 4):(1, 0, -2),
                   (0, 4):(-1, 0, 0),
                   (0, 2):(0, 0, 1), (0, 3):(-1, 0, 0),
                   (1, 3):(-1, 0, 0), (1, 4):(-1, 0, 0),
                   (2, 4):(0, 0, -1)}
        anchors = {(0, 1):(2, 1), (1, 2):(4, 1), (2, 3):(6, 1),
                   (3, 4):(8, 1), (0, 4):(0, 1),
                   (0, 2):(12, 1), (0, 3):(0, 1),
                   (1, 3):(13, 1), (1, 4):(13, 1), (2, 4):(14, 1)}
        for a, b in itertools.combinations(range(5), 2):
            match = ()
            if b == a+1:
                match = tuple((v, v) for v in
                              ((2*b,-1),(2*b,1),(2*b+1,-1),(2*b+1,1)))
            elif (a, b) == (0, 4):
                match = tuple(((i, sign),(i+10, sign)) for i in (0,1) for sign in (-1,1))
            pairs.append(ClusterPairHint((a,b),
                         (SupportHint(transform_vector(vec(normals[a,b], rank), rotation),
                                      anchors[a,b]),), match))
    else:
        pairs.append(ClusterPairHint((0, 1),
            (SupportHint(transform_vector(vec((1,0,0),0), rotation),(1,1)),
             SupportHint(transform_vector(vec((0,1,0),0), rotation),(2,1))),
            (((2,-1),(4,-1)),((2,1),(4,1)))))
    global_boundary = facets(tets)
    boundary_strata = sorted({tuple(sorted(s)) for f in global_boundary
                              for n in range(1,4) for s in itertools.combinations(f,n)})
    exterior = []
    for simplex in boundary_strata:
        nodes = {v[0] for v in simplex}
        signs = {v[1] for v in simplex}
        if name == "C1":
            if all(i < 12 and i % 2 == 0 for i in nodes):
                direction = vec((0,-1,0),rank)
            elif all(i < 12 and i % 2 == 1 for i in nodes):
                direction = vec((0,1,0),rank)
            elif nodes <= {0,1}:
                direction = vec((-1,0,0),rank)
            elif nodes <= {10,11}:
                direction = vec((1,0,0),rank)
            elif len(signs) == 1:
                sign = next(iter(signs))
                column = {i//2 for i in nodes if i < 12}
                if all(i < 12 for i in nodes) and len(column) == 1:
                    direction = directors[next(iter(column))]
                else:
                    panel = next(j for j, group in enumerate(grouped)
                                 if nodes <= {int(v) for fid in group for v in faces[fid-100]})
                    direction = vec(panel_normals[panel],rank)
                direction = tuple(tuple(sign*q for q in c) for c in direction)
            else:
                raise AssertionError("every ordinary C1 boundary stratum has an authored exterior")
        else:
            component = min(nodes)//4
            local = {i-4*component for i in nodes}
            if len(signs) == 1:
                direction = vec((0,0,next(iter(signs))),0)
            elif local <= {0,1}:
                direction = vec((0,-1,0),0)
            elif local <= {1,2}:
                direction = vec((1,0,0),0)
            elif local <= {2,3}:
                direction = vec((0,1,0),0)
            elif local <= {0,3}:
                direction = vec((-1,0,0),0)
            else:
                raise AssertionError("complete box boundary")
        exterior.append((simplex,transform_vector(direction,rotation)))
    hint = BoundaryWitnessHint(generators, tuple(clusters), tuple(pairs), tuple(exterior))
    rest = np.array(rest,dtype=np.float64) @ rotation.T + shift
    current = np.array(current,dtype=np.float64) @ rotation.T + shift
    if variant == "permutation":
        order = np.arange(len(node_ids)-1,-1,-1)
        node_ids, rest, current = node_ids[order], rest[order], current[order]
        faces, face_ids = faces[::-1], face_ids[::-1]
    ref = ConformingShellReference("boundary-control",node_ids,face_ids,faces,rest,
                                   np.full(len(node_ids),1/32,dtype=np.float64))
    return ref,current,hint,rotation,shift


def payload(value):
    if dataclasses.is_dataclass(value):
        return {"type":type(value).__name__,
                "fields":{f.name:payload(getattr(value,f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value,np.ndarray):
        return {"dtype":str(value.dtype),"shape":list(value.shape),"bytes":value.tobytes().hex()}
    if type(value) is F:
        return {"numerator":str(value.numerator),"denominator":str(value.denominator)}
    if type(value) in (tuple,list):
        return [payload(v) for v in value]
    if type(value) is dict:
        return [[payload(k),payload(v)] for k,v in sorted(value.items())]
    return value


def save(path, value):
    data=json.dumps(payload(value),sort_keys=True,separators=(",",":")).encode()
    path.write_bytes(data)
    path.with_suffix(path.suffix+".sha256").write_text(
        hashlib.sha256(data).hexdigest()+"  "+path.name+"\n",encoding="ascii")

@pytest.mark.parametrize("name",("C1","C2"))
@pytest.mark.parametrize("variant",VARIANTS)
def test_fixed_static_complete_pairs_stars_ties_maps_and_rebuild(name,variant,tmp_path):
    complete=[]
    for rebuild in range(2):
        ref,x,hint,_,_=control(name,variant)
        receipt=certify_shell_boundary(ref,x,witness_hint=hint)
        checked=verify_shell_boundary(receipt,ref,x,witness_hint=hint,limits=BoundaryProofLimits())
        assert checked and checked.verification_kind=="static"
        assert checked.created_cells==0 and 0<checked.operations<=2_000_000
        assert receipt.status=="weak_material_geometry_certified"
        assert receipt.precision=="ideal-from-stored-fp64"
        assert len(receipt.all_pairs)==(1770 if name=="C1" else 66)
        assert len(receipt.full_strata_and_fans)==(371 if name=="C1" else 102)
        assert len(receipt.basis)==(34 if name=="C1" else 16)
        assert sum(len(v.node_ids) for v in receipt.basis)==(178 if name=="C1" else 56)
        assert len(receipt.contacts)==(11 if name=="C1" else 3)
        for item in receipt.basis:
            assert max(item.point.errors+item.jacobian.errors)<F(1,2**40)
            assert not item.point.values.flags.writeable
            assert not item.jacobian.values.flags.writeable
        assert receipt.operations<=2_000_000
        complete.append(payload((receipt,checked)))
        save(tmp_path/f"{name}-{variant}-rebuild{rebuild}.json",(ref,x,hint,receipt,checked))
    assert complete[0]==complete[1]


@pytest.mark.parametrize("eta",(F(1,64),F(1,256)))
def test_c2_remote_overlap_has_positive_exact_volume(eta):
    ref,x,hint,_,_=control("C2")
    for row,node in enumerate(ref.node_ids):
        if node>=4:
            x[row,0]-=float(eta)
            x[row,1]-=float(eta)
    assert 2*F(1,32)*eta*eta>0
    with pytest.raises(ShellUnresolved):
        certify_shell_boundary(ref,x,witness_hint=hint)


@pytest.mark.parametrize("bad",("duplicate_cluster","missing_tet","wrong_face","lost_exterior",
                                "bad_axis","partial_match","duplicate_plane","bad_cycle"))
def test_untrusted_hint_cannot_remove_geometry_obligations(bad):
    ref,x,hint,_,_=control("C1")
    cluster=hint.clusters[0]
    if bad=="duplicate_cluster":
        hint=dataclasses.replace(hint,clusters=hint.clusters+(cluster,))
    elif bad=="missing_tet":
        hint=dataclasses.replace(hint,clusters=(dataclasses.replace(cluster,tetra_ids=cluster.tetra_ids[:-1]),)+hint.clusters[1:])
    elif bad=="wrong_face":
        face=dataclasses.replace(cluster.faces[0],triangles=cluster.faces[1].triangles)
        hint=dataclasses.replace(hint,clusters=(dataclasses.replace(cluster,faces=(face,)+cluster.faces[1:]),)+hint.clusters[1:])
    elif bad=="lost_exterior":
        hint=dataclasses.replace(hint,exterior=hint.exterior[:-1])
    elif bad=="bad_axis":
        p=hint.pairs[0]
        s=dataclasses.replace(p.supports[0],normal=vec((0,0,0),2))
        hint=dataclasses.replace(hint,pairs=(dataclasses.replace(p,supports=(s,)),)+hint.pairs[1:])
    elif bad=="partial_match":
        p=next(p for p in hint.pairs if p.clusters==(0,4))
        hint=dataclasses.replace(hint,pairs=tuple(dataclasses.replace(v,matching=v.matching[:-1]) if v==p else v for v in hint.pairs))
    elif bad=="duplicate_plane":
        hint=dataclasses.replace(hint,clusters=(dataclasses.replace(cluster,faces=cluster.faces+(cluster.faces[0],)),)+hint.clusters[1:])
    else:
        face=dataclasses.replace(cluster.faces[0],cycle=cluster.faces[0].cycle[::-1])
        hint=dataclasses.replace(hint,clusters=(dataclasses.replace(cluster,faces=(face,)+cluster.faces[1:]),)+hint.clusters[1:])
    with pytest.raises((ShellUnresolved,ValueError,TypeError)):
        certify_shell_boundary(ref,x,witness_hint=hint)


@pytest.mark.parametrize("bad",("float32","object","shape","nonfinite","zero_star","inverted"))
def test_input_and_actual_law_failures_are_not_success(bad):
    ref,x,hint,_,_=control("C2")
    if bad=="float32": x=x.astype(np.float32)
    elif bad=="object": x=x.astype(object)
    elif bad=="shape": x=x[:-1]
    elif bad=="nonfinite": x[0,0]=np.nan
    elif bad=="zero_star": x[1]=x[0]
    else: x[[1,3]]=x[[3,1]]
    with pytest.raises((ShellUnresolved,ValueError,TypeError)):
        certify_shell_boundary(ref,x,witness_hint=hint)

def test_nonzero_quality_stars_do_not_justify_negative_actual_tetrahedron():
    ref,x,hint,_,_=control("C1")
    scale=F(1,128)
    rho=F(1,32)
    # All raw area stars scale by scale^2, so normalized directors and quality
    # ratios are unchanged. Tet (100,2) has exact determinant
    # 2*rho*scale*(scale-2*rho/sqrt(5)) < 0, since sqrt(5)<3.
    assert F(3)*F(3)>5
    assert scale<2*rho/F(3)
    with pytest.raises(ShellUnresolved):
        certify_shell_boundary(ref,x*float(scale),witness_hint=hint)


def test_alternate_independent_generator_order_is_a_legal_witness():
    ref,x,hint,_,_=control("C1")
    def swap(vector):
        return tuple((q[0],q[2],q[1],q[3]) for q in vector)
    pairs=tuple(dataclasses.replace(p,supports=tuple(
        dataclasses.replace(s,normal=swap(s.normal)) for s in p.supports)) for p in hint.pairs)
    alternate=dataclasses.replace(hint,generators=(F(5),F(2)),pairs=pairs,
                                  exterior=tuple((s,swap(v)) for s,v in hint.exterior))
    receipt=certify_shell_boundary(ref,x,witness_hint=alternate)
    assert receipt.field.generators==(F(5),F(2))
    assert verify_shell_boundary(receipt,ref,x,witness_hint=alternate,limits=BoundaryProofLimits())
    assert len(receipt.all_pairs)==1770 and len(receipt.contacts)==11
@pytest.mark.parametrize("basis",((F(2),F(8)),(F(1),),(F(2),F(2))))
def test_dependent_or_square_field_basis_is_not_admitted(basis):
    ref,x,hint,_,_=control("C1")
    with pytest.raises(ShellUnresolved):
        certify_shell_boundary(ref,x,witness_hint=dataclasses.replace(hint,generators=basis))


def test_original_shared_vertex_does_not_exempt_other_pair_geometry():
    ref,x,hint,_,_=control("C1")
    # Move only the first fan's centre beyond the opposite material edge. Faces
    # retain their original shared IDs; no adjacency exclusion may hide the
    # resulting star/volume/support failure or convert it to an accepted pair.
    x[12]=np.array([2,0,0],dtype=np.float64)
    with pytest.raises(ShellUnresolved):
        certify_shell_boundary(ref,x,witness_hint=hint)


@pytest.mark.parametrize("cap",("nodes","pairs","strata","clusters","vertices","facets",
                                "polygon","fan","descents","operations","field"))
def test_tightened_caps_return_no_partial_geometry(cap):
    from phymeo.constraints.conforming_shell import ShellProofLimits
    ref,x,hint,_,_=control("C2" if cap!="field" else "C1")
    policy={"nodes":BoundaryProofLimits(strict_limits=ShellProofLimits(max_nodes=3)),
            "pairs":BoundaryProofLimits(strict_limits=ShellProofLimits(max_pairs=1)),
            "strata":BoundaryProofLimits(max_material_strata=1),
            "clusters":BoundaryProofLimits(max_clusters=1),
            "vertices":BoundaryProofLimits(max_cluster_vertices=1),
            "facets":BoundaryProofLimits(max_cluster_boundary_triangles=1),
            "polygon":BoundaryProofLimits(max_face_polygon_vertices=1),
            "fan":BoundaryProofLimits(max_fan_planes=1),
            "descents":BoundaryProofLimits(max_support_descents=1),
            "operations":BoundaryProofLimits(strict_limits=ShellProofLimits(max_scalar_operations=1)),
            "field":BoundaryProofLimits(max_field_rank=1)}[cap]
    with pytest.raises((ShellUnresolved,ValueError,TypeError)):
        certify_shell_boundary(ref,x,witness_hint=hint,limits=policy)

@pytest.mark.parametrize("name,ring,face_z,star_z,remote_weights",(
    ("N1",((1,0,0),(-1,1,0),(-1,-1,0),(2,0,0),(-2,2,0),(-2,-2,0)),
     (1,2,2,4,8,2),(19,3,3,4,6,12,10),(F(3,4),F(1,8),F(1,8))),
    ("N2",((1,0,0),(-1,1,0),(-1,-1,0),(1,0,0),(-1,1,0),(-1,-1,0)),
     (1,2,1,1,2,1),(8,2,3,3,2,3,3),(F(1,2),F(1,4),F(1,4))),
))
def test_connected_n1_n2_positive_local_tets_have_distinct_material_interior_overlap(
        name,ring,face_z,star_z,remote_weights):
    # The original connected six-face fan data are literal, not discovered from
    # a producer. All calculations below use elementary rational determinants.
    rest=((0,0,0),(2,0,0),(1,1,0),(-1,1,0),(-2,0,0),(-1,-1,0),(1,-1,0))
    faces=tuple((0,j,1 if j==6 else j+1) for j in range(1,7))
    current=((0,0,0),)+ring
    rho=F(1,16)
    epsilon=F.from_float(1e-10)
    def sub(a,b): return tuple(F(u)-F(v) for u,v in zip(a,b,strict=True))
    def cross(a,b):
        return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    def dot(a,b): return sum((u*v for u,v in zip(a,b,strict=True)),F(0))
    def mix(vertices,weights):
        return tuple(sum((weight*F(vertices[node][axis]) for node,weight in weights),F(0))
                     for axis in range(3))
    areas=[]
    for index,(i,j,k) in enumerate(faces):
        edges=(sub(current[j],current[i]),sub(current[k],current[i]),sub(current[k],current[j]))
        area=cross(edges[0],edges[1])
        assert area==(F(0),F(0),F(face_z[index])) and area[2]>0
        squared=dot(area,area)
        # Exact frozen shape gate, independently checked on every actual edge.
        assert all(squared>epsilon*epsilon*dot(edge,edge)**2 for edge in edges)
        areas.append(area)
    for node in range(7):
        incident=tuple(i for i,f in enumerate(faces) if node in f)
        star=tuple(sum((areas[i][axis] for i in incident),F(0)) for axis in range(3))
        assert star==(F(0),F(0),F(star_z[node])) and star[2]>0
        assert dot(star,star)>epsilon*epsilon*len(incident)*sum(dot(areas[i],areas[i]) for i in incident)
        for index in incident:
            projection=dot(star,areas[index])
            assert projection>0
            assert projection**2>=F(1,16)*dot(star,star)*dot(areas[index],areas[index])
    # Every actual director is exactly +ez. Every oriented staircase has the
    # same positive determinant 2*rho*c_face, not merely positive face areas.
    tets=product_tets(tuple(range(6)),faces)
    lifted={(node,sign):(F(x),F(y),F(z)+sign*rho)
            for node,(x,y,z) in enumerate(current) for sign in (-1,1)}
    assert len(tets)==18
    for (fid,_),vertices in sorted(tets.items()):
        a,b,c,d=(lifted[v] for v in vertices)
        determinant=dot(sub(b,a),cross(sub(c,a),sub(d,a)))
        assert determinant==2*rho*face_z[fid]>0
    local_weights=(F(1,2),F(1,4),F(1,4))
    assert all(w>0 for w in local_weights+remote_weights)
    assert sum(local_weights)==sum(remote_weights)==1
    first=tuple(zip(faces[0],local_weights,strict=True))
    remote=tuple(zip(faces[3],remote_weights,strict=True))
    # Positive face barycentrics and -rho<z<rho place BOTH material points in
    # the global reference volume interior, not on a local/global boundary.
    assert -rho<F(0)<rho
    assert mix(current,first)==mix(current,remote)==(F(0),F(1,4),F(0))
    assert mix(rest,first)==(F(3,4),F(1,4),F(0))
    expected_remote=(F(-3,8),F(-1,8),F(0)) if name=="N1" else (F(-3,4),F(-1,4),F(0))
    assert mix(rest,remote)==expected_remote!=mix(rest,first)
    # Face3 is a 2x homothety of face0 for N1 and geometrically equal for N2;
    # their common open prism has volume (1/2)*(2*rho)=rho>0.
    scale=F(2) if name=="N1" else F(1)
    assert all(sub(current[b],tuple(scale*F(v) for v in current[a]))==(F(0),)*3
               for a,b in zip(faces[0],faces[3],strict=True))
    assert F(face_z[0],2)*(2*rho)==rho>0
    ref=ConformingShellReference("fan",np.arange(7,dtype=np.int64),
        np.arange(6,dtype=np.int64),np.array(faces,dtype=np.int64),
        np.array(rest,dtype=np.float64),np.full(7,float(rho),dtype=np.float64))
    # Each complete convex cluster is ONE actual positive triangular prism.
    # Put the analytically illicit face0/face3 pair first, so rejection cannot
    # be caused by an earlier unrelated pair candidate or a missing hint.
    ordered_faces=(0,3,1,2,4,5)
    clusters=[]
    for cid,fid in enumerate(ordered_faces):
        local={key:v for key,v in tets.items() if key[0]==fid}
        boundary=facets(local)
        triangle=faces[fid]
        groups=[face_hint(boundary,tuple((v,sign) for v in triangle),2) for sign in (-1,1)]
        for v,w in zip(triangle,triangle[1:]+triangle[:1],strict=True):
            delta=sub(current[w],current[v])
            drop=0 if delta[1]!=0 else 1
            groups.append(face_hint(boundary,tuple((i,s) for i in (v,w) for s in (-1,1)),drop))
        assert len(local)==3 and len(boundary)==8 and len(groups)==5
        clusters.append(ClusterHint(cid,tuple(sorted(local)),tuple(groups)))
    pairs=[]
    for left,right in itertools.combinations(range(6),2):
        fa,fb=faces[ordered_faces[left]],faces[ordered_faces[right]]
        edge=current[fa[2]]
        normal=vec((-edge[1],edge[0],0),0)
        shared=tuple(sorted(set(fa)&set(fb)))
        matching=tuple(((v,s),(v,s)) for v in shared for s in (-1,1))
        pairs.append(ClusterPairHint((left,right),(SupportHint(normal,(0,1)),),matching))
    boundary_strata=sorted({tuple(sorted(s)) for f in facets(tets) for n in range(1,4)
                            for s in itertools.combinations(f,n)})
    exterior=[]
    for simplex in boundary_strata:
        signs={v[1] for v in simplex}
        if len(signs)==1:
            direction=(0,0,next(iter(signs)))
        else:
            nodes={v[0] for v in simplex}
            assert 0 not in nodes  # only the original ring sides are external.
            direction=(sum(current[i][0] for i in nodes),sum(current[i][1] for i in nodes),0)
        exterior.append((simplex,vec(direction,0)))
    hint=BoundaryWitnessHint((),tuple(clusters),tuple(pairs),tuple(exterior))
    assert len(hint.clusters)==6 and len(hint.pairs)==15
    assert sorted(t for c in hint.clusters for t in c.tetra_ids)==sorted(tets)
    # The first A exposed plane has n=(-1,-1,0); it supports ALL A vertices.
    # The independently proved common-interior point has n.p=-1/4, so B
    # necessarily violates the opposite closed support inequality. This is a
    # geometric contradiction, not a missing/zero/malformed separator hint.
    assert hint.pairs[0].clusters==(0,1)
    assert hint.pairs[0].supports[0].normal==vec((-1,-1,0),0)
    assert all(-F(current[v][0])-F(current[v][1])<=0 for v in faces[0])
    assert -F(0)-F(1,4)==F(-1,4)<0
    with pytest.raises(ShellUnresolved) as raised:
        certify_shell_boundary(ref,np.array(current,dtype=np.float64),witness_hint=hint)
    # This stage is reached only after the reference/law and ALL six complete
    # local cluster proofs; a structural hint failure does not satisfy the test.
    assert raised.value.category=="unsupported_matching"
    assert raised.value.detail=="complete vertex support inequality failed"
    # The API result is unresolved. The independent algebra above supplies the
    # overlap fact; no penetration classification is inferred from unresolved.
