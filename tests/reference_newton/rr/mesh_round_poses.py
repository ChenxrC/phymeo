"""Read back Newton 1.7 native mesh/round contacts for the independently frozen poses."""

import argparse
import json
import runpy
from importlib.metadata import version
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def geometry(kind):
    if kind=="u":
        return runpy.run_path(str(HERE/"baseline.py"))["_u_geometry"]([1,1,1],False)
    vertices=[(x,y,z) for z in (-1,1) for x,y in ((-1,-1),(1,-1),(1,1),(-1,1))]
    quads=((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
    faces=[tri for a,b,c,d in quads for tri in ((a,b,c),(a,c,d))]
    return np.asarray(vertices,dtype=np.float32),np.asarray(faces,dtype=np.int32).ravel()


def measure(case,device):
    import newton
    import warp as wp

    axis=np.asarray(case["axis_m"],dtype=float)
    center=axis.mean(0)
    direction=axis[-1]-axis[0]
    length=float(np.linalg.norm(direction))
    with wp.ScopedDevice(device):
        builder=newton.ModelBuilder(up_axis=newton.Axis.Z)
        cfg=builder.ShapeConfig(margin=0.0,gap=0.0,has_particle_collision=False)
        body_m=builder.add_body()
        vm,fm=geometry(case["mesh"])
        mesh=builder.add_shape_mesh(body_m,mesh=newton.Mesh(vm,fm),cfg=cfg)
        body_r=builder.add_body(xform=wp.transform(center,wp.quat_identity()))
        if len(axis)==1:
            round_id=builder.add_shape_sphere(body_r,radius=case["radius_m"],cfg=cfg)
            kind="SPHERE"
        else:
            q=wp.quat_identity()
            if length:
                unit=direction/length
                # All stored fixtures are positive principal axes. Build the
                # actual quaternion rotating a native capsule's local +Z onto it.
                if unit[2]<-1+1e-12:
                    q=wp.quat(1,0,0,0)
                else:
                    quat=np.r_[np.cross([0.,0.,1.],unit),1+unit[2]]
                    q=wp.quat(*(quat/np.linalg.norm(quat)))
            round_id=builder.add_shape_capsule(body_r,radius=case["radius_m"],half_height=length/2,
                xform=wp.transform((0,0,0),q),cfg=cfg)
            kind="CAPSULE"
        model=builder.finalize(device=device)
        types=[newton.GeoType(int(x)).name for x in model.shape_type.numpy()[[mesh,round_id]]]
        assert types==["MESH",kind]
        bodies=model.shape_body.numpy()[[mesh,round_id]]
        masses=model.body_mass.numpy()[bodies]
        assert len(set(map(int,bodies)))==2 and np.all(masses>0)
        pipeline=newton.CollisionPipeline(model,broad_phase="nxn",reduce_contacts=False,rigid_contact_max=512)
        state=model.state()
        contacts=pipeline.contacts()
        pipeline.collide(state,contacts)
        ds=wp.empty(contacts.rigid_contact_max,dtype=float,device=device)
        pts=[wp.empty(contacts.rigid_contact_max,dtype=wp.vec3,device=device) for _ in range(2)]
        newton.eval_rigid_contact_kinematics(model,state,contacts,out_distance=ds,
                 out_point0_world=pts[0],out_point1_world=pts[1])
        wp.synchronize_device(device)
        count=int(contacts.rigid_contact_count.numpy()[0])
        assert 0<=count<contacts.rigid_contact_max and str(contacts.rigid_contact_count.device)==str(device)
        fields=[arr.numpy()[:count] for arr in (contacts.rigid_contact_shape0,contacts.rigid_contact_shape1,
                ds,contacts.rigid_contact_normal,*pts)]
        rows=[]
        for i,(a,b,d,n,p,q) in enumerate(zip(*fields,strict=True)):
            assert {int(a),int(b)}=={mesh,round_id}
            if int(a)!=mesh:
                n,p,q=-n,q,p
            assert np.isfinite([d,*n,*p,*q]).all()
            foot=center if not length else axis[0]+direction*np.clip((q-axis[0])@direction/length**2,0,1)
            rows.append(dict(index=i,native_shape_ids=[int(a),int(b)],gap_m=float(d),normal_mesh_to_round=n.tolist(),
                mesh_point_m=p.tolist(),round_point_m=q.tolist(),
                round_skin_error_m=float(abs(np.linalg.norm(q-foot)-case["radius_m"]))))
    tolerance=2e-4
    comparable=case["relation"] in ("separated","boundary_overlap") and case["id"]!="capsule-pierces"
    checks=None
    if comparable and case["relation"]=="separated":
        checks=(count==0)
    elif comparable:
        checks=False
        if rows:
            lowest=min(row["gap_m"] for row in rows)
            deep=[row for row in rows if row["gap_m"]<=lowest+tolerance]
            checks=all(abs(row["gap_m"]-case["axis_gap_m"])<=tolerance and row["round_skin_error_m"]<=tolerance
                       and np.linalg.norm(np.subtract(row["normal_mesh_to_round"],case["normal_mesh_to_round"]))<=tolerance
                       and ("mesh_point_m" not in case or np.linalg.norm(np.subtract(row["mesh_point_m"],case["mesh_point_m"]))<=tolerance)
                       for row in deep)
    return dict(id=case["id"],shape_types=types,dynamic_masses_kg=masses.tolist(),count=count,
        analytic_relation=case["relation"],axis_gap_m=case["axis_gap_m"],boundary_gap_m=case["boundary_gap_m"],
        comparable_pose_pass=checks,contacts=rows)


def main():
    import sys
    import warp as wp

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    assert version("newton")=="1.7.0"
    wp.config.kernel_cache_dir=str(Path(sys.prefix)/"warp-kernels")
    wp.init()
    device=wp.get_device(args.device)
    cases=json.loads((HERE/"mesh_round_cases.json").read_text("utf-8"))["poses"]
    results=[measure(c,device) for c in cases]
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(dict(newton=version("newton"),warp=version("warp-lang"),device=str(device),
                       shape_gap_m=0,shape_margin_m=0,results=results),stream,allow_nan=False,indent=2)
    for row in results:
        print(row["id"],row["count"],row["comparable_pose_pass"])
    return 1 if any(r["comparable_pose_pass"] is False for r in results) else 0


if __name__=="__main__":
    raise SystemExit(main())
