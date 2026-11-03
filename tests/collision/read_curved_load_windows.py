"""Independently re-read both saved smooth-contact trajectories without importing Phymeo."""

import argparse
import hashlib
import json
from math import dist, sqrt
from pathlib import Path


def dot(a,b):
    return sum(x*y for x,y in zip(a,b,strict=True))


def inspect(folder,mode,family):
    label="tr" if mode=="host" else "on"
    ordinal=0 if family=="RR" else 1
    name="impact" if family=="RR" else "sliding"
    fn=folder/f"temp-smooth-windows-{mode}"/f"test_curved_contact_windows_{label}{ordinal}"/f"curved-{name}-window.json"
    payload=fn.read_bytes()
    data=json.loads(payload)
    assert data["mode"]==mode and data["family"]==family and data["step_s"]==1/256
    rows=data["rows"]; steps=80 if family=="RR" else 48
    assert [row["step"] for row in rows]==list(range(1,steps+1))
    target=[.25,0.,0.] if family=="RR" else [0.,.75,0.]
    cloth={91:[-2,-1,0],71:[2,-1,0],38:[0,2,0]}
    maxima=dict(linear_momentum_error_ns=0.,material_point_error_m=0.,force_magnitude_error_n=0.)
    forces=[]; active=[]
    for i,row in enumerate(rows,1):
        pressure=0.
        if family=="RR":
            a,b=row["end_velocity_m_s"]
            momentum=[.5*x+y for x,y in zip(a,b,strict=True)]
            xyz=row["end_positions_m"]
            assert abs(xyz[1][0]-xyz[0][0]-.375-row["end_gap_m"])<3e-7
        else:
            target=[p+x/256 for p,x in zip(target,row["external_force_n"][0],strict=True)]
            rv=row["rigid_velocities_m_s"][0]
            momentum=[.5*x+sum(v[j] for v in row["cloth_velocities_m_s"]) for j,x in enumerate(rv)]
        error=dist(momentum,target)
        maxima["linear_momentum_error_ns"]=max(error,maxima["linear_momentum_error_ns"])
        assert error<(7e-4 if mode!="host" else 2e-8)
        assert len(row["contacts"])<=1
        for contact in row["contacts"]:
            active.append(i)
            n=contact["normal"]
            assert contact["first"]["body"]==1 and contact["second"]["body"]==2
            assert contact["first"]["world"]==contact["second"]["world"]==6
            assert abs(dot(n,n)-1)<1e-5
            assert abs(contact["force_n"]-64*(1/16-contact["gap_m"]))<3e-5
            for site in contact["sites"]:
                fs=site["force_on_second_n"]
                projected=dot(n,fs)
                d=abs(projected-site["force_n"])
                maxima["force_magnitude_error_n"]=max(d,maxima["force_magnitude_error_n"])
                assert d<3e-5
                pressure+=projected
                if family=="RD":
                    tangent=[v-projected*f for v,f in zip(fs,n,strict=True)]
                    assert sqrt(dot(tangent,tangent))<=.25*projected+2e-5
                    assert site["application_point_first_m"]==site["application_point_second_m"]
                    address=site["material"]["second_weights"]
                    recovered=[sum(w*cloth[k][j] for k,w in address) for j in range(3)]
                    d=dist(recovered,site["material"]["second_point"])
                    maxima["material_point_error_m"]=max(d,maxima["material_point_error_m"])
                    assert d<3e-7
        assert abs(pressure-(row["entry_force_n"] if family=="RR" else row["entry_normal_n"]))<3e-5
        forces.append(pressure)
        if family=="RD":
            cloth=dict(zip((91,71,38),row["cloth_positions_m"],strict=True))
    assert active==list(range(active[0],active[-1]+1)) and active[-1]<steps
    result=dict(mode=mode,family=family,steps=steps,first_loaded_step=active[0],
                last_loaded_step=active[-1],peak_step=1+forces.index(max(forces)),peak_n=max(forces),
                end_momentum=momentum,stored_sha256=hashlib.sha256(payload).hexdigest(),**maxima)
    if family=="RR":
        result.update(min_core_gap_m=min(r["end_gap_m"] for r in rows),max_energy_excursion_j=max(abs(r["mechanical_energy_j"]-.0625) for r in rows))
    else:
        result.update(first_contact_power_w=rows[0]["contact_power_w"],final_omega_rad_s=rows[-1]["end_omega_rad_s"][0])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    results=[inspect(args.evidence,m,f) for m in ("host","interpret","cuda") for f in ("RR","RD")]
    with args.out.open("x",encoding="utf8") as out:
        json.dump(results,out,indent=2,allow_nan=False)
    print(json.dumps(results,allow_nan=False))


if __name__=="__main__":
    main()
