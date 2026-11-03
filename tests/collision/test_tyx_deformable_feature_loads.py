"""Explicit TYX geometry, native normal site law and normal public particle time integration."""
import json
import numpy as np
import pytest
import phymeo as ph
from test_deformable_feature_loads import conservation,dynamic,force_map,scene,system

pytestmark=pytest.mark.tyx


def test_tyx_fold_scatter_and_failed_force_evaluation_are_transactional(tyx_mode,tmp_path,monkeypatch):
    m,b,state,nodes,_=scene("DD-REMOTE-FOLD",dtype=np.float32)
    with ph.CollisionWorkspace(m,tyx_mode,cache_root=tmp_path/"cache") as work,ph.TyxNormalLayer(tyx_mode,cache_root=tmp_path/"cache") as op:
        s=system(m,b,layer_operator=op)
        loads=s.sample_particles(state,work)
        assert len(loads.contacts)==32 and loads.force_backend==tyx_mode
        stats=op.stats()
        assert stats["completed_launches"]==1 and stats["sites"]==38
        if tyx_mode=="cuda": assert stats["downloads"]>0 and work.stats()["downloads"]>0
        f=force_map(nodes,loads.nodal_force_n)
        lower=sum(f[i][2] for i in range(100,106))
        assert lower==pytest.approx(-64.,abs=1e-5)
        conservation(loads,state,3e-5)
        monkeypatch.setattr(op,"evaluate",lambda rows:np.full((len(rows),5),np.nan))
        with pytest.raises(ph.SolverFailure,match="force"):
            s.sample_particles(state,work,features=loads.geometry)
        assert s.last_sample is loads
    (tmp_path/"feature-forces-device.json").write_text(json.dumps(dict(backend=tyx_mode,contacts=32,sites=38,
         lower_n=lower,node_forces_n=loads.nodal_force_n.tolist(),stored_j=loads.potential_j,stats=stats),allow_nan=False),encoding="utf-8")


def test_original_tyx_step_changes_both_material_velocities_and_separation(tyx_mode,tmp_path):
    m,b,state,nodes,_=scene("DD-VT-TWO-SHEETS",dtype=np.float32)
    with ph.CollisionWorkspace(m,tyx_mode,cache_root=tmp_path/"cache") as work,ph.TyxNormalLayer(tyx_mode,cache_root=tmp_path/"cache") as op,ph.TyxSemiImplicitSolver(tyx_mode,cache_root=tmp_path/"cache") as solver:
        s=system(m,b,layer_operator=op)
        obs=dynamic(s,m,b,state,nodes,work,solver)
        assert obs["backend"]=="tyx:"+tyx_mode and op.stats()["completed_launches"]==2
        obs["layer"]=op.stats()
    (tmp_path/"feature-step-device.json").write_text(json.dumps(obs,allow_nan=False),encoding="utf-8")
