"""Real TYX interpolation of eight grid nodes, local derivatives and row completion tags."""

import json

import numpy as np
import pytest

from test_sdf_grid import api, external, polynomial

pytestmark=pytest.mark.tyx


def sampler():
    from phymeo.backends.tyx_sdf import TyxSdfSampler
    return TyxSdfSampler


def test_tyx_batch_samples_external_grid_with_persistent_upload(tyx_mode,tmp_path,monkeypatch):
    grid,_=external()
    local=np.array([[-0.125,0.8125,-0.375],[0.5,1.375,0.25],[0,1,-0.25],
                    [np.nextafter(0.5,np.inf),1,0]])
    ref=grid.sample_local(local)
    monkeypatch.setattr(type(grid),"sample_local",lambda *_:pytest.fail("device path called HOST interpolation"))
    shift=np.array([2,-3,0.5])
    with sampler()(grid,mode=tyx_mode,capacity=4,cache_root=tmp_path/"cache") as work:
        rows=work.sample_local(local)
        world=work.sample_world(np.roll(local[:3],1,axis=1)+shift,translation_m=shift,orientation_xyzw=(0.5,0.5,0.5,0.5))
        assert work.sample_local([])==()
        with pytest.raises(ValueError,match="capacity"):
            work.sample_local(np.zeros((5,3)))
        stats=work.stats()
    assert [r.status for r in rows]==["valid","valid","valid","outside_grid"]
    assert [r.cell_xyz for r in rows]==[r.cell_xyz for r in ref]
    for a,b,c,p in zip(rows[:3],ref[:3],world,local[:3],strict=True):
        v,g=polynomial(p)
        assert a.distance_m==pytest.approx(float(v),abs=3e-6) and b.distance_m==pytest.approx(c.distance_m,abs=3e-6)
        np.testing.assert_allclose(a.gradient,g,rtol=0,atol=3e-6)
        np.testing.assert_allclose(c.gradient,np.roll(g,1),rtol=0,atol=3e-6)
    assert stats["completed_launches"]==2 and stats["samples"]==7
    if tyx_mode=="cuda":
        assert stats["grid_uploads"]==1 and stats["grid_uploaded_bytes"]==8*grid.values_zyx_m.size
        assert stats["query_uploaded_bytes"]==16*7 and stats["downloaded_bytes"]==20*7
        assert stats["synchronizations"]==2
    (tmp_path/"tyx-sdf.json").write_text(json.dumps(dict(rows=[dict(status=r.status,value=r.distance_m,
        gradient=r.gradient,cell_xyz=r.cell_xyz) for r in rows],stats=stats),allow_nan=False),encoding="utf-8")


def test_tyx_band_flat_gradient_and_failed_kernel_close_the_workspace(tyx_mode,tmp_path,monkeypatch):
    cls,_=api()
    v=np.broadcast_to(np.array([-1,-0.25,-0.25,0.875]),(2,2,4))
    grid=cls(v,origin_m=(0,0,0),spacing_m=(0.5,0.75,1.125),band_m=0.75)
    with sampler()(grid,mode=tyx_mode,capacity=4,cache_root=tmp_path/"cache") as work:
        rows=work.sample_local([[0.25,0.4,0.5],[0.75,0.4,0.5],[1.25,0.4,0.5],[2,0,0]])
        assert [r.status for r in rows]==["outside_band","valid","outside_band","outside_grid"]
        assert rows[1].distance_m==-0.25 and rows[1].gradient==(0,0,0)
        monkeypatch.setattr(work,"_launch",lambda *args:None)
        with pytest.raises(RuntimeError,match="incomplete"):
            work.sample_local([[0.75,0.4,0.5]])
        assert work.closed
        with pytest.raises(RuntimeError,match="closed"):
            work.sample_local([[0.75,0.4,0.5]])
        assert work.stats()["completed_launches"]==1
