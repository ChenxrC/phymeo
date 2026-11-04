from dataclasses import replace

import numpy as np
import pytest

import phymeo as ph

H=1/128


def sphere(**changes):
    args=dict(mass=2.0,radius=0.125,position=(0,0,0.125),velocity=(0.25,0,0),
              gravity=(0,0,-8),plane_height=0.0,restitution=0.0)
    return ph.SpherePlaneWorld(**{**args,**changes})


def test_real_support_drop_rebound_and_free_neighbour(tyx_mode,tmp_path):
    floor=1/8
    models=[
        (13,sphere()),
        (4,sphere(velocity=(0,0,-1),restitution=0.5)),
        (71,sphere(position=(0,0,floor+3/128),velocity=(0,0,-0.5))),
        (9,ph.FreeSphere(2,floor,(0,0,floor),gravity=(0,0,-8))),
    ]
    with ph.TyxWorldBatch(models,tyx_mode,cache_root=tmp_path) as batch:
        states=batch.snapshot()
        assert states[13].normal_impulse_n_s==0 and not hasattr(states[9],"normal_impulse_n_s")
        impacts=[]
        previous_v=states[71].velocity[2]
        for step in range(1,11):
            assert all(batch.step(H).values())
            states=batch.snapshot()
            supported=states[13]
            assert supported.position==(step*H/4,0,floor) and supported.velocity==(0.25,0,0)
            assert supported.normal_impulse_n_s==2*8*H
            assert states[9].position[2]<floor
            if step==1:
                assert states[4].position[2]==floor and states[4].velocity[2]==17/32
                assert states[4].normal_impulse_n_s==2*(17/32+17/16)
            d=states[71]
            j=2*(d.velocity[2]-previous_v+8*H)
            assert j==d.normal_impulse_n_s
            if j>0: impacts.append(step)
            previous_v=d.velocity[2]
        assert impacts==list(range(5,11)) and states[71].position[2]==floor
        assert batch.stats()["submitted_kernels"]==20
        assert batch.summary()[13]["bottom_z_m"]==0


def test_normal_impulse_is_owned_and_restored_with_actual_surface(tyx_mode,tmp_path):
    with ph.TyxWorldBatch([(1,sphere()),(7,sphere(plane_height=0.25,position=(0,0,0.375)))],tyx_mode,cache_root=tmp_path) as batch:
        assert all(batch.step(H).values())
        before=batch.snapshot()
        batch.set_controls({1:(0,0,-4)})
        assert all(batch.step(H).values())
        loaded=batch.snapshot()
        assert loaded[1].normal_impulse_n_s==20*H and loaded[7].normal_impulse_n_s==16*H
        assert loaded[1].applied_effort==(0,0,-4) and before[1].normal_impulse_n_s==16*H
        batch.restore({1:before[1]})
        assert batch.snapshot()=={1:before[1],7:loaded[7]}
        with pytest.raises(ValueError):
            batch.restore({7:replace(before[1], world=7)})
        with pytest.raises(ValueError):
            batch.restore({1:replace(before[1],normal_impulse_n_s=-1)})
        assert all(batch.step(H).values()) and batch.snapshot()[1].normal_impulse_n_s==16*H


def test_input_floor_and_local_numeric_rejection(tyx_mode,tmp_path):
    with pytest.raises(ValueError):
        sphere(restitution=1.01)
    with pytest.raises(ValueError):
        sphere(plane_height=float("nan"))
    w=sphere(gravity=(0,0,0),velocity=(0,0,-1e-15),restitution=1e-30)
    with ph.TyxWorldBatch([(9,w),(3,sphere())],tyx_mode,cache_root=tmp_path) as batch:
        old=batch.snapshot()
        assert batch.step(H)=={9:False,3:True}
        after=batch.snapshot()
        assert after[9]==old[9] and after[3].tick==1
    # Round the center stop outward: height + radius lies strictly between two FP32s.
    height=2**24
    high=sphere(plane_height=height,gravity=(0,0,0),velocity=(0,0,0),position=(0,0,height))
    with ph.TyxWorldBatch([(1,high)],tyx_mode,cache_root=tmp_path) as batch:
        assert batch.step(H)=={1:True}
        p=batch.snapshot()[1]
        assert p.position[2]==np.nextafter(np.float32(height),np.float32(np.inf))
        assert p.normal_impulse_n_s==0


def test_opening_overlap_releases_without_attractive_impulse(tyx_mode,tmp_path):
    w=sphere(plane_height=-0.25,position=(0,0,-0.25),velocity=(0,0,1),gravity=(0,0,0),restitution=1)
    with ph.TyxWorldBatch([(1,w)],tyx_mode,cache_root=tmp_path) as batch:
        assert batch.step(H)=={1:True}
        s=batch.snapshot()[1]
        assert s.position[2]==-0.125 and s.velocity[2]==1 and s.normal_impulse_n_s==0
        assert batch.step(H)=={1:True}
        assert batch.snapshot()[1].position[2]==-0.125+H
