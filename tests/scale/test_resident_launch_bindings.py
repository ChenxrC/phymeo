from unittest.mock import patch

import numpy as np
import pytest

import phymeo as ph


def test_alternating_directions_each_keep_their_binding(tyx_mode, tmp_path):
    if tyx_mode != "cuda":
        pytest.skip("device ABI is CUDA-specific")
    from tyx.backends.cuda_runtime import CompiledCudaKernel, capture_cuda_submission_timing

    ball = ph.FreeSphere(2.0,0.125,(0,0,2),gravity=(0,0,-8))
    h=1/128
    with ph.TyxWorldBatch([(77,ball)],"cuda",cache_root=tmp_path) as batch:
        actual=[]
        prepare=CompiledCudaKernel._prepare_reusable

        def count(kernel,args):
            if "candidate" in args and "h" in args:
                src=0 if args["state"] is batch._states[0] else 1
                assert args["candidate"] is batch._states[1-src]
                actual.append((src,args["h"]))
            return prepare(kernel,args)

        state=np.array([2.0,0.0])
        def advance(dt,accel):
            state[1]+=dt*accel
            state[0]+=dt*state[1]
            assert batch.step(dt)=={77:True}
            out=batch.kinematics()
            np.testing.assert_array_equal(out.values[0,[2,5]],state)
            return out

        with patch.object(CompiledCudaKernel,"_prepare_reusable",count):
            for _ in range(6):
                advance(h,-8)
            assert actual==[(0,h),(1,h)]
            batch.set_controls({77:(0,0,4)})
            advance(h,-6)
            assert actual==[(0,h),(1,h)]
            # Direction one first sees the changed scalar. Direction zero retains h.
            advance(2*h,-6)
            advance(h,-6)
            advance(2*h,-6)
            assert actual==[(0,h),(1,h),(1,2*h)]
            with capture_cuda_submission_timing() as native_timing:
                advance(h,-6)
                last=advance(h,-6)
            assert native_timing.arg_pack_calls==native_timing.successful_enqueue_calls==2
            assert last.ticks.tolist()==[12]
        assert batch.stats()["submitted_kernels"]==batch.stats()["synchronizations"]==12


@pytest.mark.parametrize("mode",["interpret","cuda"])
def test_failed_download_retry_restore_and_new_workspace_keep_actual_rows(mode,tmp_path,monkeypatch):
    b=ph.FreeSphere(1.0,0.125,(0,0,3),velocity=(0,1,0),gravity=(0,0,-8))
    worlds=[(31,b),(2,b)]
    h=1/16
    with ph.TyxWorldBatch(worlds,mode,cache_root=tmp_path) as batch:
        assert all(batch.step(h).values())
        kept=batch.snapshot()
        frame=batch.kinematics()
        batch.set_controls({2:(2,0,0)})
        def fail():
            raise RuntimeError("injected acceptance download")
        with monkeypatch.context() as m:
            m.setattr(batch,"_read_status",fail)
            with pytest.raises(RuntimeError,match="acceptance download"):
                batch.step(2*h)
        np.testing.assert_array_equal(batch.kinematics().values,frame.values)
        assert batch.kinematics().ticks.tolist()==[1,1]
        assert all(batch.step(2*h).values())
        accepted=batch.snapshot()
        assert accepted[2].velocity==(0.25,1.0,-1.5)
        batch.restore({31:kept[31]})
        assert batch.snapshot()=={31:kept[31],2:accepted[2]}
        assert all(batch.step(h).values())
        final=batch.snapshot()
        assert final[31].tick==2 and final[2].tick==3
    with pytest.raises(RuntimeError,match="closed"):
        batch.step(h)
    with ph.TyxWorldBatch(worlds,mode,cache_root=tmp_path) as other:
        other.restore(final)
        assert all(other.step(2*h).values())
        assert other.kinematics().ticks.tolist()==[3,4]
        assert other.snapshot()[2].velocity==(0.625,1.0,-3.0)
