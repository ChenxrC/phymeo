"""Saved GPU planes and independent support bookkeeping; no renderer required."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

_SPEC=importlib.util.spec_from_file_location("edge_coverage",Path(__file__).with_name("coverage.py"))
coverage=importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coverage)


def fixture(compact=True,width=13,height=13):
    # oct(0,0,1), with the ordinary 10-bit quantization retained.
    normal=512|(512<<10)
    ids=np.ones((height,width),np.uint32)*158
    depth=np.ones((height,width),np.float32)
    valid=np.ones((height,width),np.uint8)
    attributes=np.zeros((height,width,4),np.float32);attributes[...,:3]=.2;attributes[...,3]=normal
    shape=((height+3)//4,(width+3)//4) if compact else (height,width)
    guide=np.zeros((*shape,4),np.float32);guide[...,:3]=(.1,.2,.3);guide[...,3]=normal
    if compact:guide[...,3]+=1<<20
    else:
        for y in range((height+3)//4):
            for x in range((width+3)//4):guide[min(y*4+2,height-1),min(x*4+2,width-1),3]+=1<<20
    return dict(ids=ids,depth=depth,valid=valid,guide=guide,attributes=attributes if compact else None,
        low_ids=np.ones(shape,np.uint32)*158 if compact else None,
        low_depth=np.ones(shape,np.float32) if compact else None,
        low_valid=np.ones(shape,np.uint8) if compact else None,compact=compact)


class CoverageContracts(unittest.TestCase):
    def test_supported_pixel_has_real_anchors_and_demodulated_average(self):
        data=fixture();row=coverage.inspect_pixel(6,6,**data)
        self.assertEqual(row["classification"],"supported_reconstruction")
        self.assertEqual(row["accepted_count"],9)
        np.testing.assert_allclose(row["incoming_reconstructed_cpu"],[.1,.2,.3],rtol=1e-6)
        self.assertEqual(sum(row["reason_counts"].values()),len(row["candidates"]))

    def test_compact_odd_extent_never_clamps_outside_anchor(self):
        data=fixture(width=13,height=13)
        row=coverage.inspect_pixel(12,12,**data)
        self.assertTrue(any(c["reason"]=="anchor_outside_full_extent" for c in row["candidates"]))
        self.assertTrue(all(c["anchor"][0]!=12 and c["anchor"][1]!=12 for c in row["candidates"] if c["reason"]=="accepted"))
        old=coverage.inspect_pixel(12,12,**fixture(compact=False))
        self.assertTrue(any(c["anchor"]==[12,12] and c["reason"]=="accepted" for c in old["candidates"]))

    def test_zero_support_preserves_own_sh_and_reports_identity_gate(self):
        data=fixture();data["low_ids"][:]=202
        row=coverage.inspect_pixel(6,6,**data)
        self.assertEqual(row["classification"],"geometry_fallback")
        self.assertEqual(row["accepted_count"],0)
        np.testing.assert_allclose(row["incoming_reconstructed_cpu"],[.2]*3)
        self.assertGreater(row["reason_counts"]["low_identity"],0)

    def test_low_hit_must_match_actual_full_anchor(self):
        for key,value,reason in (("ids",202,"full_anchor_identity"),("depth",1.02,"low_full_depth")):
            data=fixture();data[key][2::4,2::4]=value
            # Receiver is not itself an edited anchor.
            row=coverage.inspect_pixel(5,5,**data)
            self.assertEqual(row["accepted_count"],0)
            self.assertGreater(row["reason_counts"][reason],0)

    def test_receiver_gate_and_untraced_cannot_be_silently_shared(self):
        data=fixture();data["depth"][5,5]=1.02
        row=coverage.inspect_pixel(5,5,**data)
        self.assertEqual(row["accepted_count"],0)
        self.assertGreater(row["reason_counts"]["receiver_depth"],0)
        data=fixture();data["guide"][...,3]-=1<<20
        row=coverage.inspect_pixel(6,6,**data)
        self.assertEqual(row["accepted_count"],0)
        self.assertGreater(row["reason_counts"]["not_traced"],0)

    def test_normal_gate_is_not_an_instance_or_roi_exception(self):
        data=fixture();data["guide"][...,3]=(1023|(512<<10))+(1<<20)
        row=coverage.inspect_pixel(6,6,**data)
        self.assertEqual(row["accepted_count"],0)
        self.assertGreater(row["reason_counts"]["low_full_normal"],0)
        a=fixture();b=fixture();b["ids"][:]=77;b["low_ids"][:]=77
        self.assertEqual(coverage.inspect_pixel(6,6,**a)["accepted_count"],coverage.inspect_pixel(6,6,**b)["accepted_count"])

    def test_error_is_signed_and_zero_count_does_not_become_zero_error(self):
        rgb=np.array([[128,10,10],[40,40,40]],np.uint8)
        ref=np.array([[80,10,10],[40,40,40]],np.uint8)
        score=coverage.display_error(rgb,ref)
        self.assertGreater(score["signed_rgb"][0],0)
        self.assertGreater(score["signed_luminance"],0)
        self.assertIsNone(coverage.display_error(rgb[:0],ref[:0]))
        self.assertEqual(coverage.display_error(ref,ref)["rmse"],0)

    def test_reveal_requires_previous_blocker_and_current_finite_foreground(self):
        before=dict(ids=np.array([[202,202,202,9]],np.uint32),depth=np.array([[1,1,1,1]],np.float32),valid=np.ones((1,4),np.uint8))
        after=dict(ids=np.array([[158,0,158,158]],np.uint32),depth=np.array([[2,np.inf,.5,2]],np.float32),valid=np.array([[1,0,1,1]],np.uint8))
        np.testing.assert_array_equal(coverage.reveal_mask(before,after,blocker=202),[[True,False,False,False]])


if __name__=="__main__":unittest.main()
