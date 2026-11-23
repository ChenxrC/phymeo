"""Analytic controls for ratio bias/bounds, not a claim of GPU validation."""
import unittest
from .ratio_reference import estimate_ratio


class RatioTests(unittest.TestCase):
    def test_open_is_existing_sh_for_colored_nonuniform_sky(self):
        sh=(.31,.62,.13)
        samples=[((1.,.2,4.),.25,"miss",(0.,0.,0.)),((.1,3.,.7),1.8,"miss",(0.,0.,0.))]
        got=estimate_ratio(sh,samples)
        self.assertEqual(got["ratio"],(1.,1.,1.))
        self.assertEqual(got["incoming"],sh)

    def test_fully_blocked_removes_sky_but_keeps_colored_hit(self):
        got=estimate_ratio((10.,4.,2.),[((5.,2.,3.),1.,"hit",(3.,.1,.2)),((1.,3.,2.),1.,"hit",(1.,.2,.1))])
        self.assertEqual(got["ratio"],(0.,0.,0.))
        self.assertEqual(got["incoming"],(2.,.15000000000000002,.15000000000000002))

    def test_black_environment_is_original_weighted_hit_mean(self):
        got=estimate_ratio((0.,0.,0.),[((0.,0.,0.),.5,"hit",(2.,4.,6.)),((0.,0.,0.),1.5,"miss",(0.,0.,0.))])
        self.assertEqual(got["incoming"],(.5,1.,1.5))

    def test_zero_channel_denominator_uses_same_weight_visibility(self):
        got=estimate_ratio((1.,2.,3.),[((0.,2.,0.),.5,"miss",(0.,0.,0.)),((0.,1.,0.),1.5,"hit",(0.,0.,0.))])
        self.assertEqual(got["ratio"],(.25,.4,.25))
        self.assertEqual(got["incoming"],(.25,.8,.75))

    def test_geometric_clip_does_not_renormalize_into_visibility(self):
        got=estimate_ratio((2.,2.,2.),[((1.,1.,1.),1.,"miss",(0.,0.,0.)),((1.,1.,1.),1.,"clipped",(100.,0.,0.))])
        self.assertEqual(got["incoming"],(1.,1.,1.))

    def test_zero_weight_and_zero_input_remain_zero(self):
        self.assertEqual(estimate_ratio((0.,0.,0.),[((0.,0.,0.),0.,"miss",(0.,0.,0.))])["incoming"],(0.,0.,0.))
        self.assertEqual(estimate_ratio((1.,2.,3.),[((0.,0.,0.),0.,"clipped",(0.,0.,0.))])["ratio"],(0.,0.,0.))

    def test_no_arbitrary_small_energy_cutoff(self):
        got=estimate_ratio((.1,.2,.3),[((1e-30,2e-30,3e-30),1.,"miss",(0.,0.,0.))])
        self.assertEqual(got["incoming"],(.1,.2,.3))

    def test_nonfinite_or_negative_input_rejected(self):
        for bad in (-1.,float("inf"),float("nan")):
            with self.assertRaises(ValueError):
                estimate_ratio((1.,1.,1.),[((1.,1.,1.),bad,"miss",(0.,0.,0.))])
        with self.assertRaises(ValueError):estimate_ratio((1.,1.,1.),[])


if __name__=="__main__":unittest.main()
