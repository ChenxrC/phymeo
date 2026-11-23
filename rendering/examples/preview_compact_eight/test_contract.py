"""Eight actual proposals must keep stratification and balance normalization."""
import importlib
import math
import unittest


class EightProposalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=importlib.import_module("preview_compact_eight.build_source")

    def test_frozen_budget_is_eight_not_a_new_tunable_sweep(self):
        plan=self.m.proposal_layout()
        self.assertEqual(plan["total"],8)
        self.assertEqual(plan["cosine_grid"],(4,2))
        self.assertEqual(plan["mis_cosine_grid"],(2,2))
        self.assertEqual((plan["mis_cosine_count"],plan["mis_environment_count"]),(4,4))
        self.assertEqual(plan["normalization"],.125)

    def test_all_nonmis_strata_are_unique_and_inside_unit_square(self):
        samples=[self.m.proposal(i,(.5,.5),False) for i in range(8)]
        self.assertEqual(len({v["q"] for v in samples}),8)
        self.assertTrue(all(0<q<1 for v in samples for q in v["q"]))
        self.assertEqual({v["q"] for v in samples},{((x+.5)/4,(y+.5)/2) for y in range(2) for x in range(4)})

    def test_mis_has_four_cosine_cells_and_four_distinct_cdf_strata(self):
        values=[self.m.proposal(i,(.5,.5),True) for i in range(8)]
        cosine=[v for v in values if v["kind"]=="cosine"]
        environment=[v for v in values if v["kind"]=="environment"]
        self.assertEqual({v["q"] for v in cosine},{(.25,.25),(.75,.25),(.25,.75),(.75,.75)})
        self.assertEqual([v["cdf_stratum"] for v in environment],[0,1,2,3])
        self.assertTrue(all(v["cdf_strata"]==4 for v in environment))

    def test_balance_mixture_cancels_with_each_discrete_technique(self):
        # Finite directional domain: each proposal distribution sums to one.
        # Expected weighted mean must equal the target cosine integral, with
        # exactly four samples per technique and a final 1/8 factor.
        cosine=(.1,.2,.3,.4);environment=(.4,.3,.2,.1);radiance=(2.,.5,4.,1.)
        actual=0.
        for a,b,value in zip(cosine,environment,radiance):
            actual+=(4*a+4*b)*self.m.balance_weight(a,b)*value/8
        self.assertAlmostEqual(actual,sum(a*v for a,v in zip(cosine,radiance)),places=14)

    def test_ng_clipped_directions_remain_in_fixed_denominator(self):
        radiance=(0.,0.,0.,0.,2.,2.,2.,2.)
        self.assertEqual(sum(radiance)*self.m.proposal_layout()["normalization"],1.)

    def test_zero_pdf_is_zero_and_environment_ratio_is_bounded(self):
        self.assertEqual(self.m.balance_weight(0.,0.),0.)
        for a,b in ((.1,0.),(.2,.9),(1.,1.)):
            self.assertTrue(0<=self.m.balance_weight(a,b)<=2)
        with self.assertRaises(ValueError):self.m.balance_weight(.1,float("nan"))

    def test_bad_indices_and_jitter_never_enter_sampling(self):
        for i,jitter in ((8,(.5,.5)),(-1,(.5,.5)),(True,(.5,.5)),(0,(1.,.5)),(0,(math.nan,.5))):
            with self.subTest(i=i,jitter=jitter),self.assertRaises((ValueError,TypeError)):
                self.m.proposal(i,jitter,True)

    def test_parent_must_be_full_distance_unmodified_compact4(self):
        parent=dict(schema="phymeo-preview-sparse-v1",tile=4,paths_per_anchor=4,area_light_grid=2,
            neighbor_candidates=25,reconstruction_revision=2,estimator="visibility-ratio",bounces=1,
            history="none; one independent exposure per capture",
            compact=dict(schema="phymeo-preview-compact-v1",tile=4,history="none"))
        self.m.validate_parent(parent)
        for key,value in (("paths_per_anchor",16),("local_diffuse",{"radius_m":.3}),("history",{}),("pooled_sky",{})):
            with self.subTest(key=key),self.assertRaises(ValueError):self.m.validate_parent(dict(parent,**{key:value}))

    def test_build_macro_patch_rejects_a_different_budget(self):
        key="source/_native/CMakeLists.txt"
        source="compiler -DPR_PREVIEW_PATHS=4 -DPR_PREVIEW_TILE=4\n"
        self.assertEqual(self.m.patch_text(key,source),"compiler -DPR_PREVIEW_PATHS=8 -DPR_PREVIEW_TILE=4\n")
        for bad in (source.replace("PATHS=4","PATHS=16"),source+source):
            with self.subTest(source=bad),self.assertRaises(ValueError):self.m.patch_text(key,bad)

    def test_manifest_cannot_copy_an_unbound_external_file(self):
        self.assertEqual(self.m.safe_relative("runtime\\phymeo_rendering\\native.py").as_posix(),
                         "runtime/phymeo_rendering/native.py")
        for value in ("../private", "source/../../private", "C:/private", "/source/a", "build/a.dll"):
            with self.subTest(path=value),self.assertRaises(ValueError):self.m.safe_relative(value)


if __name__=="__main__":unittest.main()
