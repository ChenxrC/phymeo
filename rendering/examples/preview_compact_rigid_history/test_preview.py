"""Offline preview evidence must not disguise stale state or unequal sampling."""
from copy import deepcopy
import importlib
import unittest


class PreviewContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.p=importlib.import_module("preview_compact_rigid_history.preview")

    def record(self,index=0):
        return dict(pose_index=index,frame_key=dict(tick=53+54*index),time_s=.033125+.03375*index,
            state_sha256="a"*64,camera_sha256="b"*64,rgb_sha256="c"*64,
            sensors={"depth":"d"*64},effective_seed=index,current_sha256="e"*64,
            history_actual=dict(sequence=index,last_reset=index==0))

    def test_full_movie_retains_every_prior_presented_pose(self):
        self.assertEqual(self.p.POSES,tuple(range(300)))
        self.assertEqual(self.p.WARMUP,tuple(range(10)))
        self.assertEqual(self.p.POSES[55:61],(55,56,57,58,59,60))

    def test_same_method_pair_allows_changed_rgb_but_not_geometry_or_seed(self):
        a=self.record();b=deepcopy(a);b["rgb_sha256"]="f"*64
        self.p.verify_pair([a],[b])
        for key,value in (("effective_seed",99),("sensors",{}),("current_sha256","0"*64),
                          ("camera_sha256","1"*64),("state_sha256","2"*64)):
            bad=dict(b,**{key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):self.p.verify_pair([a],[bad])

    def test_reference_needs_distinct_nonzero_seed_and_full_budget(self):
        good=dict(schema="phymeo-preview-sparse-reference-v1",status="complete",seed=1,batches=512,
                  primary_proposals_per_pixel=8192)
        self.p.verify_reference_header(good)
        for key,value in (("seed",0),("batches",64),("status","partial_failure"),("primary_proposals_per_pixel",1024)):
            with self.subTest(key=key),self.assertRaises(ValueError):self.p.verify_reference_header(dict(good,**{key:value}))

    def test_sources_ignore_only_relocatable_uri(self):
        original={"uri":"one","sha256":"a"*64,"roughness":.3,"nested":[{"uri":"two","x":1}]}
        moved=deepcopy(original);moved["uri"]="other";moved["nested"][0]["uri"]="else"
        self.assertEqual(self.p.clean_source(original),self.p.clean_source(moved))
        moved["roughness"]=.4
        self.assertNotEqual(self.p.clean_source(original),self.p.clean_source(moved))

    def test_presentation_order_and_actual_seed_are_not_inferred_from_physics_tick(self):
        values=[self.record(i) for i in range(3)]
        self.p.verify_chain(values,expected_indices=(0,1,2))
        with self.assertRaises(ValueError):self.p.verify_chain(values,expected_indices=(0,2,3))
        values[2]["history_actual"]["sequence"]=1
        with self.assertRaises(ValueError):self.p.verify_chain(values,expected_indices=(0,1,2))

    def test_reset_is_explicit_in_actual_sequence(self):
        values=[self.record(i) for i in range(3)]
        values[2]["history_actual"]=dict(sequence=0,last_reset=True)
        self.p.verify_chain(values,expected_indices=(0,1,2))
        values[2]["history_actual"]["last_reset"]=False
        with self.assertRaises(ValueError):self.p.verify_chain(values,expected_indices=(0,1,2))

    def test_full_binding_rejects_missing_or_duplicate_presentations(self):
        value={"actual_full_inputs":[{"pose_index":i} for i in range(300)]}
        self.p.registered_source(value)
        for records in (value["actual_full_inputs"][:-1],value["actual_full_inputs"]+[value["actual_full_inputs"][0]]):
            with self.assertRaises(ValueError):self.p.registered_source({"actual_full_inputs":records})

    def test_reset_pair_requires_identical_rgb(self):
        a=self.record();b=deepcopy(a);b["rgb_sha256"]="f"*64
        with self.assertRaises(ValueError):self.p.verify_pair([a],[b],require_reset_rgb=True)


if __name__=="__main__":unittest.main()
