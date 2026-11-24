"""Offline preview evidence must not disguise stale state or unequal sampling."""
from copy import deepcopy
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


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

    def test_presentation_defaults_preserve_old_namespace_and_cli(self):
        self.assertEqual(self.p.presentation_options(SimpleNamespace()),("detail","recorded"))
        argv=["--prepared","p","--source-inputs","i","--delivery","d","--output","o"]
        self.assertEqual(self.p.presentation_options(self.p.parse_args(argv)),("detail","recorded"))
        self.assertEqual(self.p.presentation_options(self.p.parse_args(argv+[
            "--camera","sequence","--studio-look","soft-studio"])),("sequence","soft-studio"))

    def test_new_presentation_reuses_declared_factory_and_camera_helpers(self):
        frames=tuple(range(300));scene=object();studio=object();look_scene=object();look_studio=object()
        declaration={"extent":[1280,720],"exposure":8.,"settings":{"area_light_grid":2}}
        factory=SimpleNamespace(bind_scene=Mock(return_value=((object(),scene),frames,studio,"trace")))
        shared=SimpleNamespace(apply_look=Mock(return_value=(look_scene,look_studio)),
            base=SimpleNamespace(preset=Mock(return_value=declaration),make_inputs=Mock(return_value="inputs")))
        for camera,look in (("detail","recorded"),("sequence","soft-studio")):
            with self.subTest(camera=camera,look=look):
                result=self.p.build_presentation(factory,shared,"geometry",SimpleNamespace(camera=camera,studio_look=look))
                shared.apply_look.assert_called_with(scene,studio,look)
                shared.base.make_inputs.assert_called_with(look_scene,frames,declaration,camera)
                self.assertEqual(result,(look_scene,look_studio,"trace","inputs",declaration))
        self.assertEqual(frames,tuple(range(300)))

    def test_saved_presentation_cannot_change_camera_light_energy_or_grid(self):
        declaration={"extent":[1280,720],"exposure":8.,"settings":{"area_light_grid":2}}
        source=dict(camera="sequence",studio_look="soft-studio",declaration=declaration,area_light_grid=2)
        self.p.verify_presentation_source(source,"sequence","soft-studio",declaration)
        for key,value in (("camera","detail"),("studio_look","recorded"),("area_light_grid",8),
                          ("declaration",dict(declaration,exposure=4.))):
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.p.verify_presentation_source(dict(source,**{key:value}),"sequence","soft-studio",declaration)
        self.p.verify_presentation_source({},"detail","recorded",declaration)
        with self.assertRaises(ValueError):self.p.verify_presentation_source({},"sequence","soft-studio",declaration)

    def test_display_movie_name_does_not_relabel_registered_numeric_configuration(self):
        policy=dict(paths_per_anchor=8,area_light_grid=2,estimator="visibility-ratio")
        old="rigid-t4-p8-g2-k25-r2-age4-visibility-ratio"
        self.assertEqual(self.p.media_profile_name(policy,{}),old)
        self.assertEqual(self.p.media_profile_name(policy,dict(camera="sequence",studio_look="soft-studio")),
                         old+"-sequence-soft-studio")
        policy["surface_quality"]={"panels":{"primary_samples_per_panel":16}}
        self.assertEqual(self.p.media_profile_name(policy,{}),old+"-panel16")
        policy["surface_quality"]["contours"]={"marker":104}
        self.assertEqual(self.p.media_profile_name(policy,{}),old+"-panel16-contour4")

    def test_camera_cut_keeps_the_observed_reset_and_pair_binding(self):
        values=[self.record(i) for i in range(3)]
        values[1]["history_actual"]=dict(sequence=0,last_reset=True)
        values[1]["history_reset_reasons"]=["camera_cut"]
        values[2]["history_actual"]=dict(sequence=1,last_reset=False)
        self.p.verify_chain(values,expected_indices=(0,1,2))
        self.p.verify_pair(deepcopy(values),values,require_reset_rgb=True)
        wrong=deepcopy(values);wrong[1]["camera_sha256"]="f"*64
        with self.assertRaises(ValueError):self.p.verify_pair(wrong,values,require_reset_rgb=True)


if __name__=="__main__":unittest.main()
