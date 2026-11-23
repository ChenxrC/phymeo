"""Author-only contracts for the isolated sparse delivery wrapper; no GPU here."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

_IMPORT_PATH=list(sys.path)
try:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import run
finally:
    sys.path[:]=_IMPORT_PATH
    del _IMPORT_PATH


class SparseMotionContracts(unittest.TestCase):
    def test_full_trajectory_and_registered_append_do_not_duplicate_measured_states(self):
        self.assertEqual(run.indices(False,[59,60,149]),tuple(range(300)))
        self.assertEqual(run.indices(True,[59,60,149]),(0,59,60,149,239,299))
        with self.assertRaises(ValueError):run.indices(True,[300])

    def test_second_round_reverses_jobs_never_pose_chronology(self):
        first=run.jobs(("rgb","all"),1)
        second=run.jobs(("rgb","all"),2)
        self.assertEqual(second,list(reversed(first)))
        self.assertEqual(first,[("baseline16","rgb"),("baseline16","all"),("sparse","rgb"),("sparse","all")])

    def test_unique_source_frame_identity_is_not_replaced_by_count_only(self):
        frames=[dict(pose_index=i,frame_key={"frame":i},time_s=(i+1)/30) for i in range(3)]
        result=run.identity_audit(frames,range(3))
        self.assertEqual(result["unique_frame_keys"],3)
        bad=deepcopy(frames);bad[1]["frame_key"]=bad[0]["frame_key"]
        with self.assertRaises(ValueError):run.identity_audit(bad,range(3))
        bad=deepcopy(frames);bad[2]["time_s"]=bad[1]["time_s"]
        with self.assertRaises(ValueError):run.identity_audit(bad,range(3))

    def test_per_phase_timing_keeps_finger_window_and_never_claims_300_frame_gate(self):
        frames=[dict(pose_index=i,time_s=t,capture_ms=float(i+1),
            native_profile={"total_ns":{"gpu_total_ns":1000*(i+1)}}) for i,t in enumerate((.49,.5,2.299,2.3))]
        phases=run.phase_statistics(frames)["phases"]
        self.assertEqual([phases[k]["count"] for k in phases],[1,2,1])
        self.assertEqual(phases["finger_view_window"]["timing"]["mean_ms"],2.5)
        self.assertIsNone(phases["finger_view_window"]["timing"]["latency_gate_passed"])

    def test_optional_soft_look_never_changes_recorded_default_or_materials(self):
        @dataclass(frozen=True)
        class Scene:
            key:str="fixture"
            lights:tuple=("direction",)
            materials:tuple=("unchanged",)
        @dataclass(frozen=True)
        class Studio:
            panel_size_scale:float=1.
            panel_strength:float=.125
            environment_map:str="same HDR"
        scene=Scene();studio=Studio()
        a,b=run.apply_look(scene,studio,"recorded")
        self.assertIs(a,scene);self.assertIs(b,studio)
        a,b=run.apply_look(scene,studio,"soft-studio")
        self.assertEqual(a.lights,());self.assertEqual(a.materials,scene.materials)
        self.assertEqual(b.panel_size_scale,2.)
        self.assertEqual(b.panel_strength,studio.panel_strength)
        self.assertEqual(b.environment_map,studio.environment_map)
    def test_timing_is_nearest_rank_and_full_requires_300(self):
        result=run.statistics([float(i) for i in range(1,301)],full=True)
        self.assertEqual(result["p95_ms"],285.)
        self.assertEqual(result["p50_ms"],150.)
        self.assertFalse(result["latency_gate_passed"])
        self.assertIsNone(run.statistics([1.,2.],full=False)["latency_gate_passed"])
        with self.assertRaises(ValueError):run.statistics([1.],full=True)

    def test_sparse_counter_partition_and_actual_proposals_are_not_full_image_spp(self):
        a=dict(visible_pixels=10,traced_anchor_pixels=2,reconstructed_pixels=5,fallback_pixels=2,
               nondiffuse_pixels=1,primary_proposals=8,paths_per_anchor=4,invalid=0,
               rgb_captures=11,sensor_only_captures=0,last_capture_rgb=True,
               tile=2,area_light_grid=2,neighbor_candidates=25)
        p=dict(tile=2,paths_per_anchor=4,area_light_grid=2,neighbor_candidates=25)
        run.check_actual(a,p,11)
        bad=deepcopy(a);bad["primary_proposals"]=160
        with self.assertRaises(ValueError):run.check_actual(bad,p,11)
        bad=deepcopy(a);bad["reconstruction_revision"]=2
        with self.assertRaises(ValueError):run.check_actual(bad,p,11)
        bad=deepcopy(a);bad["fallback_pixels"]=3
        with self.assertRaises(ValueError):run.check_actual(bad,p,11)

    def test_sparse_and_baseline_capture_counters_stay_separate(self):
        before={"captures":10,"timestamp_samples":10,"gpu_total_ns":100,"timestamp_period_ns":1.}
        after={"captures":11,"timestamp_samples":11,"gpu_total_ns":150,"timestamp_period_ns":1.}
        result=run.profile_delta(before,after)
        self.assertEqual(result["total_ns"]["gpu_total_ns"],50)
        with self.assertRaises(ValueError):run.profile_delta(before,{**after,"captures":12})

    def test_encoder_uses_actual_contract_when_legacy_profile_is_a_timing_dict(self):
        policy=dict(tile=4,paths_per_anchor=16,area_light_grid=2,neighbor_candidates=25,
                    reconstruction_revision=2,estimator="visibility-ratio")
        actual=dict(**policy,invalid=0,last_capture_rgb=True,rgb_captures=310,sensor_only_captures=0,
                    visible_pixels=10,traced_anchor_pixels=2,reconstructed_pixels=5,fallback_pixels=2,
                    nondiffuse_pixels=1,primary_proposals=32)
        row=dict(name="sparse",profile={"mean_ms":{"gpu_total_ns":10.}},backend={"contract":policy,"actual":actual})
        saved=deepcopy(row)
        self.assertEqual(run.media_profile_name(row,policy),"sparse-t4-p16-g2-k25-r2-visibility-ratio")
        self.assertEqual(row,saved)
        bad=deepcopy(row);bad["backend"]["actual"]["tile"]=2
        with self.assertRaises(ValueError):run.media_profile_name(bad,policy)

    def test_encoder_cannot_inject_unbound_profile_or_filename_components(self):
        row=dict(name="baseline16",profile={"mean_ms":12.})
        self.assertEqual(run.media_profile_name(row,{}),"baseline16")
        with self.assertRaises(ValueError):run.media_profile_name(row,{},camera="../wrong")
        with self.assertRaises(ValueError):run.media_profile_name({**row,"name":"../wrong"},{})
    def test_profile_name_binds_all_runtime_contract_dimensions(self):
        self.assertEqual(run.profile_name(dict(tile=2,paths_per_anchor=4,area_light_grid=2,neighbor_candidates=25)),
                         "sparse-t2-p4-g2-k25-r1")

    @staticmethod
    def row(label,channels):
        return dict(name=label,channels=channels,extent=[1280,720],frames=[dict(pose_index=60,
            frame_key={"tick":60},time_s=2.,state_sha256="q",camera_sha256="cam",rgb_sha256=label,
            sensors={"id":"sensor"} if channels=="all" else {})])

    def test_geometry_must_match_different_lighting_models_but_rgb_need_not(self):
        a=self.row("baseline16","all");b=self.row("sparse","all")
        run.compare([a,b])
        b["frames"][0]["sensors"]={"id":"changed"}
        with self.assertRaises(ValueError):run.compare([a,b])

    def test_same_mode_rgb_all_and_fresh_rounds_require_exact_rgb(self):
        a=self.row("sparse","rgb");b=self.row("sparse","all")
        run.compare([a,b]);run.compare_repeat(a,deepcopy(a))
        b["frames"][0]["rgb_sha256"]="bad"
        with self.assertRaises(ValueError):run.compare([a,b])
        b=deepcopy(a);b["frames"][0]["camera_sha256"]="bad"
        with self.assertRaises(ValueError):run.compare_repeat(a,b)


if __name__=="__main__":unittest.main()