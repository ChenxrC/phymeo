"""Portable compact delivery checks; no renderer/device execution."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

_IMPORT_PATH=list(sys.path)
try:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import candidate_adapter as adapter
    import run
finally:
    sys.path[:]=_IMPORT_PATH
    del _IMPORT_PATH


def policy(compact=True):
    p=dict(schema="phymeo-preview-sparse-v1",tile=4,paths_per_anchor=16,
           area_light_grid=2,neighbor_candidates=25,reconstruction_revision=2,
           estimator="visibility-ratio",bounces=1)
    if compact:p["compact"]=dict(schema="phymeo-preview-compact-v1",tile=4,history="none")
    return p


def actual(compact=True):
    return dict(tile=4,paths_per_anchor=16,area_light_grid=2,neighbor_candidates=25,
        reconstruction_revision=2,estimator="visibility-ratio",invalid=0,last_capture_rgb=True,
        rgb_captures=310,sensor_only_captures=0,visible_pixels=10,traced_anchor_pixels=2,
        reconstructed_pixels=7 if compact else 5,fallback_pixels=2,nondiffuse_pixels=1,
        primary_proposals=32)


def compact_info():
    return dict(abi_version=1,enabled=True,tile=4,full_width=1280,full_height=720,
        guide_width=320,guide_height=180,last_capture_rgb=True,rgb_captures=310,
        sensor_only_captures=0,successful_guide_draws=310,primary_color_attachments=5,
        reconstruction_sampled_images=8)


class CompactDeliveryContracts(unittest.TestCase):
    def test_finite_range_diffuse_has_a_distinct_measured_profile(self):
        fixed=policy()
        original=run.profile_name(fixed)
        fixed["local_diffuse"]={"radius_m":.12,"marker":96}
        self.assertEqual(run.profile_name(fixed),original+"-local-120mm")
        fixed["local_diffuse"]["radius_m"]=float("nan")
        with self.assertRaises(ValueError):
            run.profile_name(fixed)

    def test_candidate_only_presentation_never_invents_a_paired_baseline(self):
        self.assertEqual(run.jobs(("rgb",),1,candidate_only=True),[("sparse","rgb")])
        self.assertEqual(run.jobs(("rgb","all"),2,candidate_only=True),[("sparse","all"),("sparse","rgb")])
        self.assertEqual(len(run.jobs(("rgb","all"),1)),4)
    def test_backend_kind_requires_explicit_matching_source_contract(self):
        self.assertEqual(adapter.kind(policy()),"compact")
        self.assertEqual(adapter.kind(policy(False)),"sparse")
        adapter.require_kind(policy(),"compact")
        with self.assertRaises(ValueError):adapter.require_kind(policy(),"sparse")
        with self.assertRaises(ValueError):adapter.require_kind(policy(False),"compact")
        bad=policy();bad["compact"]["schema"]="wrong"
        with self.assertRaises(ValueError):adapter.kind(bad)

    def test_compact_full_partition_does_not_count_low_raster_anchors_twice(self):
        run.check_actual(actual(),policy(),310)
        with self.assertRaises(ValueError):run.check_actual(actual(False),policy(),310)
        run.check_actual(actual(False),policy(False),310)
        with self.assertRaises(ValueError):run.check_actual(actual(),policy(False),310)

    def test_compact_proposals_remain_low_anchor_times_paths(self):
        a=actual();a["primary_proposals"]=160
        with self.assertRaises(ValueError):run.check_actual(a,policy(),310)

    def test_compact_actual_extent_draw_and_attachment_counts_are_checked(self):
        value=compact_info();adapter.check_compact(value,(1280,720),310,actual())
        for key,value in (("guide_width",1280),("full_height",719),("successful_guide_draws",309),
                          ("primary_color_attachments",4),("reconstruction_sampled_images",7),
                          ("sensor_only_captures",1),("last_capture_rgb",False)):
            bad=compact_info();bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                adapter.check_compact(bad,(1280,720),310,actual())
        bad=actual();bad["traced_anchor_pixels"]=320*180+1
        with self.assertRaises(ValueError):adapter.check_compact(compact_info(),(1280,720),310,bad)

    def test_compact_odd_extent_uses_ceil_without_claiming_clamped_anchor(self):
        info=compact_info();info.update(full_width=1279,full_height=719)
        adapter.check_compact(info,(1279,719),310,actual())

    def test_explicit_class_selection_never_falls_back_to_sparse(self):
        sparse=object();compact=object();module=SimpleNamespace(SparseBackend=sparse,CompactBackend=compact)
        self.assertIs(adapter.backend_class(module,policy()),compact)
        self.assertIs(adapter.backend_class(module,policy(False)),sparse)
        with self.assertRaises(ValueError):adapter.backend_class(SimpleNamespace(SparseBackend=sparse),policy())

    def test_compact_media_name_binds_actual_diagnostics_and_keeps_timing_dict(self):
        p=policy();a=actual()
        row=dict(name="sparse",extent=[1280,720],profile={"mean_ms":{"gpu_total_ns":12.}},
            backend=dict(contract=p,actual=a,compact=compact_info(),
                         backend="experimental-native-compact-raster-incoming-v1",history="none"))
        saved=deepcopy(row)
        self.assertEqual(run.media_profile_name(row,p),"compact-t4-p16-g2-k25-r2-visibility-ratio")
        self.assertEqual(row,saved)
        bad=deepcopy(row);bad["backend"]["compact"]["guide_width"]=1280
        with self.assertRaises(ValueError):run.media_profile_name(bad,p)

    def test_saved_vr2_timing_profile_receipt_remains_encodable_without_compact(self):
        p=policy(False);row=dict(name="sparse",profile={"mean_ms":{"gpu_total_ns":12.}},
                              backend=dict(contract=p,actual=actual(False)))
        self.assertEqual(run.media_profile_name(row,p),"sparse-t4-p16-g2-k25-r2-visibility-ratio")


if __name__=="__main__":unittest.main()
