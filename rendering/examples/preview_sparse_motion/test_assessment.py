"""Reference identity failures, without rendering or fabricated quality images."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
_IMPORT_PATH=list(sys.path)
try:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import assess
finally:
    sys.path[:]=_IMPORT_PATH
    del _IMPORT_PATH


class SparseReferenceContracts(unittest.TestCase):
    def fixture(self):
        native=dict(library_sha256="baseline",shaders={"main.spv":"shader"})
        diag=dict(library_sha256="baseline",shader_sha256={"frag":"shader"},
            indirect=dict(mode="diffuse_bounce",bounces=1,seed=1,samples_per_pixel=16),
            progressive=dict(batches=64,primary_samples_per_pixel=1024),area_light_grid=2,
            exposure=8.,display_transform="pbr-neutral",reflection=dict(mode="single_bounce"),
            spatial_filter=dict(last_capture_filtered=False))
        return native,diag

    def test_aesthetic_camera_or_lighting_cannot_inherit_registered_quality_gates(self):
        good=dict(resolution="720p",camera="detail",studio_look="recorded")
        assess.audit_configuration(good)
        for key,value in (("resolution","360p"),("camera","sequence"),("studio_look","soft-studio")):
            with self.assertRaises(ValueError):assess.audit_configuration({**good,key:value})
    def test_real_reference_contract_accepts_original_one_hop_only(self):
        native,diag=self.fixture()
        assess.verify_reference(diag,seed=1,batches=64,baseline=native,grid=2)
        bad=deepcopy(diag);bad["indirect"]["bounces"]=2
        with self.assertRaises(ValueError):assess.verify_reference(bad,seed=1,batches=64,baseline=native,grid=2)

    def test_same_seed_low_samples_or_changed_model_cannot_sign_quality(self):
        native,diag=self.fixture()
        for kwargs in (dict(seed=0,batches=64,baseline=native,grid=2),
                       dict(seed=1,batches=1,baseline=native,grid=2),
                       dict(seed=1,batches=64,baseline=native,grid=8)):
            with self.assertRaises(ValueError):assess.verify_reference(diag,**kwargs)
        bad=deepcopy(diag);bad["shader_sha256"]["frag"]="sparse"
        with self.assertRaises(ValueError):assess.verify_reference(bad,seed=1,batches=64,baseline=native,grid=2)

    def test_changing_display_or_library_rejects_same_pose_reference(self):
        native,diag=self.fixture()
        for key,value in (("library_sha256","other"),("exposure",1.),("display_transform","clip")):
            bad=deepcopy(diag);bad[key]=value
            with self.assertRaises(ValueError):assess.verify_reference(bad,seed=1,batches=64,baseline=native,grid=2)


if __name__=="__main__":unittest.main()