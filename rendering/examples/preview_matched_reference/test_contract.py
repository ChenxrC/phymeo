"""No renderer dependency or project execution in the comparison contract tests."""
import importlib.util
from pathlib import Path
import unittest


def module():
    path=Path(__file__).with_name('compare.py');spec=importlib.util.spec_from_file_location('matched_reference',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class MatchedReferenceContracts(unittest.TestCase):
    def test_registered_single_pose_common_material_and_color_are_fixed(self):
        p=module().protocol()
        self.assertEqual((p['pose'],p['extent'],p['exposure']), (60,[1280,720],8.))
        self.assertEqual(p['display_transform'],'pbr-neutral')
        self.assertEqual(p['scene_index'],0);self.assertFalse(p['normal_map'])
        self.assertEqual(p['pt_tile'],[320,180]);self.assertEqual(p['pt_spp'],1024)

    def test_different_actual_pose_material_or_lighting_is_rejected(self):
        m=module();a={'input_sha256':'a','status':'complete'}
        m.check_match([a,a,a])
        with self.assertRaises(ValueError):m.check_match([a,dict(a,input_sha256='b'),a])
        with self.assertRaises(ValueError):m.check_match([a,dict(a,status='failed'),a])

    def test_tile_plan_keeps_all_pixel_rays_and_extent(self):
        m=module();tiles=m.tile_plan(1280,720,320,180)
        self.assertEqual(len(tiles),16)
        self.assertEqual(tiles[0],(0,0,320,180));self.assertEqual(tiles[-1],(960,540,320,180))
        odd=m.tile_plan(7,5,4,3)
        self.assertEqual(sum(w*h for _,_,w,h in odd),35)
        for args in ((0,5,4,3),(7,5,0,3),(7,5,4,True)):
            with self.assertRaises(ValueError):m.tile_plan(*args)

    def test_differences_are_diagnostic_not_a_cross_backend_correctness_claim(self):
        p=module().protocol()
        self.assertIn('not ground truth',p['comparison_scope'])
        self.assertIn('tile-local',p['pt_sampling_scope'])
        self.assertFalse(p['performance_claim']);self.assertEqual(p['physics_steps'],0)


    def test_missing_final_resource_evidence_is_not_success(self):
        m=module()
        self.assertFalse(m.closed_ok('full',{}));self.assertFalse(m.closed_ok('sapien',{}))
        self.assertTrue(m.closed_ok('sapien',{'lifecycle':{'closed':True,'attached_entities_after_clear':0,'studio_temporary_files_exist':False}}))
        d=dict(closed=True,live_contexts=0,live_frames=0,live_resources=0,validation_errors=0,
            resident_telemetry=dict(final_after_destroy=True,device_local_bytes_current=0,host_visible_bytes_current=0,allocation_count=8,free_count=8))
        self.assertTrue(m.closed_ok('near',dict(device=d)))
        d['validation_errors']=1
        self.assertFalse(m.closed_ok('near',dict(device=d)))

if __name__=='__main__':unittest.main()