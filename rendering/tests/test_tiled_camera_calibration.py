"""Synthetic controls for the auditor only; no renderer quality is claimed here."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from phymeo_rendering.outputs import Capture,Channel,ImagePlane


@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy for independent image audit')
class TiledCalibrationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np=np
        path=Path(__file__).resolve().parents[1]/'examples/tiled_camera_calibration.py'
        spec=importlib.util.spec_from_file_location('tiled_camera_calibration_tested',path)
        self.module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=self.module; spec.loader.exec_module(self.module)
        self.directory=TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.scene,self.frame,self.camera,self.surfaces=self.module.build_fixture(self.directory.name)
        self.oracle=self.module.analytic_image(self.camera,self.surfaces)
        self.full=self.capture()

    def capture(self,depth=None,ids=None):
        np=self.np; depth=self.oracle['depth'] if depth is None else depth; ids=self.oracle['ids'] if ids is None else ids
        return Capture(self.frame.key,self.frame.time_s,self.camera,(
            ImagePlane(Channel.RGB,bytes(self.camera.width*self.camera.height*3)),
            ImagePlane(Channel.FORWARD_DEPTH,depth.astype('<f4').tobytes()),
            ImagePlane(Channel.INSTANCE_ID,ids.astype('<u4').tobytes()),
            ImagePlane(Channel.VALID,(ids!=0).astype('u1').tobytes())))

    def audit(self,tiled,full=None):
        return self.module.audit_pair(self.full if full is None else full,tiled,self.frame,self.camera,self.surfaces,257,173)

    def test_fixture_is_noncentral_nondisible_and_has_all_three_depths(self):
        self.assertNotEqual(self.camera.cx_px,self.camera.width/2)
        self.assertNotEqual(self.camera.cy_px,self.camera.height/2)
        self.assertNotEqual(self.camera.width%257,0); self.assertNotEqual(self.camera.height%173,0)
        self.assertEqual({-s.center[2] for s in self.surfaces},{1.1,1.8,3.2})
        ids=self.oracle['ids']
        for surface in self.surfaces:
            self.assertGreater(self.np.count_nonzero(ids==surface.identity),100)
        thin=ids==self.surfaces[-1].identity
        self.assertTrue(thin[:,:257].any() and thin[:,257:].any())
        self.assertTrue(thin[:173,:].any() and thin[173:,:].any())

    def test_identical_analytic_images_are_nonvacuous_positive_control(self):
        report,masks=self.audit(self.full)
        self.assertTrue(report['passed']); self.assertEqual(report['id_mismatches_all'],0)
        self.assertEqual(len(report['seams']),4)
        self.assertNotEqual(report['center_full_id'],0)
        self.assertGreater(self.np.count_nonzero(masks['seams']),0)

    def test_three_mm_interior_depth_error_rejects_without_redefining_edges(self):
        depth=self.oracle['depth'].copy(); mask=(self.oracle['ids']!=0)&~self.oracle['edge']
        depth[mask]+=.003
        report,_=self.audit(self.capture(depth=depth))
        self.assertFalse(report['passed']); self.assertGreater(report['interior_max_depth_difference_m'],.002)

    def test_one_edge_id_error_is_not_excluded_from_full_image_acceptance(self):
        ids=self.oracle['ids'].copy(); points=self.np.argwhere(self.oracle['edge']&(ids!=0))
        y,x=points[len(points)//2]; ids[y,x]=7
        report,masks=self.audit(self.capture(ids=ids))
        self.assertFalse(report['passed']); self.assertEqual(report['id_mismatches_all'],1)
        self.assertEqual(report['id_mismatches_geometric_edge'],1); self.assertTrue(masks['id_difference'][y,x])

    def test_shared_empty_seams_cannot_pass_by_pair_equality(self):
        ids=self.oracle['ids'].copy(); depth=self.oracle['depth'].copy()
        for x in (257,514): ids[:,x-1:x+1]=0; depth[:,x-1:x+1]=self.np.inf
        for y in (173,346): ids[y-1:y+1]=0; depth[y-1:y+1]=self.np.inf
        empty=self.capture(depth,ids); report,_=self.audit(empty,empty)
        self.assertFalse(report['passed']); self.assertEqual(report['id_mismatches_all'],0)
        self.assertTrue(all(row['full_foreground']==0 for row in report['seams']))

    def test_shifted_sensor_rejects_and_records_mismatch_locations(self):
        ids=self.np.roll(self.oracle['ids'],1,axis=1); depth=self.np.roll(self.oracle['depth'],1,axis=1)
        report,_=self.audit(self.capture(depth,ids))
        self.assertFalse(report['passed']); self.assertGreater(report['id_mismatches_all'],0)
        self.assertTrue(report['mismatch_xy_prefix'])

    def test_stale_capture_metadata_rejects_before_pixel_comparison(self):
        with self.assertRaisesRegex(ValueError,'exposure'):
            self.audit(replace(self.full,time_s=.1))


if __name__=='__main__': unittest.main()
