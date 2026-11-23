"""Independent geometry/temporal controls, before any native MSAA implementation."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
sys.path.insert(0,str(EXAMPLES))
SPEC = importlib.util.spec_from_file_location('msaa_calibration',EXAMPLES/'msaa_calibration.py')
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


class CoverageOracleTests(unittest.TestCase):
    def test_empty_full_and_winding_do_not_depend_on_samples(self):
        full = ((-1.,-1.),(3.,-1.),(3.,3.),(-1.,3.))
        self.assertEqual(calibration.coverage((),2,2),((0.,0.),(0.,0.)))
        self.assertEqual(calibration.coverage(full,2,2),((1.,1.),(1.,1.)))
        self.assertEqual(calibration.coverage(full[::-1],2,2),calibration.coverage(full,2,2))

    def test_exact_quarter_half_three_quarter_and_diagonal_triangle(self):
        for fraction in (.25,.5,.75):
            rectangle=((0.,0.),(fraction,0.),(fraction,1.),(0.,1.))
            with self.subTest(fraction=fraction):
                self.assertEqual(calibration.coverage(rectangle,1,1),((fraction,),))
        self.assertEqual(calibration.coverage(((0.,0.),(1.,0.),(0.,1.)),1,1),((.5,),))
        self.assertEqual(calibration.coverage(((1.,0.),(2.,0.),(2.,1.),(1.,1.)),1,1),((0.,),))

    def test_occluded_visible_area_is_not_double_counted(self):
        back=((0.,0.),(1.,0.),(1.,1.),(0.,1.))
        front=((-.5,-.5),(.25,-.5),(.25,1.5),(-.5,1.5))
        self.assertEqual(calibration.visible_coverage(back,front,1,1),((.75,),))
        self.assertEqual(calibration.visible_coverage(back,back,1,1),((0.,),))

    def test_half_coverage_resolve_must_be_linear_before_srgb(self):
        self.assertEqual(calibration.srgb8(.5),188)
        reference=((.5,),)
        mask=((True,),)
        good=calibration.rgb_coverage(bytes((188,188,188)),1,1,black=(0,0,0),white=(255,255,255))
        gamma_wrong=calibration.rgb_coverage(bytes((128,128,128)),1,1,black=(0,0,0),white=(255,255,255))
        self.assertLess(calibration.spatial_error(good,reference,mask)['mae'],.004)
        self.assertGreater(calibration.spatial_error(gamma_wrong,reference,mask)['mae'],.28)

    def test_measured_nonunit_contrast_and_uniform_controls(self):
        black=(32,32,32); white=(220,220,220)
        self.assertEqual(calibration.rgb_coverage(bytes(black),1,1,black=black,white=white),((0.,),))
        self.assertEqual(calibration.rgb_coverage(bytes(white),1,1,black=black,white=white),((1.,),))
        half=calibration.srgb8((calibration.linear8(32)+calibration.linear8(220))/2)
        result=calibration.rgb_coverage(bytes((half,)*3),1,1,black=black,white=white)
        self.assertAlmostEqual(result[0][0],.5,delta=.006)
        with self.assertRaises(ValueError):
            calibration.rgb_coverage(bytes(black),1,1,black=black,white=black)

    def test_geometric_mask_rejects_shifted_image_and_empty_measurement(self):
        polygon=((.25,0.),(1.25,0.),(1.25,2.),(.25,2.))
        reference=calibration.coverage(polygon,3,2)
        shifted=calibration.coverage(tuple((x+1,y) for x,y in polygon),3,2)
        mask=calibration.edge_mask((reference,))
        self.assertEqual(calibration.spatial_error(reference,reference,mask)['mae'],0.)
        self.assertGreater(calibration.spatial_error(shifted,reference,mask)['mae'],.4)
        with self.assertRaises(ValueError):
            calibration.spatial_error(reference,reference,((False,)*3,)*2)

    def test_temporal_metric_catches_flicker_even_when_total_light_is_constant(self):
        reference=tuple(((1-i/16,i/16),) for i in range(16))
        bad=tuple(((float(i%2),float(1-i%2)),) for i in range(16))
        mask=((True,True),)
        self.assertEqual(calibration.temporal_error(reference,reference,mask)['mae'],0.)
        self.assertGreater(calibration.temporal_error(bad,reference,mask)['mae'],.9)
        self.assertEqual(calibration.temporal_error(bad,reference,mask)['transitions'],15)

    def test_improvement_gates_do_not_pass_zero_baseline_or_wrong_sign(self):
        self.assertTrue(calibration.improvement(.30,.20,.25)['passed'])
        self.assertFalse(calibration.improvement(.30,.29,.25)['passed'])
        self.assertFalse(calibration.improvement(0.,0.,.20)['passed'])
        with self.assertRaises(ValueError):
            calibration.improvement(.3,-.1,.25)

    def test_pixel_centre_depth_and_id_are_nearest_opaque_surface(self):
        back=((0.,0.),(2.,0.),(2.,2.),(0.,2.))
        front=((0.,0.),(.75,0.),(.75,2.),(0.,2.))
        truth=calibration.centre_truth(((back,1.,11),(front,.75,22)),3,2)
        self.assertEqual(truth['ids'],((22,11,0),(22,11,0)))
        self.assertEqual(truth['valid'],((1,1,0),(1,1,0)))
        self.assertEqual(truth['depth'][0],(.75,1.,float('inf')))
        self.assertEqual(truth['boundary_pixels'],())

    def test_invalid_geometry_or_nonfinite_readback_is_rejected(self):
        for polygon in (((0.,0.),(float('nan'),0.),(1.,1.)),
                        ((0.,0.),(1.,0.),(0.,1.),(1.,1.))):
            with self.subTest(polygon=polygon),self.assertRaises(ValueError):
                calibration.coverage(polygon,2,2)
        with self.assertRaises(ValueError):
            calibration.coverage((),0,2)
        with self.assertRaises(ValueError):
            calibration.spatial_error(((float('nan'),),),((.5,),),((True,),))


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_MSAA_LIBRARY'),'requires actual MSAA DLL and serialized GPU slot')
class PublicDeviceAcceptance(unittest.TestCase):
    def test_public_msaa_coverage_motion_and_centre_sensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            report=calibration.run_experiment(
                Path(os.environ['PHYMEO_NATIVE_MSAA_LIBRARY']),Path(temporary)/'run',
                shaders=Path(os.environ['PHYMEO_NATIVE_MSAA_SHADERS']) if os.environ.get('PHYMEO_NATIVE_MSAA_SHADERS') else None,
                device_index=int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','1')),validation=True)
            self.assertEqual(report['status'],'passed')
            self.assertTrue(report['diagnostic_contract'])
            self.assertTrue(report['acceptance']['edge_mae_25_percent'])
            self.assertTrue(report['acceptance']['motion_error_20_percent'])
            self.assertTrue(report['acceptance']['linear_resolve'])
            self.assertTrue(report['acceptance']['centre_sensors'])
            self.assertTrue(report['acceptance']['request_combinations'])
            self.assertTrue(report['acceptance']['resize_and_reset'])
            self.assertTrue(report['acceptance']['cleanup'])


if __name__=='__main__':
    unittest.main()



