"""Tests-first independent finite-patch diffuse transport calibration."""
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

EXAMPLES=Path(__file__).resolve().parents[1]/'examples'
sys.path.insert(0,str(EXAMPLES))
spec=importlib.util.spec_from_file_location('diffuse_bounce_calibration',EXAMPLES/'diffuse_bounce_calibration.py')
chart=importlib.util.module_from_spec(spec)
spec.loader.exec_module(chart)


class DiffuseOracleTests(unittest.TestCase):
    def test_parallel_rectangle_closed_form_limits_and_symmetry(self):
        f=chart.rectangle_factor((0.,0.,0.),(-1.,1.,-1.,1.),1.)
        expected=4/math.pi/math.sqrt(2)*math.atan(1/math.sqrt(2))
        self.assertAlmostEqual(f,expected,places=14)
        self.assertLess(chart.rectangle_factor((0.,0.,0.),(-.01,.01,-.01,.01),1.),.001)
        self.assertGreater(chart.rectangle_factor((0.,0.,0.),(-100.,100.,-100.,100.),1.),.999)
        self.assertAlmostEqual(chart.rectangle_factor((.2,.3,0.),(-1.,1.,-1.,1.),1.),
                               chart.rectangle_factor((-.2,-.3,0.),(-1.,1.,-1.,1.),1.),places=14)

    def test_offset_patch_numeric_integral_is_independent_of_closed_form(self):
        point=(.07,-.09,0.)
        analytic=chart.rectangle_factor(point,chart.PATCH,chart.HEIGHT_M)
        actual=chart.patch_integral(point,128,'form-factor')
        self.assertAlmostEqual(actual,analytic,delta=analytic*1e-4)
        self.assertGreater(actual,0.)
        self.assertLess(actual,1.)

    def test_finite_patch_partition_is_additive(self):
        p=(.1,.05,0.)
        whole=chart.rectangle_factor(p,(.4,1.2,-.4,.4),.6)
        parts=sum(chart.rectangle_factor(p,b,.6) for b in ((.4,.8,-.4,.4),(.8,1.2,-.4,.4)))
        self.assertAlmostEqual(whole,parts,places=14)
        with self.assertRaises(ValueError):
            chart.rectangle_factor(p,(1.,0.,-1.,1.),.6)

    def test_black_baffle_blocks_all_receiver_to_plate_corner_segments_but_not_light(self):
        for p in chart.roi_points():
            self.assertTrue(abs(p[0])<.35 and abs(p[1])<.35 and p[2]==0.)
            for x in (chart.PATCH[0],chart.PATCH[1]):
                for y in (chart.PATCH[2],chart.PATCH[3]):
                    self.assertTrue(chart.baffle_blocks(p,(x,y,chart.HEIGHT_M)))
                    self.assertFalse(chart.baffle_blocks((x,y,chart.HEIGHT_M),(x+2,y,chart.HEIGHT_M-2)))

    def test_equal_linear_luminance_gray_is_not_equal_srgb_gray(self):
        self.assertAlmostEqual(sum(a*b for a,b in zip(chart.RED,chart.LUMA)),chart.GRAY[0],delta=2e-8)
        self.assertEqual(len(set(chart.GRAY)),1)
        self.assertGreater(chart.RED[0],chart.GRAY[0])
        self.assertLess(chart.RED[1],chart.GRAY[1])
        self.assertNotAlmostEqual(chart.inverse_srgb(chart.GRAY[0]),chart.GRAY[0],places=3)

    def test_native_and_reference_have_distinct_predeclared_transport_models(self):
        native=chart.patch_integral((0.,0.,0.),64,'native')
        pt=chart.patch_integral((0.,0.,0.),64,'sapien')
        self.assertGreater(native/pt,.8)
        self.assertLess(native/pt,.98)
        self.assertAlmostEqual(pt,chart.RECEIVER_ALBEDO*chart.LIGHT_E/math.sqrt(2)/math.pi*
                               chart.rectangle_factor((0.,0.,0.),chart.PATCH,chart.HEIGHT_M),delta=pt*.001)

    def test_inverse_transfer_has_explicit_half_code_quantization_bounds(self):
        for value in (0.,.001,.02,.1,.8):
            code=chart.encode_srgb(value)
            centre,lo,hi=chart.decode_code(code)
            self.assertLessEqual(lo,value)
            self.assertGreaterEqual(hi,value)
            self.assertTrue(lo<=centre<=hi)
        with self.assertRaises(ValueError):
            chart.decode_code(255)

    def test_fixed_mean_and_mse_gates_reject_scale_and_spatial_bias(self):
        reference=[.02]*16
        self.assertTrue(chart.numeric_gate(reference,reference,'native')['passed'])
        self.assertFalse(chart.numeric_gate([.04]*16,reference,'native')['passed'])
        # Same mean but spatial error cannot be hidden by a mean-only gate.
        self.assertFalse(chart.numeric_gate([0.,.04]*8,reference,'native')['passed'])
        self.assertFalse(chart.numeric_gate([.017,.023]*8,reference,'sapien')['passed'])

    def test_four_seed_evidence_records_spread_and_rejects_duplicates(self):
        result=chart.seed_summary([0,1,2,3],[1.,2.,3.,4.])
        self.assertEqual(result['count'],4)
        self.assertEqual(result['mean'],2.5)
        self.assertAlmostEqual(result['sample_sd'],math.sqrt(5/3))
        self.assertAlmostEqual(result['standard_error'],math.sqrt(5/3)/2)
        with self.assertRaises(ValueError):
            chart.seed_summary([0,0,0,0],[1.,1.,1.,1.])

    def test_blocked_color_gate_rejects_color_leaking_through_black_geometry(self):
        self.assertTrue(chart.blocked_gate([0.]*16,[.02]*16)['passed'])
        self.assertFalse(chart.blocked_gate([.002]*16,[.02]*16)['passed'])

    def test_sensor_contract_preserves_every_byte(self):
        before={c:b'1234' for c in chart.SENSORS}
        self.assertTrue(chart.sensor_equal(before,dict(before)))
        after=dict(before);after['FORWARD_DEPTH']=b'1235'
        self.assertFalse(chart.sensor_equal(before,after))

    def test_frozen_protocol_has_original_seeds_roi_samples_and_bounce_scope(self):
        p=chart.protocol()
        self.assertEqual(p['seeds'],[0,1,2,3])
        self.assertEqual(p['native_samples_per_pixel'],16)
        self.assertEqual(p['sapien_samples_per_pixel'],512)
        self.assertEqual(p['sapien_max_bounces'],[0,1])
        self.assertEqual(p['roi'],[64,44,96,76])
        self.assertEqual(p['native_mean_relative_limit'],.10)
        self.assertEqual(p['sapien_mean_relative_limit'],.05)
        self.assertEqual(p['native_nrmse_limit'],.60)
        self.assertEqual(p['sapien_nrmse_limit'],.12)
        self.assertEqual(p['blocked_relative_limit'],.03)


@unittest.skipUnless(os.environ.get('PHYMEO_DIFFUSE_DLL'),'new diffuse-bounce DLL and SAPIEN required')
class DiffuseDeviceTests(unittest.TestCase):
    def test_public_native_and_seeded_pt_against_separate_independent_integrals(self):
        with tempfile.TemporaryDirectory() as root:
            report=chart.run_experiment(Path(os.environ['PHYMEO_DIFFUSE_DLL']),Path(root)/'result',
                shaders=Path(os.environ['PHYMEO_DIFFUSE_DLL']).parent,validation=True)
            self.assertEqual(report['status'],'passed',report.get('acceptance'))
            self.assertEqual(report['physics_steps'],0)


if __name__=='__main__':unittest.main()
