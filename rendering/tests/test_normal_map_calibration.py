"""Independent normal-chart contracts; no shader or tangent implementation is imported."""
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

EXAMPLES=Path(__file__).resolve().parents[1]/'examples'
sys.path.insert(0,str(EXAMPLES))
SPEC=importlib.util.spec_from_file_location('normal_map_calibration',EXAMPLES/'normal_map_calibration.py')
chart=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chart)


class NormalOracleTests(unittest.TestCase):
    def assertVector(self,a,b,tolerance=1e-7):
        self.assertEqual(len(a),len(b))
        for x,y in zip(a,b):
            self.assertAlmostEqual(x,y,delta=tolerance)

    def test_linear_flat_map_retains_real_unorm_bias(self):
        actual=chart.decode_normal((128,128,255))
        length=math.sqrt(1+2/255**2)
        self.assertVector(actual,(1/255/length,1/255/length,1/length),1e-14)
        self.assertGreater(actual[0],0)
        self.assertVector(chart.decode_normal((255,128,128)),(actual[2],actual[0],actual[1]),1e-14)

    def test_standard_and_mirrored_u_have_correct_bitangent_handedness(self):
        panels=chart.panel_definitions()
        a=panels['flat']; b=panels['mirror']
        ta,ba,na,ha=chart.tangent_frame(a['positions'],a['uv'],a['normal'])
        tb,bb,nb,hb=chart.tangent_frame(b['positions'],b['uv'],b['normal'])
        self.assertVector(ta,(1.,0.,0.));self.assertVector(tb,(-1.,0.,0.))
        self.assertVector(ba,(0.,1.,0.));self.assertVector(bb,(0.,1.,0.))
        self.assertEqual((ha,hb),(1,-1))
        self.assertVector(na,nb)

    def test_geometry_and_texture_tilts_are_both_accounted_for(self):
        panel=chart.panel_definitions()['tilted']
        tangent,bitangent,normal,hand=chart.tangent_frame(panel['positions'],panel['uv'],panel['normal'])
        self.assertVector(normal,(math.sin(math.radians(25)),0.,math.cos(math.radians(25))))
        self.assertVector(bitangent,(0.,1.,0.))
        mapped=chart.mapped_normal(panel,chart.NORMALS['direction'])
        sample=chart.decode_normal(chart.NORMALS['direction'])
        self.assertVector(mapped,chart.unit(tuple(sample[0]*tangent[i]+sample[1]*bitangent[i]+sample[2]*normal[i] for i in range(3))))
        self.assertGreater(mapped[0],sample[0])
        self.assertEqual(hand,1)

    def test_positive_y_on_mirrored_u_rejects_missing_handedness_and_green_flip(self):
        panel=chart.panel_definitions()['mirror']
        correct=chart.mapped_normal(panel,chart.NORMALS['positive-y'])
        omitted=chart.mapped_normal(panel,chart.NORMALS['positive-y'],omit_handedness=True)
        flipped=chart.mapped_normal(panel,chart.NORMALS['positive-y'],flip_green=True)
        self.assertGreater(correct[1],.5)
        self.assertLess(omitted[1],-.5)
        self.assertLess(flipped[1],-.5)
        def difference(n):
            return chart.radiance(n,(0.,0.,1.),chart.LIGHTS['plus'])-chart.radiance(n,(0.,0.,1.),chart.LIGHTS['minus'])
        self.assertGreater(difference(correct),0)
        self.assertLess(difference(omitted),0)
        self.assertLess(difference(flipped),0)

    def test_normal_incidence_has_closed_form_rough_dielectric_radiance(self):
        actual=chart.radiance((0.,0.,1.),(0.,0.,1.),(0.,0.,1.))
        self.assertAlmostEqual(actual,2*(.96*.28+.04/4)/math.pi,places=14)
        self.assertEqual(chart.radiance((0.,0.,1.),(0.,0.,1.),(0.,0.,-1.)),0.)

    def test_fixed_rois_are_inside_real_geometry_and_have_reliable_predicted_margin(self):
        for name,panel in chart.panel_definitions().items():
            for mode in ('direction','positive-y'):
                with self.subTest(panel=name,mode=mode):
                    plus=chart.roi_prediction(panel,mode,'plus')
                    minus=chart.roi_prediction(panel,mode,'minus')
                    self.assertEqual(len(plus['rgb8']),24*24*3)
                    self.assertGreaterEqual(abs(plus['mean_rgb8']-minus['mean_rgb8']),8.)
                    self.assertTrue(plus['all_inside'])
                    self.assertTrue(min(plus['linear'])>=0 and max(plus['linear'])<1)

    def test_signed_gate_rejects_reversed_light_response_or_insufficient_margin(self):
        self.assertTrue(chart.direction_gate(130.,90.,129.,91.)['passed'])
        self.assertFalse(chart.direction_gate(130.,90.,90.,130.)['passed'])
        self.assertFalse(chart.direction_gate(100.,96.,100.,96.)['passed'])
        self.assertFalse(chart.direction_gate(130.,90.,145.,90.)['passed'])

    def test_flat_gate_is_per_pixel_and_sensors_are_byte_exact(self):
        self.assertTrue(chart.flat_gate(bytes((100,100,100)),bytes((101,99,100)))['passed'])
        self.assertFalse(chart.flat_gate(bytes((100,100,100)),bytes((102,100,100)))['passed'])
        base={name:b'1234' for name in chart.SENSOR_NAMES}
        self.assertTrue(chart.sensor_equal(base,dict(base)))
        for name in chart.SENSOR_NAMES:
            changed=dict(base);changed[name]=b'1235'
            self.assertFalse(chart.sensor_equal(base,changed))

    def test_degenerate_uv_and_invalid_map_values_are_rejected(self):
        panel=chart.panel_definitions()['flat']
        with self.assertRaises(ValueError):
            chart.tangent_frame(panel['positions'],((0.,0.),)*4,panel['normal'])
        for rgb in ((256,128,255),(128,-1,255),(True,128,255),(128,128)):
            with self.subTest(rgb=rgb),self.assertRaises((ValueError,TypeError)):
                chart.decode_normal(rgb)


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_NORMAL_LIBRARY'),'requires actual normal-map product and serialized GPU slot')
class PublicNormalDeviceTests(unittest.TestCase):
    def test_actual_chart_shading_and_unchanged_sensor_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            report=chart.run_experiment(Path(os.environ['PHYMEO_NATIVE_NORMAL_LIBRARY']),Path(directory)/'run',
                shaders=Path(os.environ['PHYMEO_NATIVE_NORMAL_SHADERS']) if os.environ.get('PHYMEO_NATIVE_NORMAL_SHADERS') else None,
                device_index=int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','1')),validation=True)
            self.assertEqual(report['status'],'passed')
            self.assertTrue(all(report['acceptance'].values()))


if __name__=='__main__':
    unittest.main()
