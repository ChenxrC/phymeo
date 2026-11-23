"""HOST contracts; actual Vulkan and endpoint probes are separate, opt-in."""
import importlib.util
from pathlib import Path
import unittest


def module():
    path=Path(__file__).with_name('overlay.py')
    spec=importlib.util.spec_from_file_location('opaque_overlay',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class OpaqueContracts(unittest.TestCase):
    def test_shadow_retains_complete_original_fallback(self):
        m=module();root=Path(__file__).resolve().parents[2]/'src/phymeo_rendering/backends/_shaders'
        original=(root/'resident_rayquery.glsl').read_text()
        result=m.patch_shadow(original)
        self.assertIn(original[original.index('bool occluded'):].replace('bool occluded(', 'bool previewOriginalOccluded(',1),result)
        self.assertIn('gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT',result)
        self.assertIn('t>u.options.y && t<tMax',result)
        self.assertIn('return previewOriginalOccluded(origin,direction,tMax);',result)

    def test_closest_opaque_does_not_terminate_on_first_candidate(self):
        m=module();root=Path(__file__).resolve().parents[2]/'src/phymeo_rendering/backends/_shaders'
        original=(root/'resident_reflection.glsl').read_text()
        result=m.patch_closest(original)
        head=result[result.index('rayQueryEXT query;'):result.index('uint slot=')]
        self.assertIn('gl_RayFlagsOpaqueEXT',head)
        self.assertNotIn('TerminateOnFirstHit',head)
        self.assertIn(m.original_query(original),head)
        self.assertIn('previewT>u.options.y && previewT<3.402823466e38',head)
        self.assertEqual(result[result.index('uint slot='):],original[original.index('uint slot='):])

    def test_open_interval_endpoint_fallback_is_whole_segment(self):
        # Hardware first-hit order is deliberately arbitrary, including upper first.
        m=module()
        for distances in ((),(1.,),(0.,),(.5,),(1.,.5),(0.,.5),(0.,1.)):
            expected=any(0.<t<1. for t in distances)
            for selected in distances or (None,):
                self.assertEqual(m.shadow_decision(selected,distances,0.,1.),expected)

    def test_source_drift_rejected_and_only_registered_variants(self):
        m=module()
        with self.assertRaises(ValueError):m.patch_shadow('wrong')
        with self.assertRaises(ValueError):m.patch_closest('wrong')
        args=m.variant_defines('resident_rayquery_normal_reflection_indirect_progressive_area8_secondary2.frag.spv')
        for expected in ('PR_RESIDENT_NORMAL_MAPS=1','PR_RESIDENT_PROGRESSIVE=1','PR_STUDIO_SECONDARY_GRID=2'):
            self.assertIn(expected,args)
        with self.assertRaises(ValueError):m.variant_defines('resident_software.frag.spv')


if __name__=='__main__':unittest.main()
