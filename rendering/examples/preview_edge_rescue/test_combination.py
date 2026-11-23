"""Focused source/adapter controls for the optional near+full-rescue profile."""
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent

def module(name):
    spec=importlib.util.spec_from_file_location('edge_'+name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class CombinationTests(unittest.TestCase):
    def test_local_radius_is_scoped_to_original_low_guide(self):
        m=module('build_source')
        policy={'local_diffuse':{'radius_m':.12,'marker':96,'fade':'1-smoothstep(0,R,t)'}}
        reflection='#ifdef PR_PREVIEW_SPARSE_GUIDE\n#define PR_DIFFUSE_GUIDE_LIMIT 0.1200\n#else\n#define PR_DIFFUSE_GUIDE_LIMIT 3.402823466e38\n#endif\n'
        reflection+='rayQueryInitializeEXT(x,y,z,P,0.,direction,PR_DIFFUSE_GUIDE_LIMIT);\n'*2
        ratio='PR_NEARFIELD_DIFFUSE_ABI 1.0-smoothstep(0.0,PR_DIFFUSE_GUIDE_LIMIT,hit.previewDistance)'
        d=m.distance_policy(policy,reflection,ratio)
        self.assertEqual(d['sparse_radius_m'],.12)
        self.assertIsNone(d['rescue_radius_m'])
        for broken in (reflection.replace('0.1200','0.2'),reflection.replace('#ifdef PR_PREVIEW_SPARSE_GUIDE','#if 1')):
            with self.assertRaises(ValueError):m.distance_policy(policy,broken,ratio)

    def test_unlimited_original_remains_valid_but_unbound_cap_does_not(self):
        m=module('build_source')
        self.assertIsNone(m.distance_policy({},'unlimited original','')['sparse_radius_m'])
        with self.assertRaises(ValueError):m.distance_policy({},'#define PR_DIFFUSE_GUIDE_LIMIT .12','')

    def test_profile_is_explicitly_new_and_not_the_old_compact_label(self):
        m=module('motion')
        policy=dict(tile=4,paths_per_anchor=4,area_light_grid=2,neighbor_candidates=25,reconstruction_revision=2,
            local_diffuse={'radius_m':.12,'marker':96},edge_rescue={'schema':'phymeo-preview-edge-rescue-v1',
                'distance':{'sparse_radius_m':.12,'rescue_radius_m':None}})
        label=m.profile_name(policy)
        self.assertIn('local120mm',label);self.assertIn('rescue16-full',label)
        self.assertEqual(m.backend_name(policy),'experimental-native-compact-edge-rescue-v1')
        with self.assertRaises(ValueError):m.profile_name(dict(policy,edge_rescue={}))

    def test_filter_weights_and_limits_are_explicit(self):
        self.assertIn('*(1.0/256.0)',(HERE/'edge_merge.glsl').read_text())
        source=(HERE/'build_source.py').read_text()
        for name in ('maxPerStageDescriptorSampledImages','maxDescriptorSetStorageBuffers','maxPerStageResources'):
            self.assertIn(name,source)

if __name__=='__main__':unittest.main()