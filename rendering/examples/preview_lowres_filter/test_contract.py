"""Fixed low-grid filter contract; no Vulkan imports."""
import importlib.util
from pathlib import Path
import unittest


def module():
    p=Path(__file__).with_name('build_source.py');s=importlib.util.spec_from_file_location('lowres_builder',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


class LowresFilterContracts(unittest.TestCase):
    def test_registration_is_one_fixed_stride2_kernel(self):
        m=module();p=m.policy()
        self.assertEqual((p['marker'],p['kernel'],p['stride']),(100,3,2))
        self.assertEqual(p['passes'],1);self.assertEqual(p['relative_depth'],.003)
        self.assertEqual(p['normal_dot'],.98);self.assertFalse(p['changes_center_alpha'])

    def test_uniform_and_isolated_zero_support_center_are_exact(self):
        m=module()
        value=m.reference_filter((.3,.2,.1,17.),[(.3,.2,.1)]*9,[True]*9)
        for actual,expected in zip(value,(.3,.2,.1,17.)):self.assertAlmostEqual(actual,expected)
        self.assertEqual(m.reference_filter((.3,.2,.1,17.),[(1.,2.,3.)]*9,[False]*9),(.3,.2,.1,17.))
        self.assertEqual(m.reference_filter((.3,.2,.1,17.),[(1.,2.,3.)]*9,[True]*9,center_supported=False),(.3,.2,.1,17.))

    def test_explicit_same_identity_depth_normal_support_gates(self):
        m=module()
        self.assertTrue(m.accept(7,7,1,1,1.,1.002,1.,True))
        for values in ((7,8,1,1,1.,1.,1.,True),(7,7,0,1,1.,1.,1.,True),(7,7,1,1,1.,1.01,1.,True),(7,7,1,1,1.,1.,.97,True),(7,7,1,1,1.,1.,1.,False)):
            self.assertFalse(m.accept(*values))

    def test_zero_support_between_endpoints_cannot_be_bridged(self):
        m=module();cells=m.corridor((2,2))
        self.assertEqual(set(cells),{(x,y) for y in range(3) for x in range(3)})
        self.assertIn((1,1),cells);self.assertIn((1,0),cells)
        self.assertEqual(m.corridor((0,0)),[(0,0)])

    def test_reconstruction_replaces_only_candidate_rgb_and_keeps_real_counter_source(self):
        m=module();source='''#version 450
bool depthMatches(float a,float b) {return true;}
void main() {
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy),extent=ivec2(p.width,p.height);
    vec4 candidate=texelFetch(incomingGuide,cell,0);
    atomicAdd(status.primaryProposals,uint(PR_PREVIEW_PATHS));
}'''
        changed=m.patch_compute(source)
        self.assertIn('candidate.rgb=lowFiltered.values',changed)
        self.assertIn('atomicAdd(status.primaryProposals,uint(PR_PREVIEW_PATHS));',changed)
        self.assertIn('if((p.mode&256u)!=0u)',changed)
        with self.assertRaises(ValueError):m.patch_compute(changed)

    def test_pingpong_barrier_and_readonly_source_are_explicit(self):
        text=Path(__file__).with_name('lowres.glsl').read_text()
        native=Path(__file__).with_name('lowres_native.inc').read_text()
        self.assertIn('texelFetch(incomingGuide',text)
        self.assertNotIn('imageStore(incomingGuide',text)
        self.assertIn('lowFiltered.values',text)
        self.assertIn('VK_ACCESS_SHADER_WRITE_BIT',native)
        self.assertIn('VK_ACCESS_SHADER_READ_BIT',native)
        self.assertIn('VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT',native)


if __name__=='__main__':unittest.main()