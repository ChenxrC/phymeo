"""Author-only tests; execution belongs to the parent serial queue."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent

def load():
    spec = importlib.util.spec_from_file_location('edge_build_source_under_test', ROOT / 'build_source.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class EdgeSourceTests(unittest.TestCase):
    def test_contract_names_actual_full_rate_paths_and_generic_support(self):
        c = load().contract(262144)
        self.assertEqual(c['max_support'], 2)
        self.assertEqual(c['paths_per_selected_fragment'], 16)
        self.assertEqual(c['selection'], 'actual accepted K25 support <= 2 and nonzero diffuse coefficient')
        self.assertIn('not hardware ray', c['counter_scope'])
        self.assertFalse(c['history'])

    def test_bound_is_explicit_and_bool_is_not_count(self):
        for n in (True, 0, -1, 1048577, 1.5):
            with self.subTest(n=n), self.assertRaises(ValueError): load().contract(n)

    def test_strict_anchor_rejects_missing_and_duplicate(self):
        m=load()
        for source in ('abc','x x'):
            with self.assertRaises(ValueError): m.once(source,'x','y')
        self.assertEqual(m.once('axc','x','y'),'ayc')

    def test_shader_only_uses_generic_identity_and_real_trace(self):
        fragment=(ROOT/'edge_fragment.glsl').read_text()
        self.assertIn('residentDiffuseIncoming(worldPosition,N,reflectionFace)', fragment)
        self.assertIn('texelFetch(edgeIds,pixel,0).r!=ids.x', fragment)
        self.assertIn('layout(early_fragment_tests)', fragment)
        self.assertIn('edgeStatus.maxInvocations',fragment)
        for forbidden in ('156u','158u','69u','pose-0015'):
            self.assertNotIn(forbidden,fragment)

    def test_merge_does_not_filter_direct_or_surface_coefficient(self):
        merge=(ROOT/'edge_merge.glsl').read_text()
        self.assertIn('primary.rgb+rho*incoming',merge)
        self.assertIn('texelFetch(instanceIds,q,0).r!=id',merge)
        self.assertIn('depthMatches(z,',merge)
        self.assertIn('dot(normal,',merge)
        self.assertIn('status.missingRescue',merge)

    def test_success_counts_require_complete_real_rescue(self):
        m=load()
        d=dict(selected_pixels=69,fragment_invocations=70,traced_invocations=70,
            reconstructed_rescue_pixels=69,filtered_neighbors=170,missing_rescue=0,
            budget_exhausted=0,max_invocations=262144,invalid=0)
        self.assertEqual(m.validate_counts(d)['extra_primary_proposals'],1120)
        for key,value in (('missing_rescue',1),('budget_exhausted',1),('invalid',1),
                          ('reconstructed_rescue_pixels',68),('traced_invocations',71)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_counts(dict(d,**{key:value}))

if __name__=='__main__': unittest.main()