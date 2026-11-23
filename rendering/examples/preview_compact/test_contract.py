"""Pure HOST geometry/layout contracts, not GPU validation."""
from fractions import Fraction
import importlib
import unittest


class CompactContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder=importlib.import_module("preview_compact.build_source")
        cls.runtime=importlib.import_module("preview_compact.runtime")

    def test_true_small_extent(self):
        self.assertEqual(self.builder.low_extent(1280,720),(320,180))
        self.assertEqual(self.builder.low_extent(97,95),(25,24))
        self.assertEqual(self.builder.low_extent(1,1),(1,1))

    def test_viewport_center_matches_regular_full_pixel(self):
        # Solve the actual viewport transform independently with exact rationals.
        for width in (1,7,97,1280):
            vx,vw=self.builder.viewport_axis(width)
            for i in range((width+3)//4):
                ndc=2*(Fraction(2*i+1,2)-Fraction(vx))/Fraction(vw)-1
                original=(ndc+1)*width/2
                self.assertEqual(original,Fraction(4*i+2,1)+Fraction(1,2))

    def test_odd_end_is_not_clamped_to_wrong_full_pixel(self):
        self.assertIsNone(self.builder.anchor((24,23),(97,95)))
        self.assertEqual(self.builder.anchor((23,23),(97,95)),(94,94))
        self.assertIsNone(self.builder.anchor((0,0),(1,1)))

    def test_random_key_uses_full_integer_anchor(self):
        self.assertEqual(self.builder.anchor((12,9),(1280,720)),(50,38))
        self.assertNotEqual(self.builder.anchor((12,9),(1280,720)),(12,9))

    def test_only_declared_sparse_parent_is_admitted(self):
        source=dict(schema="phymeo-preview-sparse-v1",tile=4,paths_per_anchor=16,area_light_grid=2,
                    neighbor_candidates=25,reconstruction_revision=2,estimator="visibility-ratio",bounces=1)
        policy=self.builder.contract(source)
        self.assertEqual(policy["compact"]["tile"],4)
        self.assertEqual(policy["compact"]["history"],"none")
        self.assertIn("own independent",policy["compact"]["guide_attributes"])
        for key,value in (("tile",2),("paths_per_anchor",8),("neighbor_candidates",4),("reconstruction_revision",1),("bounces",2)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.builder.contract(dict(source,**{key:value}))

    def test_counter_partition_does_not_count_low_anchors_as_full_pixels(self):
        counts=dict(visible_pixels=9,traced_anchor_pixels=3,reconstructed_pixels=6,fallback_pixels=2,nondiffuse_pixels=1,primary_proposals=48,paths_per_anchor=16)
        self.runtime.validate_counters(counts)
        with self.assertRaises(ValueError):self.runtime.validate_counters(dict(counts,reconstructed_pixels=9))

    def test_smoke_runtime_binding_accepts_spelling_not_content_changes(self):
        from preview_compact.run_smoke import runtime_hashes
        a={"generated_sha256":{"runtime\\phymeo_rendering\\a.py":"a"}}
        b={"generated_sha256":{"Runtime/phymeo_rendering/a.py":"a"}}
        self.assertEqual(runtime_hashes(a),runtime_hashes(b))
        b["generated_sha256"]["Runtime/phymeo_rendering/a.py"]="b"
        self.assertNotEqual(runtime_hashes(a),runtime_hashes(b))
        b["generated_sha256"]["runtime/phymeo_rendering/a.py"]="b"
        with self.assertRaisesRegex(ValueError,"aliases"):runtime_hashes(b)


if __name__=="__main__":unittest.main()
