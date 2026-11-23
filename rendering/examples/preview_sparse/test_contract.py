"""HOST-only isolated sparse prototype contracts; no GPU result is inferred."""
import importlib
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


class SparseContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder=importlib.import_module("preview_sparse.build_source")
        cls.runtime=importlib.import_module("preview_sparse.runtime")

    def test_scope_does_not_claim_progressive_samples(self):
        p=self.builder.contract(tile=2,paths=4,grid=2)
        self.assertEqual(p["history"],"none; one independent exposure per capture")
        self.assertEqual(p["paths_per_anchor"],4)
        self.assertEqual(p["guide_format"],"RGBA32F")
        self.assertEqual(p["fallback"],"current-pixel unoccluded diffuse environment SH; not traced GI")

    def test_only_bounded_supported_configurations(self):
        for tile,paths,grid in ((0,4,2),(3,4,2),(2,1,2),(2,8,2),(2,4,3),(True,4,2)):
            with self.subTest(tile=tile,paths=paths,grid=grid),self.assertRaises((ValueError,TypeError)):
                self.builder.contract(tile=tile,paths=paths,grid=grid)

    def test_odd_extent_anchor_clamps_inside_final_tile(self):
        self.assertEqual(self.builder.anchor((4,6),(5,7),4),(4,6))
        self.assertEqual(self.builder.anchor((0,0),(5,7),4),(2,2))
        self.assertEqual(self.builder.anchor((3,1),(5,7),2),(3,1))

    def test_rgb_integer_pack_is_exact_finite_and_chromatic(self):
        for rgb in ((0,0,0),(1,1,1),(.9,.1,.5),(.003,.24,.789)):
            packed=self.builder.pack_rho(rgb)
            self.assertLessEqual(packed,2**24-1)
            decoded=self.builder.unpack_rho(packed)
            self.assertTrue(all(abs(a-b)<=.5/255+1e-15 for a,b in zip(rgb,decoded)))
        self.assertNotEqual(self.builder.pack_rho((1,0,0)),self.builder.pack_rho((0,1,0)))

    def test_reject_patch_drift_instead_of_partial_silent_generation(self):
        with self.assertRaises(ValueError):self.builder.replace_once("a a","a","b","fixture")
        with self.assertRaises(ValueError):self.builder.replace_once("x","a","b","fixture")
        self.assertEqual(self.builder.replace_once("a","a","b","fixture"),"b")

    def test_generated_resources_and_markers_are_consistent(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)/"isolated"
            manifest=self.builder.prepare(root,destination,tile=2,paths=4,grid=2)
            self.assertEqual(manifest["contract"]["paths_per_anchor"],4)
            native=destination/"source/_native"
            spatial=(native/"resident_spatial.inc").read_text()
            renderer=(native/"renderer.cpp").read_text()
            compute=(destination/"source/_shaders/resident_spatial.comp").read_text()
            self.assertIn("PR_PREVIEW_SPARSE_GUIDE_ABI",spatial)
            self.assertIn("VK_FORMAT_R32G32B32A32_SFLOAT",spatial)
            self.assertIn("pr_preview_sparse_info",renderer)
            self.assertIn("PR_PREVIEW_SPARSE_COMPUTE_ABI",compute)
            self.assertIn("readonly uniform image2D currentLinear",compute)
            self.assertNotIn("imageLoad(meanLinear",compute)
            self.assertIn("uint fallback",compute)
            self.assertTrue((destination/"runtime/phymeo_rendering/backends/native.py").is_file())
            with self.assertRaises(FileExistsError):
                self.builder.prepare(root,destination,tile=2,paths=4,grid=2)

    def test_wrapper_refuses_unverified_or_historical_claims(self):
        self.assertEqual(self.runtime.COUNTERS[0],"invalid")
        self.assertEqual(self.runtime.COUNTERS[-1],"primary_proposals")
        self.assertNotIn("batches",self.runtime.COUNTERS)
        self.assertNotIn("samples_per_pixel",self.runtime.COUNTERS)

    def test_compact_revision_preserves_unique_anchor_iteration_order(self):
        # Independent integer-window oracle: retain old unique clamp order,
        # compare with the new bounded-index enumeration at odd/tiny extents.
        for width,height in ((1,1),(3,5),(7,9),(17,13)):
            for tile in (1,2,4):
                for side in (2,5):
                    last=((width-1)//tile,(height-1)//tile)
                    for y in range(height):
                        for x in range(width):
                            g=((x-tile//2)/tile,(y-tile//2)/tile)
                            base=tuple(math.floor(v+.5)-2 if side==5 else math.floor(v) for v in g)
                            old=[]
                            for j in range(side):
                                for i in range(side):
                                    q=tuple(min(max(0,min(base[k]+(i,j)[k],last[k]))*tile+tile//2,(width,height)[k]-1) for k in (0,1))
                                    if q not in old:old.append(q)
                            new=[]
                            for j in range(max(base[1],0),min(base[1]+side-1,last[1])+1):
                                for i in range(max(base[0],0),min(base[0]+side-1,last[0])+1):
                                    new.append((min(i*tile+tile//2,width-1),min(j*tile+tile//2,height-1)))
                            self.assertEqual(new,old,(width,height,tile,side,x,y))

    def test_reconstruction_revision_is_explicit_and_bounded(self):
        p=self.builder.contract(tile=4,paths=16,grid=2,neighbors=25,reconstruction_revision=2)
        self.assertEqual(p["reconstruction_revision"],2)
        self.assertEqual(p["neighbor_candidates"],25)
        self.assertIn("one accepted-neighbor atomic",p["reconstruction_implementation"])
        for invalid in (0,3,True,1.0):
            with self.assertRaises(ValueError):
                self.builder.contract(tile=4,paths=16,grid=2,reconstruction_revision=invalid)

    def test_ordinary_dll_does_not_resolve_private_setter(self):
        from .dispatch import progressive_setter
        class Only:
            def __init__(self,name):self.name=name;self.lookups=[]
            def __getattr__(self,name):
                self.lookups.append(name)
                if name!=self.name:raise AssertionError("unexpected optional symbol: "+name)
                return name
        for spatial,settling,name in ((False,False,"pr_resident_set_progressive"),(True,False,"pr_resident_set_spatial"),(True,True,"pr_resident_set_settling")):
            lib=Only(name)
            device=SimpleNamespace(_lib=lib,_spatial_requested=spatial,_settling_requested=settling)
            self.assertEqual(progressive_setter(device),name)
            self.assertEqual(lib.lookups,[name])

    def test_explicit_sparse_requires_private_abi(self):
        from .dispatch import progressive_setter
        def version():return 1
        sentinel=object()
        device=SimpleNamespace(_lib=SimpleNamespace(pr_preview_sparse_abi_version=version,pr_preview_sparse_set=sentinel),
            _spatial_requested=True,_settling_requested=False,_preview_sparse_requested=True)
        self.assertIs(progressive_setter(device),sentinel)
        device._settling_requested=True
        with self.assertRaises(ValueError):progressive_setter(device)

    def test_ratio_mode_is_explicit_and_separate_from_default(self):
        self.assertEqual(self.builder.contract(tile=4,paths=16,grid=2)["estimator"],"radiance")
        p=self.builder.contract(tile=4,paths=16,grid=2,estimator="visibility-ratio")
        self.assertIn("ratio bias",p["estimator_scope"])
        with self.assertRaises(ValueError):self.builder.contract(tile=4,paths=16,grid=2,estimator="unknown")


if __name__=="__main__":unittest.main()
