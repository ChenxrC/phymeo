"""Portable overlay admission tests; no compiler, Vulkan, or project execution."""
from pathlib import Path
import struct
import tempfile
import unittest
import build_overlay as overlay


def instruction(opcode, *operands):
    return [((len(operands)+1)<<16)|opcode, *operands]


def text_words(value):
    b=value.encode()+b"\0"
    b+=b"\0"*((-len(b))%4)
    return struct.unpack("<"+"I"*(len(b)//4),b)


def marker_fixture(samples=2, sharing=1, *, omit_opcode=None, omit_capability=None):
    # Structural fixture ONLY for the admission parser, not executable SPIR-V.
    words=[0x07230203,0x10500,0,100,0]
    for capability in (61,64,68):
        if capability!=omit_capability:
            words+=instruction(17,capability)
    words+=instruction(21,1,32,0)
    for i,(name,(specid,default)) in enumerate(overlay.MARKERS.items(),2):
        value=samples if name.endswith("PROPOSALS") else sharing if name.endswith("REUSE") else default
        words+=instruction(5,i,*text_words(name))
        words+=instruction(71,i,1,specid)
        words+=instruction(50,1,i,value)
    for opcode in (339,341,365):
        if opcode!=omit_opcode:
            words+=instruction(opcode,1,50,3,4)
    return struct.pack("<"+"I"*len(words),*words)


class PreviewQuadAdmissionTests(unittest.TestCase):
    def inspect(self,data,samples=2,sharing=True):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"fixture.spv"
            p.write_bytes(data)
            return overlay.inspect_spirv(p,samples,sharing)

    def test_parser_checks_compiled_defaults_for_both_real_budgets(self):
        for samples in (2,4):
            result=self.inspect(marker_fixture(samples),samples)
            self.assertEqual(result["PR_PREVIEW_PRIMARY_PROPOSALS"]["default"],samples)

    def test_legacy_sixteen_cannot_be_presented_as_two(self):
        with self.assertRaises(ValueError):
            self.inspect(marker_fixture(16))

    def test_missing_quad_or_ballot_is_rejected_even_with_matching_names(self):
        for opcode in (339,341,365):
            with self.subTest(opcode=opcode),self.assertRaises(ValueError):
                self.inspect(marker_fixture(omit_opcode=opcode))

    def test_missing_fragment_group_capability_is_rejected(self):
        for capability in (61,64,68):
            with self.subTest(capability=capability),self.assertRaises(ValueError):
                self.inspect(marker_fixture(omit_capability=capability))

    def test_sharing_mode_is_bound_separately_from_sample_budget(self):
        self.inspect(marker_fixture(sharing=0),sharing=False)
        with self.assertRaises(ValueError):
            self.inspect(marker_fixture(sharing=0),sharing=True)

    def test_bad_binary_extent_or_magic_is_rejected(self):
        for data in (b"",b"SPIRV",b"\0"*20,marker_fixture()[:-1]):
            with self.subTest(length=len(data)),self.assertRaises(ValueError):
                self.inspect(data)

    def test_exact_main_patch_leaves_old_direct_and_sensor_text_untouched(self):
        source=Path(__file__).resolve().parents[2]/"src/phymeo_rendering/backends/_shaders/resident.frag"
        text=source.read_text(encoding="utf-8-sig")
        result=overlay.patch_main(text)
        self.assertEqual(result.count("diffuseIncoming=previewQuadIncoming("),1)
        self.assertIn("    forwardDepth = -(u.view*vec4(worldPosition, 1.0)).z;",result)
        self.assertEqual(text[text.index("    if(u.counts.z!=0u && shadingFacesView && (reflected"):],
                         result[result.index("    if(u.counts.z!=0u && shadingFacesView && (reflected"):])
        with self.assertRaises(ValueError):
            overlay.patch_main(result)

    def test_unknown_main_anchor_cannot_silently_make_a_partial_overlay(self):
        with self.assertRaises(ValueError):
            overlay.patch_main("#version 450\nvoid main(){}")


if __name__=="__main__":
    unittest.main()

