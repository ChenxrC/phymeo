"""C+opaque wrapper contracts; no compiler or project imports."""
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))


def module():
    p=Path(__file__).with_name('compose_sparse.py');s=importlib.util.spec_from_file_location('compose_sparse',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


class SparseOpaqueContracts(unittest.TestCase):
    def test_all_six_exact_c_fragments_keep_real_sample_defines(self):
        m=module();specs=m.variant_specs(dict(tile=4,paths_per_anchor=16,area_light_grid=2,neighbor_candidates=25,reconstruction_revision=2))
        self.assertEqual(len(specs),6)
        self.assertEqual(sum('PR_PREVIEW_SPARSE_GUIDE=1' in flags for _,flags in specs),2)
        for name,flags in specs:
            self.assertIn('PR_PREVIEW_PATHS=16',flags);self.assertIn('PR_PREVIEW_TILE=4',flags)
            self.assertIn('PR_STUDIO_AREA_GRID=2',flags);self.assertNotIn('_area2',name)
            self.assertIn('PR_RESIDENT_PROGRESSIVE=1',flags)

    def test_manifest_rejects_changed_file_and_path_escape(self):
        m=module()
        with TemporaryDirectory() as d:
            root=Path(d);(root/'runtime').mkdir();file=root/'runtime/example.py';file.write_text('first')
            value=dict(contract={'schema':'phymeo-preview-sparse-v1'},generated_sha256={'runtime/example.py':m.digest(file)})
            (root/'source-manifest.json').write_text(json.dumps(value));m.validate_prepared(root)
            file.write_text('changed')
            with self.assertRaises(ValueError):m.validate_prepared(root)
            value['generated_sha256']={'../outside':'a'*64};(root/'source-manifest.json').write_text(json.dumps(value))
            with self.assertRaises(ValueError):m.validate_prepared(root)

    def test_pair_comparison_requires_actual_output_and_counter_equivalence(self):
        m=module();r=dict(contract={'paths_per_anchor':16},records=[dict(key='s0-f60',input_sha256='q',
            geometry_sha256={'mesh':'h'},rgb_sha256='r',all_rgb_sha256='r',sensors={'ID':'i'},linear_sha256='l',guide_sha256='g',
            actual={'primary_proposals':32},rgb_wall_ms=20.,gpu_render_ms=10.)],cleanup_passed=True,status='complete')
        self.assertTrue(m.compare_reports([r,r,r,r])['equivalent'])
        bad=json.loads(json.dumps(r));bad['records'][0]['guide_sha256']='wrong'
        self.assertFalse(m.compare_reports([r,bad,bad,r])['equivalent'])

    def test_windows_manifest_keys_are_normalized_without_rewriting_source(self):
        m=module()
        with TemporaryDirectory() as d:
            root=Path(d);(root/'source').mkdir();f=root/'source/a';f.write_bytes(b'a')
            value=dict(contract={'schema':'phymeo-preview-sparse-v1'},generated_sha256={'source\\a':m.digest(f)})
            raw=json.dumps(value);(root/'source-manifest.json').write_text(raw)
            _,manifest=m.validate_prepared(root)
            self.assertEqual(manifest['generated_sha256'],{'source/a':m.digest(f)})
            self.assertEqual((root/'source-manifest.json').read_text(),raw)

    def test_aliases_and_windows_path_escapes_are_rejected(self):
        m=module();h='a'*64
        for mapping in ({'a/b':h,'a\\b':h},{'a/b':h,'A/B':h},
                        {'../x':h},{'a/../x':h},{'/x':h},{'C:x':h},
                        {'C:/x':h},{'\\\\server\\x':h},{'a//b':h},{'a/./b':h}):
            with self.subTest(mapping=mapping),self.assertRaises(ValueError):m.normalized_hashes(mapping)

    def test_recovery_accepts_only_proven_two_shader_alias_defect(self):
        m=module();before='a'*64;after='b'*64;other='c'*64
        keys=('source/_shaders/resident_rayquery.glsl','source/_shaders/resident_reflection.glsl')
        original={k:before for k in keys};original['runtime/a.py']=other
        patches={k:dict(before=before,after=after) for k in keys}
        broken={k.replace('/','\\'):v for k,v in original.items()}
        broken.update({k:after for k in keys})
        expected={**original,**{k:after for k in keys}}
        self.assertEqual(m.recovered_hashes(original,broken,patches),expected)
        broken['runtime/a.py']=other
        with self.assertRaises(ValueError):m.recovered_hashes(original,broken,patches)
        del broken['runtime/a.py'];broken[keys[0]]='d'*64
        with self.assertRaises(ValueError):m.recovered_hashes(original,broken,patches)
    def test_copy_recovery_preserves_binary_bytes_and_rejects_tampered_spv(self):
        from unittest.mock import patch
        m=module()
        with TemporaryDirectory() as d:
            root=Path(d);original=root/'original';failed=root/'failed'
            for p in (original,failed):
                (p/'source/_shaders').mkdir(parents=True);(p/'build').mkdir()
            policy=dict(schema='phymeo-preview-sparse-v1',tile=4,paths_per_anchor=16,area_light_grid=2,
                        neighbor_candidates=25,reconstruction_revision=2)
            original_hashes={};patches={};broken={}
            for name in ('resident_rayquery.glsl','resident_reflection.glsl'):
                key='source/_shaders/'+name;(original/key).write_bytes(b'original');(failed/key).write_bytes(b'patched')
                before=m.digest(original/key);after=m.digest(failed/key)
                original_hashes[key.replace('/','\\')]=before;patches[key]=dict(before=before,after=after)
                broken[key.replace('/','\\')]=before;broken[key]=after
            base=dict(contract=policy,generated_sha256=original_hashes,profile_marker='C_PROFILE')
            m.write_json(original/'source-manifest.json',base);original_sha=m.digest(original/'source-manifest.json')
            oldmarks={'83':dict(name='C_PROFILE',value=[1])}
            newmarks={**oldmarks,'74':dict(name='PR_PREVIEW_OPAQUE_ABI',value=[1])}
            original_spv={};spv={};fragments=[]
            for name,_ in m.variant_specs(policy):
                (original/'build'/name).write_bytes(b'old');(failed/'build'/name).write_bytes(b'new')
                original_spv[name]=m.digest(original/'build'/name);spv[name]=m.digest(failed/'build'/name)
                fragments.append(dict(file=name,original_sha256=original_spv[name],sha256=spv[name],markers=newmarks))
                (failed/(name+'.log')).write_text('successful compile')
            for p in (original,failed):(p/'build/phymeo_native.dll').write_bytes(b'dll')
            old={**base,'generated_sha256':broken,'opaque_overlay':dict(patched_sources=patches,
                 original_source_manifest_sha256=original_sha),'state':'compiled'}
            m.write_json(failed/'source-manifest.json',old)
            proof=dict(status='ready',source_manifest_sha256=m.digest(failed/'source-manifest.json'),
                original_source_manifest_sha256=original_sha,original_policy=policy,patched_sources=patches,
                dll_sha256=m.digest(failed/'build/phymeo_native.dll'),original_spv_sha256=original_spv,
                spv_sha256=spv,fragments=fragments)
            m.write_json(failed/'opaque-overlay.json',proof)
            failed_manifest=(failed/'source-manifest.json').read_bytes()
            with patch.object(m,'marker_map',side_effect=lambda blob:oldmarks if blob==b'old' else newmarks):
                result=m.recover(failed,original,root/'recovered',original_sha)
                self.assertEqual(result['status'],'ready');self.assertFalse(result['project_execution'])
                m.verify_composed(root/'recovered')
                self.assertEqual((failed/'source-manifest.json').read_bytes(),failed_manifest)
                for name in spv:self.assertEqual((root/'recovered/build'/name).read_bytes(),b'new')
                (failed/'build'/next(iter(spv))).write_bytes(b'tampered')
                with self.assertRaises(ValueError):m.recover(failed,original,root/'rejected',original_sha)
                self.assertFalse((root/'rejected').exists())
if __name__=='__main__':unittest.main()
