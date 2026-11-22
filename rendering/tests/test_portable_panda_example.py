"""Portable example construction and corruption controls; no renderer imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


class PortablePandaTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.work=Path(self.temp.name)
        self.builder=load('portable_builder',ROOT/'tools/build_panda_example.py')
        self.entry=load('portable_entry',ROOT/'examples/portable_panda_replay.py')

    def build(self,name='example.zip'):
        target=self.work/name; receipt=self.builder.build_bundle(ROOT,target)
        return target,receipt

    def unpack(self):
        archive,receipt=self.build(); target=self.work/'moved'/'panda'; target.mkdir(parents=True)
        with zipfile.ZipFile(archive) as bundle: bundle.extractall(target)
        return target,receipt

    def test_fixed_zip_moves_without_engine_robot_mesh_or_native_binaries(self):
        target,receipt=self.unpack(); manifest=self.entry.verify_bundle(target)
        self.assertEqual(manifest['schema'],'phymeo-portable-panda-example-v1')
        self.assertIn('tests/fixtures/panda_film/trace.json',manifest['files'])
        self.assertIn('tests/fixtures/panda_replay/LICENSE',manifest['files'])
        self.assertIn('licenses/CC0-1.0.txt',manifest['files'])
        self.assertIn('replay.py',manifest['files'])
        for path in manifest['files']:
            self.assertFalse(path.startswith(('src/','assets/panda/','phymeo/','phymeo_rendering/')))
            self.assertNotIn(Path(path).suffix,('.dll','.spv','.exe','.gltf'))
        self.assertNotIn(str(ROOT),json.dumps(manifest))
        self.assertEqual(receipt['sha256'],hashlib.sha256((self.work/'example.zip').read_bytes()).hexdigest())

    def test_same_inputs_produce_identical_zip_and_existing_destination_rejects(self):
        first,a=self.build('first.zip'); second,b=self.build('second.zip')
        self.assertEqual(first.read_bytes(),second.read_bytes()); self.assertEqual(a['sha256'],b['sha256'])
        with self.assertRaises(FileExistsError): self.builder.build_bundle(ROOT,first)

    def test_corrupt_trace_rejects_even_if_manifest_digest_is_rewritten(self):
        target,_=self.unpack(); relative='tests/fixtures/panda_film/trace.json'
        path=target/relative; path.write_bytes(path.read_bytes()+b' ')
        manifest=json.loads((target/'manifest.json').read_text()); manifest['files'][relative]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['files'][relative]['bytes']=path.stat().st_size
        (target/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'pinned|SHA256'): self.entry.verify_bundle(target)

    def test_missing_license_extra_code_and_manifest_escape_reject(self):
        target,_=self.unpack(); license=target/'licenses/CC0-1.0.txt'; original=license.read_bytes(); license.unlink()
        with self.assertRaises(ValueError): self.entry.verify_bundle(target)
        license.write_bytes(original); extra=target/'extra.py'; extra.write_text('pass')
        with self.assertRaises(ValueError): self.entry.verify_bundle(target)
        extra.unlink(); manifest=json.loads((target/'manifest.json').read_text()); manifest['files']['../escape']={}
        (target/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        with self.assertRaises(ValueError): self.entry.verify_bundle(target)

    def test_wrong_source_hash_prevents_publishing_a_zip(self):
        source=self.work/'source'; source.mkdir()
        for spec in self.entry.SOURCES.values():
            original=ROOT/spec['source']; dest=source/spec['source']; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(original,dest)
        film=source/'examples/panda_realtime_film.py'; film.write_bytes(film.read_bytes()+b'\n# changed\n')
        with self.assertRaisesRegex(ValueError,'pinned|SHA256'): self.builder.build_bundle(source,self.work/'bad.zip')
        self.assertFalse((self.work/'bad.zip').exists())

    def test_relocation_only_replaces_panda_directory_argument(self):
        calls=[]
        def build(*args,**kwargs): calls.append((args,kwargs)); return 'actual scene'
        film=SimpleNamespace(ROOT=self.work,build_scene=build)
        self.entry.relocate_panda(film)
        result=film.build_scene(self.work/'assets/panda','generated',(1,2,3),(4,5,6))
        self.assertEqual(result,'actual scene'); self.assertEqual(calls, [((None,'generated',(1,2,3),(4,5,6)),{})])
        with self.assertRaises(ValueError): film.build_scene(self.work/'foreign','generated',(1,2,3),(4,5,6))

    def test_short_single_frame_and_whole_episode_preserve_original_selection(self):
        single=self.entry.parse_args(['native','--library','engine.dll','--output','out'])
        full=self.entry.parse_args(['studio','--all','--output','out'])
        first=self.entry.film_arguments(single,self.work); all_frames=self.entry.film_arguments(full,self.work)
        self.assertEqual(first.frame,60); self.assertEqual(first.stride,1); self.assertEqual(first.tiles,1)
        self.assertIsNone(all_frames.frame); self.assertEqual(all_frames.tiles,2)
        self.assertTrue(first.daylight and first.photographic and first.surface_roughness)
        self.assertEqual(first.excerpt,self.work/'tests/fixtures/panda_film/trace.json')


if __name__=='__main__': unittest.main()
