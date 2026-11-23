import importlib.util
from pathlib import Path
import unittest

P=Path(__file__).resolve().parent/'archived_score.py'

def module():
    s=importlib.util.spec_from_file_location('edge_archive_test',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class ArchivedScoreTests(unittest.TestCase):
    def test_complete_unique_original_300_identity_required(self):
        rows=[dict(pose_index=i,frame_key={'tick':i},time_s=i/30.,state_sha256=str(i),camera_sha256=str(i)) for i in range(300)]
        module().same_frames(rows,rows)
        bad=[dict(x) for x in rows];bad[149]['camera_sha256']='changed'
        with self.assertRaises(ValueError):module().same_frames(rows,bad)
        with self.assertRaises(ValueError):module().same_frames(rows[:-1],rows[:-1])

    def test_actual_baseline_cannot_be_a_reference_or_two_bounce(self):
        h={'indirect':{'mode':'diffuse_bounce','seed':0,'bounces':1,'samples_per_pixel':16},
           'progressive':{'batches':1,'primary_samples_per_pixel':16}}
        module().baseline_history(h)
        for key,value in (('seed',1),('bounces',2),('samples_per_pixel',4)):
            bad={**h,'indirect':dict(h['indirect'],**{key:value})}
            with self.assertRaises(ValueError):module().baseline_history(bad)

    def test_uri_paths_are_not_material_values(self):
        m=module()
        self.assertEqual(m.clean({'uri':'old','sha256':'a','roughness':.5}),m.clean({'uri':'new','sha256':'a','roughness':.5}))
        self.assertNotEqual(m.clean({'uri':'old','sha256':'a','roughness':.5}),m.clean({'uri':'new','sha256':'a','roughness':.6}))

if __name__=='__main__':unittest.main()