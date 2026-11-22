"""R4 scheduling/isolation checks; recorder is explicitly not rendering evidence."""
import importlib.util
import struct
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
sys.path.insert(0,str(EXAMPLES))
import native_multiworld as runner
from phymeo_rendering.outputs import Capture, Channel, ImagePlane
from phymeo_rendering.scene import FrameKey, Pose
from phymeo_rendering.cameras import Camera


def capture(identity=1, depth=1.):
    key=FrameKey('scene','world',0,0,0,'accepted_endpoint')
    camera=Camera('front',key,Pose(),2,1,1.,1.,1.,.5,.1,5.)
    return Capture(key,0.,camera,(
        ImagePlane(Channel.RGB,bytes(6)),
        ImagePlane(Channel.FORWARD_DEPTH,struct.pack('<ff',depth,float('inf'))),
        ImagePlane(Channel.INSTANCE_ID,struct.pack('<II',identity,0)),
        ImagePlane(Channel.VALID,b'\1\0')))


class MultiworldSchedule(unittest.TestCase):
    def test_300_batches_sample_real_endpoints_without_interpolation(self):
        values=[runner.source_slot(i,300,221,reset_at=None) for i in range(300)]
        self.assertEqual(values[0],(0,0)); self.assertEqual(values[-1],(220,0))
        self.assertTrue(all(a[0]<=b[0] for a,b in zip(values,values[1:])))
        self.assertTrue(all(type(i) is int for i,e in values))
    def test_only_reset_world_restarts_source_index_and_epoch(self):
        self.assertEqual(runner.source_slot(150,300,221,reset_at=150),(0,1))
        self.assertGreater(runner.source_slot(150,300,221,reset_at=None)[0],0)
        self.assertEqual(runner.source_slot(299,300,221,reset_at=150),(220,1))
    def test_bad_schedule_rejected(self):
        for args in ((-1,300,221),(300,300,221),(0,0,221),(0,300,0)):
            with self.assertRaises(ValueError): runner.source_slot(*args,reset_at=None)
        with self.assertRaises(ValueError): runner.source_slot(0,300,221,reset_at=0)


class MultiworldComparison(unittest.TestCase):
    def test_exact_id_valid_and_common_hit_depth(self):
        report=runner.compare_capture(capture(),capture(depth=1.001))
        self.assertTrue(report['passed']); self.assertEqual(report['compared_hit_pixels'],1)
        self.assertLess(report['max_depth_difference_m'],.002)
    def test_foreign_identity_wrong_labels_and_depth_are_failures(self):
        with self.assertRaises(ValueError):
            runner.compare_capture(capture(),replace(capture(),time_s=1.))
        self.assertFalse(runner.compare_capture(capture(),capture(identity=2))['passed'])
        self.assertFalse(runner.compare_capture(capture(),capture(depth=1.01))['passed'])
    def test_reference_storage_binds_bytes_and_metadata(self):
        with TemporaryDirectory() as root:
            path=Path(root)/'view.zlib'
            meta=runner.save_reference(path,capture())
            self.assertTrue(runner.compare_reference(path,meta,capture())['passed'])
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'digest'):
                runner.compare_reference(path,meta,capture())
    def test_statistics_require_complete_batches_and_do_not_invent_memory(self):
        report=runner.timing_report([.1,.2],worlds=2)
        self.assertEqual(report['completed_batches'],2)
        self.assertAlmostEqual(report['camera_frames_per_second'],8/.3)
        self.assertIsNone(report['peak_device_memory_bytes'])
        self.assertEqual(report['submission'],'serial public capture calls; one persistent session per world')


class RealArchiveSource(unittest.TestCase):
    def test_real_eight_model_assets_and_single_world_replay_reset(self):
        import os
        from _r4_archive import ArchiveWorlds
        root=os.environ.get('PHYMEO_R4_ARCHIVE')
        if not root: self.skipTest('set PHYMEO_R4_ARCHIVE to the actual eight-world archive')
        with TemporaryDirectory() as output:
            source=ArchiveWorlds(root,Path(output)/'assets')
            self.assertEqual(len(source.scenes),8)
            self.assertEqual(len({s.key for s in source.scenes}),8)
            self.assertGreater(source.record_count,1)
            first=source.frame(0,0,0); reset=source.frame(0,0,1)
            self.assertEqual(first.nodes,reset.nodes)
            self.assertEqual(first.time_s,reset.time_s)
            self.assertNotEqual(first.key,reset.key)
            other=source.frame(1,source.record_count-1,0)
            source.frame(0,0,1)
            self.assertEqual(other,source.frame(1,source.record_count-1,0))
            for world in range(8):
                a=source.frame(world,0,0); b=source.frame(world,source.record_count-1,0)
                self.assertNotEqual(a.nodes,b.nodes)
                self.assertEqual(len(source.cameras(world,a)),2)
                self.assertEqual(source.cameras(world,a)[0].width,320)
            source.assert_unchanged()
            self.assertEqual(source.physics_steps,0)


class ProcessMemorySampling(unittest.TestCase):
    def test_luid_bytes_bind_the_actual_adapter_not_ordinal(self):
        from _r4_resources import parse_counter_name
        self.assertEqual(parse_counter_name('pid_42_luid_0x00000001_0x12345678_phys_0'),
                         (42,'7856341201000000',0))
    def test_peak_excludes_other_pid_and_other_adapter(self):
        from _r4_resources import summarize_samples
        samples=[dict(received_perf_s=1.,status='ok',rows=[
            dict(name='pid_42_luid_0x00000001_0x12345678_phys_0',dedicated_bytes=10,shared_bytes=3),
            dict(name='pid_99_luid_0x00000001_0x12345678_phys_0',dedicated_bytes=1000,shared_bytes=3),
            dict(name='pid_42_luid_0x00000001_0x12345679_phys_0',dedicated_bytes=2000,shared_bytes=3)])]
        result=summarize_samples(samples,pid=42,luid='7856341201000000',start=0,end=2,period_s=.25)
        self.assertEqual(result['peak_device_memory_bytes'],10)
        self.assertEqual(result['samples'],1)
        missing=summarize_samples(samples,pid=42,luid=None,start=0,end=2,period_s=.25)
        self.assertIsNone(missing['peak_device_memory_bytes'])
        self.assertFalse(missing['adapter_bound'])
    def test_duplicate_adapter_row_is_not_double_counted(self):
        from _r4_resources import summarize_samples
        row=dict(name='pid_42_luid_0x00000001_0x12345678_phys_0',dedicated_bytes=10,shared_bytes=3)
        sample=dict(received_perf_s=1.,status='ok',rows=[row,row])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            summarize_samples([sample],pid=42,luid='7856341201000000',start=0,end=2,period_s=.25)


class TransferAccounting(unittest.TestCase):
    def test_tlas_and_studio_are_counted_once_not_duplicated_with_host_matrices(self):
        device=dict(resident_captures=2,resident_upload_calls=1,resident_vertex_count=3,
            resident_index_count=3,resident_mesh_count=1,resident_instance_count=1,resident_bvh_nodes=1,
            resident_geometry_upload_bytes=236,resident_transform_upload_bytes=256,
            resident_host_transform_input_bytes=128,resident_parameter_upload_bytes=688,
            resident_profile=dict(tlas_upload_bytes=96),resident_telemetry=None)
        report=runner.transfer_report(dict(device=device))
        self.assertEqual(report['host_to_device_bytes'],236+256+688+96)
        self.assertEqual(report['upload_components']['static_geometry']['vertex_buffer_bytes'],96)
        self.assertEqual(report['upload_components']['static_geometry']['index_buffer_bytes'],12)
        self.assertTrue(report['upload_components']['static_geometry']['sum_matches_counter'])
        self.assertEqual(report['upload_components']['uniforms']['studio_setup_bytes_in_counter'],144)
        self.assertEqual(report['upload_components']['host_transform_input_bytes_not_added'],128)
    def test_absent_profile_does_not_invent_zero_tlas_or_total(self):
        device=dict(resident_captures=1,resident_geometry_upload_bytes=100,
            resident_transform_upload_bytes=128,resident_parameter_upload_bytes=272)
        report=runner.transfer_report(dict(device=device))
        self.assertIsNone(report['host_to_device_bytes'])
        self.assertEqual(report['known_upload_subtotal_bytes'],500)
        self.assertIsNone(report['upload_components']['tlas_upload_bytes'])


if __name__=='__main__': unittest.main()
