"""Delivery checks only; no rendering/device execution."""
import importlib.util
from pathlib import Path
import unittest


def module():
    path=Path(__file__).with_name('build.py');s=importlib.util.spec_from_file_location('delivery_build',path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


class DeliveryContracts(unittest.TestCase):
    def test_real_video_contract_counts_pts_and_rate(self):
        m=module();data=dict(streams=[dict(codec_name='h264',width=1280,height=720,avg_frame_rate='30/1',duration='10.000')],
            frames=[dict(best_effort_timestamp_time=str(i/30)) for i in range(300)])
        self.assertEqual(m.video_contract(data)['frames'],300)
        data['frames'][20]['best_effort_timestamp_time']='0.5'
        with self.assertRaises(ValueError):m.video_contract(data)
        data['frames']=data['frames'][:-1]
        with self.assertRaises(ValueError):m.video_contract(data)

    def test_receipt_pointer_does_not_replace_missing_measurements(self):
        m=module();r={'actual':{'profile':'raster / secondary2','rgb_mean_ms':123.5,'status':'failed'}}
        v=m.receipt_values(r,{'profile':'/actual/profile','rgb_mean_ms':'/actual/rgb_mean_ms','status':'/actual/status'})
        self.assertEqual(v['status'],'failed');self.assertEqual(v['rgb_mean_ms'],123.5)
        with self.assertRaises(ValueError):m.receipt_values(r,{'profile':'/actual/profile','rgb_mean_ms':'/fake','status':'/actual/status'})
        r['actual']['rgb_mean_ms']=float('nan')
        with self.assertRaises(ValueError):m.receipt_values(r,{'profile':'/actual/profile','rgb_mean_ms':'/actual/rgb_mean_ms','status':'/actual/status'})

    def test_crop_rejects_empty_offscreen_and_noninteger(self):
        m=module();self.assertEqual(m.crop_box([2,3,12,13],20,20),(2,3,12,13))
        for box in ([0,0,0,10],[-1,0,4,4],[0,0,21,4],[0.,0,4,4]):
            with self.subTest(box=box),self.assertRaises(ValueError):m.crop_box(box,20,20)

    def test_same_trajectory_requires_verified_receipt_hash(self):
        m=module();good='a'*64
        self.assertEqual(m.require_same_trajectory(good,[good,good]),good)
        with self.assertRaises(ValueError):m.require_same_trajectory(good,[good,'b'*64])
        with self.assertRaises(ValueError):m.require_same_trajectory(good,[None,good])

    def test_html_escapes_labels_keeps_cost_and_playback_distinct(self):
        m=module();text=m.page(dict(lanes=[dict(label='<script>x</script>',profile='actual',rgb_mean_ms=80,status='failed',video='file:///left.mp4'),
            dict(label='right',profile='other',rgb_mean_ms=20,status='passed',video='file:///right.mp4')],frames=[],source_times_s=[i/30 for i in range(300)]))
        self.assertNotIn('<script>x</script>',text)
        self.assertIn('RGB render mean',text);self.assertIn('30 FPS playback',text)
        self.assertIn('requestVideoFrameCallback',text);self.assertIn('failed',text)

    def test_png_assertion_is_real_pixels_not_only_dimensions(self):
        from PIL import Image
        m=module();a=Image.new('RGB',(4,4),(10,20,30));b=Image.new('RGB',(4,4),(10,20,30))
        self.assertEqual(m.image_assertion(a,b,0)['rgb8_mae'],0)
        b.putpixel((0,0),(255,255,255))
        with self.assertRaises(ValueError):m.image_assertion(a,b,0)
        with self.assertRaises(ValueError):m.image_assertion(a,Image.new('RGB',(3,4)),10)

if __name__=='__main__':unittest.main()
