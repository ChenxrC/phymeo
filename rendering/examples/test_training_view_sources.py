"""Training source projection, timeline and single-plane request checks without a device."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from _training_view_sources import _channels, _image, open_training_source
from phymeo_rendering.outputs import Channel, ImagePlane


class TrainingViewSourceTests(unittest.TestCase):
    def image(self, channel, data, width):
        shot = SimpleNamespace(camera=SimpleNamespace(width=width, height=1),
                               planes=(ImagePlane(channel, data),))
        return np.asarray(Image.open(BytesIO(_image(shot, channel, (.5,2.5)))))[0]

    def test_depth_fixed_metres_and_black_miss_and_rgb_order(self):
        depth = np.asarray([np.inf,.1,.5,1.5,2.5,3.], '<f4')
        np.testing.assert_array_equal(self.image(Channel.FORWARD_DEPTH, depth.tobytes(), 6),
                                      [0,255,255,144,32,32])
        np.testing.assert_array_equal(self.image(Channel.RGB, bytes((208,49,2)), 1), [[208,49,2]])

    def test_id_display_uses_all_owner_bits_and_matches_published_legend(self):
        ids = np.asarray([0,1,2,3,1001,257,1<<24], '<u4')
        colors = self.image(Channel.INSTANCE_ID, ids.tobytes(), len(ids))
        self.assertTrue(np.all(colors[0] == 0))
        self.assertTrue(np.all(np.any(colors[1:] != 0, axis=1)))
        self.assertFalse(np.array_equal(colors[1], colors[5]))
        swatches = _channels()[2]['legend']
        self.assertEqual([entry['color'] for entry in swatches], ['#'+bytes(c).hex() for c in colors[:5]])

    def test_panda_seek_preserves_original_time_and_requests_only_selected_camera_plane(self):
        # Inspect the two real archived endpoint requests at the graphics API boundary.
        with TemporaryDirectory() as temp, patch('phymeo_rendering.backends.native.NativeBackend') as backend:
            session = backend.return_value.open_scene.return_value
            session.capture.side_effect = RuntimeError('request observed')
            source = open_training_source('panda', library=Path(__file__), assets=Path(temp))
            try:
                info = source.describe()
                self.assertEqual((info['frame_count'], info['fps'], info['seekable']), (300,30,True))
                for index, camera, key, time in ((299,'wrist-1','id',10.), (0,'overview','depth',.033125)):
                    with self.assertRaisesRegex(RuntimeError, 'request observed'):
                        source.capture(camera, index, key)
                    frame, request = session.capture.call_args.args
                    self.assertAlmostEqual(frame.time_s, time)
                    self.assertEqual(request.camera.id, camera)
                    self.assertEqual(request.camera.frame_key, frame.key)
                    self.assertEqual(len(request.channels), 1)
                    self.assertEqual(request.channels[0], Channel.INSTANCE_ID if key == 'id' else Channel.FORWARD_DEPTH)
                backend.return_value.open_scene.assert_called_once()
                self.assertEqual(session.capture.call_count, 2)
            finally:
                source.close()
            session.close.assert_called_once()
