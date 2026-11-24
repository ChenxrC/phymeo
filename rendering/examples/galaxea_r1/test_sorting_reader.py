import unittest

import numpy as np

from .sorting_reader import classify_station_rgb


class StationColorReader(unittest.TestCase):
    def test_uses_only_rgb_in_the_central_inspection_window(self):
        image = np.full((96,160,3), (220,125,42), dtype=np.uint8)
        image[36:60,60:100] = (37,105,165)
        kind, metrics = classify_station_rgb(image)
        self.assertEqual(kind, "hex")
        self.assertEqual(metrics["fraction"]["hex"], 1.)
        image[36:60,60:100] = (220,125,42)
        self.assertEqual(classify_station_rgb(image)[0], "block")

    def test_neutral_and_conflicting_reads_stop_sorting(self):
        image = np.full((96,160,3), 112, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "unclassified"):
            classify_station_rgb(image)
        image[:,:80] = (220,125,42)
        image[:,80:] = (37,105,165)
        with self.assertRaisesRegex(ValueError, "mixed"):
            classify_station_rgb(image)
