import os
import unittest
from unittest.mock import patch

import numpy as np

from .sorting_device_reader import AmbiguousStationRead, classify_station_rgb_cuda, compare_archive
from .sorting_reader import classify_station_rgb


def threshold_image(block, hexagon):
    # 10 x 10 ROI. Exterior blue detects a missed crop. Each of the four
    # background swatches misses one inclusive chroma threshold by exactly 1.
    image = np.full((40,40,3), (3,170,210), dtype=np.uint8)
    roi = np.tile(np.array([(92,75,57), (93,75,59), (21,28,53), (20,28,51)], dtype=np.uint8),
                  (25,1))
    roi[:block] = (93,75,58)
    roi[block:block+hexagon] = (20,28,52)
    image[15:25,15:25] = roi.reshape(10,10,3)
    return image


class CudaStationReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("optional PyTorch CUDA consumer")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA device required for station reader parity")
        cls.torch = torch
        cls.producer = torch.cuda.Stream()
        cls.consumer = torch.cuda.Stream()

    def strided_device_rgb(self, image):
        # Both row pitch and pixel stride exceed the logical RGB8 extent.
        # Alpha deliberately differs from blue in every element.
        padded = np.full((image.shape[0],image.shape[1]+7,4), 149, dtype=np.uint8)
        padded[:,1:1+image.shape[1],:3] = image
        with self.torch.cuda.stream(self.producer):
            rgba = self.torch.from_numpy(padded).to(self.producer.device, non_blocking=True)
            return rgba[:,1:1+image.shape[1],:3]

    def read(self, rgb):
        return classify_station_rgb_cuda(rgb, consumer_stream=self.consumer, producer_stream=self.producer)

    def test_exact_signed_thresholds_strides_and_decision_boundaries(self):
        for block,hexagon in ((55,12),(12,55),(55,13),(54,12),(0,0),(50,50)):
            with self.subTest(block=block, hexagon=hexagon):
                image = threshold_image(block, hexagon)
                rgb = self.strided_device_rgb(image)
                self.assertFalse(rgb.is_contiguous())
                try:
                    cpu_kind, cpu = classify_station_rgb(image)
                except ValueError as exc:
                    self.assertIn("unclassified", str(exc))
                    with self.assertRaises(AmbiguousStationRead) as rejected:
                        self.read(rgb)
                    measured = rejected.exception.measurement
                else:
                    kind, measured = self.read(rgb)
                    self.assertEqual(kind, cpu_kind)
                    self.assertEqual(measured["fraction"], cpu["fraction"])
                self.assertEqual(measured["roi_xyxy"], (15,15,25,25))
                self.assertEqual(measured["roi_pixels"], 100)
                self.assertEqual(measured["pixel_count"], {"block":block,"hex":hexagon})
                self.assertEqual((measured["d2h_copies"], measured["d2h_bytes"]), (1,16))

    def test_nondefault_stream_reduces_before_the_single_host_copy(self):
        torch = self.torch
        image = np.full((73,95,3), (32,95,174), dtype=np.uint8)
        rgb = self.strided_device_rgb(image)
        transfers = []
        original_to = torch.Tensor.to
        original_list = torch.Tensor.tolist
        def traced_to(tensor, *args, **kwargs):
            result = original_to(tensor, *args, **kwargs)
            if tensor.is_cuda and result.device.type == "cpu":
                stream = torch.cuda.current_stream(tensor.device)
                transfers.append((tuple(tensor.shape), tensor.dtype, stream.cuda_stream))
            return result
        def host_list(tensor):
            self.assertEqual(tensor.device.type, "cpu")
            return original_list(tensor)
        with (patch.object(torch.Tensor, "to", traced_to),
              patch.object(torch.Tensor, "tolist", host_list),
              patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected .cpu()")),
              patch.object(torch.Tensor, "numpy", side_effect=AssertionError("unexpected .numpy()"))):
            kind, result = self.read(rgb)
        self.assertEqual(kind, "hex")
        self.assertEqual(result["roi_xyxy"], classify_station_rgb(image)[1]["roi_xyxy"])
        self.assertEqual(transfers, [((2,),torch.int64,self.consumer.cuda_stream)])

    def test_rejects_host_tensor_default_stream_and_invalid_rgb_views(self):
        torch = self.torch
        rgb = self.strided_device_rgb(threshold_image(100,0))
        with self.assertRaisesRegex(ValueError, "CUDA tensor"):
            self.read(torch.zeros((32,32,3), dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "non-default"):
            classify_station_rgb_cuda(rgb, consumer_stream=torch.cuda.default_stream(rgb.device))
        with self.assertRaisesRegex(ValueError, "RGB8"):
            self.read(rgb[:,:,:2])
        with self.assertRaisesRegex(ValueError, "too small"):
            self.read(rgb[:31])

    @unittest.skipUnless(os.environ.get("PHYMEO_R1_READER_ARCHIVE"), "four archived station PNGs required")
    def test_archive_hashes_counts_and_permuted_reads(self):
        path = os.environ["PHYMEO_R1_READER_ARCHIVE"]
        original = compare_archive(path, device=self.consumer.device)
        changed = compare_archive(path, device=self.consumer.device, permuted=True)
        def identity(row):
            return (row["frame"], row["rgb_sha256"], row["decision"], row["pixel_count"])
        self.assertEqual([identity(row) for row in changed["reads"]],
                         [identity(original["reads"][i]) for i in (1,3,0,2)])
        self.assertNotEqual([row["decision"] for row in original["reads"]],
                            [row["decision"] for row in changed["reads"]])
        for report in (original, changed):
            self.assertIs(report["direct_vulkan_input"], False)
            self.assertEqual(report["transfers"], dict(h2d_copies=4, h2d_bytes=4*320*192*3,
                                                       d2h_copies=4, d2h_bytes=4*16))


if __name__ == "__main__":
    unittest.main()
