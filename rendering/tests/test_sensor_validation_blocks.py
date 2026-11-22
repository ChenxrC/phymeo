"""Compare bulk publication checks with the scalar first-error contract."""
from dataclasses import replace
import struct
import unittest
from unittest.mock import patch

from phymeo_rendering.outputs import Capture, Channel, ImagePlane
from test_outputs import _camera, _key

try:
    import numpy
except ImportError:
    numpy = None


@unittest.skipUnless(numpy is not None, "optional array acceleration")
class SensorBlockTests(unittest.TestCase):
    size = 32771

    def make(self, depths=None, ids=None, valid=None):
        plane = ImagePlane
        return (
            plane(Channel.FORWARD_DEPTH, depths or struct.pack("<I", 1) * self.size),
            plane(Channel.INSTANCE_ID, ids or struct.pack("<I", 0xfffffffd) * self.size),
            plane(Channel.VALID, valid or bytes([1])*self.size),
        )

    def publish(self, planes):
        return Capture(_key(), 0., _camera(self.size, 1), planes)

    def test_valid_and_empty_blocks_do_not_enter_scalar_sample_iteration(self):
        planes = self.make()
        empty = self.make(struct.pack("<f",float("inf"))*self.size,bytes(4*self.size),bytes(self.size))
        with patch("phymeo_rendering.outputs._visible_samples", side_effect=AssertionError("scalar")):
            for fields in (planes,empty):
                for subset in (fields,fields[::-1],fields[:1],fields[1:]):
                    self.assertIs(self.publish(subset).planes, subset)

    def test_exceptional_float_encodings_preserve_scalar_failure_at_block_edges(self):
        for bits in (0,0x80000000,0x80000001,0xff800000,0x7f800001,0x7fc00000):
            for index in (8191,8192,self.size-1):
                raw = bytearray(struct.pack("<I", 1)*self.size)
                struct.pack_into("<I", raw, 4*index, bits)
                with self.subTest(bits=bits,index=index), self.assertRaisesRegex(ValueError, f"pixel {index}"):
                    self.publish(self.make(depths=bytes(raw)))

    def test_first_mismatch_precedes_later_invalid_sensor(self):
        planes = list(self.make())
        ids = bytearray(planes[1].data)
        struct.pack_into("<I",ids,17*4,0)
        mask = bytearray(planes[2].data)
        mask[18] = 2
        planes[1] = replace(planes[1],data=bytes(ids))
        planes[2] = replace(planes[2],data=bytes(mask))
        with self.assertRaisesRegex(ValueError,"background mismatch at pixel 17"):
            self.publish(tuple(planes))

    def test_scene_membership_includes_background_and_high_bit_ids(self):
        from phymeo_rendering.backends import _scene_ids_valid

        values = struct.pack("<III",0,0x80000000,0xffffffff)*12000
        self.assertTrue(_scene_ids_valid(values,frozenset((0x80000000,0xffffffff))))
        self.assertFalse(_scene_ids_valid(values,frozenset((0xffffffff,))))
