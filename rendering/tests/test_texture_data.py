"""Owned texture bytes and declarations; no SDK/device required by these tests."""
import builtins
from dataclasses import FrozenInstanceError, fields
import hashlib
import importlib
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from phymeo_rendering import assets

RGB = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255))


def png_bytes(width=2, height=2, pixels=RGB, *, alpha=False, extras=()):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    channels = 4 if alpha else 3
    scanlines = b"".join(b"\0" + pixels[y * width * channels:(y + 1) * width * channels] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6 if alpha else 2, 0, 0, 0))
            + b"".join(chunk(kind, value) for kind, value in extras)
            + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


class TextureAssetTests(unittest.TestCase):
    def test_material_layout_stays_compatible_and_texture_is_explicit(self):
        self.assertEqual([field.name for field in fields(assets.Material)], ["id", "base_color", "metallic", "roughness"])
        texture = assets.TextureMap("not-opened.png", "a" * 64)
        material = assets.TexturedMaterial("paint", (.4, .5, .6), .1, .7, texture)
        self.assertIsInstance(material, assets.Material)
        self.assertIs(material.base_color_map, texture)
        self.assertEqual((material.base_color, material.metallic, material.roughness), ((.4, .5, .6), .1, .7))
        with self.assertRaises(FrozenInstanceError):
            texture.sha256 = "b" * 64
        with self.assertRaises(FrozenInstanceError):
            material.base_color_map = texture

    def test_texture_and_subclass_validate_original_material_fields(self):
        for uri, digest in (("", "a" * 64), ("x.png", "a" * 63), ("x.png", "A" * 64), ("x.png", "g" * 64), (4, "a" * 64)):
            with self.subTest(uri=uri, digest=digest), self.assertRaises((ValueError, TypeError)):
                assets.TextureMap(uri, digest)
        texture = assets.TextureMap("x.png", "a" * 64)
        for color, metallic, roughness, image in (([1., 1., 1.], 0., .5, texture), ((1., 1., 1.), -1., .5, texture),
                                                ((1., 1., 1.), 0., float("nan"), texture), ((1., 1., 1.), 0., .5, "x.png")):
            with self.subTest(color=color, metallic=metallic, roughness=roughness), self.assertRaises((ValueError, TypeError)):
                assets.TexturedMaterial("bad", color, metallic, roughness, image)

    def test_import_and_declaration_do_not_import_pillow_or_graphics(self):
        original = builtins.__import__
        def no_optional(name, *args, **kwargs):
            if name.split(".")[0] in {"PIL", "numpy", "sapien"}:
                raise AssertionError("optional dependency imported during declaration: " + name)
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=no_optional):
            from phymeo_rendering import texture_data
            importlib.reload(texture_data)
            texture = assets.TextureMap("unread.png", "a" * 64)
            assets.TexturedMaterial("paint", (1., 1., 1.), 0., .5, texture)


class TextureDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "four-corners.png"

    def load(self, raw, *, suffix=".png", digest=None):
        from phymeo_rendering.texture_data import load_texture
        path = self.path.with_suffix(suffix)
        path.write_bytes(raw)
        descriptor = assets.TextureMap(path.as_uri(), hashlib.sha256(raw).hexdigest() if digest is None else digest)
        return load_texture(descriptor)

    def test_png_pixels_are_owned_srgb8_top_row_first_without_transform(self):
        raw = png_bytes(extras=((b"gAMA", struct.pack(">I", 100000)),))
        decoded = self.load(raw)
        self.assertEqual((decoded.width, decoded.height), (2, 2))
        self.assertEqual(decoded.rgb8, RGB)
        self.assertEqual(decoded.source_sha256, hashlib.sha256(raw).hexdigest())
        self.path.write_bytes(b"replaced after loading")
        self.assertEqual(decoded.rgb8, RGB)
        with self.assertRaises(FrozenInstanceError):
            decoded.width = 1

    def test_fully_opaque_rgba_is_admitted_but_one_transparent_texel_is_not(self):
        rgba = bytes(value for i in range(4) for value in (*RGB[3*i:3*i+3], 255))
        self.assertEqual(self.load(png_bytes(pixels=rgba, alpha=True)).rgb8, RGB)
        bad = bytearray(rgba)
        bad[-1] = 254
        with self.assertRaisesRegex(ValueError, "opaque|alpha"):
            self.load(png_bytes(pixels=bytes(bad), alpha=True))

    def test_jpeg_is_explicitly_supported_without_changing_color_space(self):
        from PIL import Image
        stream = io.BytesIO()
        Image.new("RGB", (8, 8), (150, 80, 30)).save(stream, format="JPEG", quality=100, subsampling=0)
        decoded = self.load(stream.getvalue(), suffix=".jpg")
        self.assertEqual((decoded.width, decoded.height, len(decoded.rgb8)), (8, 8, 8*8*3))
        for actual, expected in zip(decoded.rgb8[:3], (150, 80, 30), strict=True):
            self.assertLessEqual(abs(actual - expected), 2)

    def test_wrong_hash_precedes_image_decode(self):
        from PIL import Image
        with patch.object(Image, "open", side_effect=AssertionError("decoded untrusted image")):
            with self.assertRaisesRegex(ValueError, "SHA256|digest"):
                self.load(png_bytes(), digest="0" * 64)

    def test_only_local_explicit_image_formats_and_complete_images(self):
        from phymeo_rendering.texture_data import load_texture
        for uri in ("https://example.invalid/a.png", "file://server/share/a.png", "//server/share/a.png", self.path.as_uri() + "?v=1"):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                load_texture(assets.TextureMap(uri, "a" * 64))
        for raw, suffix in ((b"GIF89a" + b"x"*40, ".png"), (png_bytes()[:-14], ".png"), (png_bytes(), ".jpg")):
            with self.subTest(suffix=suffix, tail=raw[-10:]), self.assertRaises(ValueError):
                self.load(raw, suffix=suffix)

    def test_byte_and_decoded_dimension_limits_precede_pixel_allocation(self):
        from phymeo_rendering import texture_data
        from PIL import Image
        with patch.object(texture_data, "_MAX_FILE_BYTES", 16):
            with patch.object(Image, "open", side_effect=AssertionError("opened oversized file")):
                with self.assertRaisesRegex(ValueError, "byte|size|limit"):
                    self.load(png_bytes())
        with patch.object(texture_data, "_MAX_DIMENSION", 1):
            with self.assertRaisesRegex(ValueError, "dimension|width|height|limit"):
                self.load(png_bytes())
        with patch.object(texture_data, "_MAX_PIXELS", 3):
            with self.assertRaisesRegex(ValueError, "pixel|limit"):
                self.load(png_bytes())

    def test_unsupported_bit_depth_cmyk_and_animation_are_not_silently_reduced(self):
        from PIL import Image
        for image, fmt, suffix, kwargs in (
            (Image.new("I;16", (2, 2), 1024), "PNG", ".png", {}),
            (Image.new("CMYK", (2, 2), (0, 0, 0, 0)), "JPEG", ".jpg", {}),
            (Image.new("RGB", (2, 2), (255, 0, 0)), "PNG", ".png", {"save_all": True, "append_images": [Image.new("RGB", (2, 2), (0, 255, 0))]}),
        ):
            stream = io.BytesIO()
            image.save(stream, format=fmt, **kwargs)
            with self.subTest(mode=image.mode, animated=bool(kwargs)), self.assertRaises(ValueError):
                self.load(stream.getvalue(), suffix=suffix)

    def test_missing_pillow_is_reported_only_when_explicitly_loading(self):
        from phymeo_rendering.texture_data import load_texture
        raw = png_bytes()
        self.path.write_bytes(raw)
        descriptor = assets.TextureMap(self.path.as_uri(), hashlib.sha256(raw).hexdigest())
        original = builtins.__import__
        def missing(name, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise ModuleNotFoundError("PIL deliberately absent")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=missing):
            with self.assertRaisesRegex(ImportError, "Pillow"):
                load_texture(descriptor)

    def test_texture_data_rejects_mutable_bytes_bad_dimensions_and_digest(self):
        from phymeo_rendering.texture_data import TextureData
        for args in ((True, 2, RGB, "a"*64), (2, 0, RGB, "a"*64), (2, 2, bytearray(RGB), "a"*64),
                     (2, 2, RGB[:-1], "a"*64), (2, 2, RGB, "not-a-digest")):
            with self.subTest(args=args), self.assertRaises((ValueError, TypeError)):
                TextureData(*args)


if __name__ == "__main__":
    unittest.main()
