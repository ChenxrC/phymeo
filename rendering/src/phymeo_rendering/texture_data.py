"""Bounded, explicitly loaded, hash-pinned sRGB color images.

Only local PNG/JPEG and opaque 8-bit color are admitted. Pillow is imported at
load_texture(), never by importing the package or declaring a material. Output
owns tightly packed RGB8, top row first. No flip, EXIF transpose, ICC conversion,
resizing, mip generation, network fetch or renderer is involved. glTF defines
(0,0) at the top left and ignores image color-space metadata in favor of the
material's declared transfer function:
https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#images
"""
from dataclasses import dataclass
import hashlib
import io
import stat
import warnings

from .assets import TextureMap
from .mesh_data import _integer, _local_path

_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_DIMENSION = 4096
_MAX_PIXELS = 4096 * 4096


def _dimensions(width, height):
    _integer(width, "texture width", 1, _MAX_DIMENSION)
    _integer(height, "texture height", 1, _MAX_DIMENSION)
    if width * height > _MAX_PIXELS:
        raise ValueError("texture pixel count exceeds limit")


@dataclass(frozen=True, slots=True)
class TextureData:
    """Owned, top-row-first, opaque sRGB8 texels and original-file identity."""

    width: int
    height: int
    rgb8: bytes
    source_sha256: str

    def __post_init__(self):
        _dimensions(self.width, self.height)
        if not isinstance(self.rgb8, bytes):
            raise TypeError("texture rgb8 must be immutable bytes")
        if len(self.rgb8) != self.width * self.height * 3:
            raise ValueError("texture rgb8 byte length does not match dimensions")
        if (not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.source_sha256)):
            raise ValueError("texture source_sha256 must be a lowercase SHA256 hex digest")


def load_texture(texture: TextureMap) -> TextureData:
    """Verify original bytes before decoding; no mutable image escapes.

    PNG RGB/RGBA/palette/8-bit grayscale and JPEG RGB/grayscale are supported.
    Transparency (including palette tRNS), 16-bit images, animation and CMYK are
    rejected rather than discarded. Ancillary ICC/gamma and EXIF orientation do
    not alter the explicit sRGB/top-row contract. File I/O errors stay OS errors.
    """
    if not isinstance(texture, TextureMap):
        raise TypeError("texture must be a TextureMap")
    path = _local_path(texture.uri, suffixes=(".png", ".jpg", ".jpeg"))
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("texture source must be a regular file")
    if info.st_size > _MAX_FILE_BYTES:
        raise ValueError("texture source exceeds byte limit")
    with path.open("rb") as stream:
        raw = stream.read(_MAX_FILE_BYTES + 1)
    if len(raw) > _MAX_FILE_BYTES:
        raise ValueError("texture source exceeds byte limit")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != texture.sha256:
        raise ValueError(f"texture SHA256 digest mismatch: {path}")
    expected = "PNG" if path.suffix.lower() == ".png" else "JPEG"
    if expected == "PNG":
        if len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
            raise ValueError("texture PNG signature/header is invalid")
        if raw[24] == 16:
            raise ValueError("texture PNG 16-bit data is unsupported")
    elif not raw.startswith(b"\xff\xd8\xff"):
        raise ValueError("texture JPEG signature is invalid")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required only when explicitly loading a TextureMap") from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != expected:
                    raise ValueError("texture format does not match its explicit PNG/JPEG URI")
                _dimensions(*image.size)
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("animated textures are unsupported")
                if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
                    raise ValueError(f"texture mode {image.mode!r} is unsupported")
                image.verify()
            # verify() invalidates the decoder; reopen the same owned bytes,
            # not the original pathname, so file changes cannot replace input.
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                if "A" in image.getbands() or "transparency" in image.info:
                    with image.convert("RGBA") as rgba:
                        if rgba.getchannel("A").getextrema() != (255, 255):
                            raise ValueError("texture must be fully opaque; alpha is not discarded")
                with image.convert("RGB") as rgb:
                    return TextureData(rgb.width, rgb.height, rgb.tobytes(), digest)
    except (OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f"invalid or unsupported texture image: {path}: {exc}") from exc
