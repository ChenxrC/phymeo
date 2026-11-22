"""Reference display conversion from a real linear HDR readback, not Color.

SAPIEN 3.0.3 default/Color uses gamma 2.2. Reading Lighting (raster) or
HdrColor (PT) avoids that display transform and permits exactly one sRGB OETF.
"""

from math import isfinite
from functools import lru_cache
import struct


def encode_linear_rgb(values):
    """Flattened linear RGB -> owned sRGB8; exposure=1, clipping, half-up UNORM.

    This reference conversion intentionally uses Python double arithmetic.
    It accepts no NaN/Inf, performs no automatic exposure and no tone mapping.
    """
    output = bytearray()
    for value in values:
        value = float(value)
        if not isfinite(value):
            raise ValueError("linear RGB readback must be finite")
        value = min(1.0, max(0.0, value))
        srgb = 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1.0 / 2.4) - 0.055
        output.append(min(255, int(srgb * 255.0 + 0.5)))
    return bytes(output)


@lru_cache(maxsize=1)
def _float32_code_boundaries():
    """First binary32 input which the scalar transform assigns each code 1..255."""
    import numpy as np

    boundaries = []
    for code in range(1,256):
        # Positive IEEE floats are ordered by their unsigned bit patterns.
        low,high = 0,0x3f800000
        while low < high:
            middle = (low+high)//2
            value = struct.unpack("<f",struct.pack("<I",middle))[0]
            if encode_linear_rgb((value,))[0] >= code:
                high = middle
            else:
                low = middle+1
        boundaries.append(low)
    return np.frombuffer(struct.pack("<255I",*boundaries),dtype="<f4")


def encode_linear_image(values, *, exposure=1.0, display_transform="clip"):
    """Encode an H×W×3 readback; clip preserves the original quantizer exactly."""
    import numpy as np

    if display_transform not in ("clip", "pbr-neutral"):
        raise ValueError("display_transform must be clip or pbr-neutral")
    image = np.asarray(values)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("linear RGB image must have three channels")
    if not np.isfinite(image).all():
        raise ValueError("linear RGB readback must be finite")
    if display_transform == "pbr-neutral":
        from ._sapien_neutral import encode_neutral
        return encode_neutral(image, exposure)
    if image.dtype != np.float32:
        return encode_linear_rgb(np.clip(image.astype(np.float64) * exposure, 0., 1.).flat)
    boundaries = _float32_code_boundaries()
    # Search in row blocks so the temporary intp image stays small at 4K.
    rows = max(1, 65536//max(1,image.shape[1]))
    output = bytearray()
    for start in range(0, image.shape[0], rows):
        block = image[start:start + rows]
        if exposure != 1.0:
            # Multiply before the display transform. Use f64 for the product
            # so large finite binary32 operands can saturate without overflow.
            block = np.clip(block.astype(np.float64) * exposure, 0., 1.).astype(np.float32)
        output.extend(np.searchsorted(boundaries, block, side="right").astype(np.uint8).tobytes())
    return bytes(output)
