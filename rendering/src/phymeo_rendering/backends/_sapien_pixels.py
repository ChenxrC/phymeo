"""Reference display conversion from a real linear HDR readback, not Color.

SAPIEN 3.0.3 default/Color uses gamma 2.2. Reading Lighting (raster) or
HdrColor (PT) avoids that display transform and permits exactly one sRGB OETF.
"""
from math import isfinite


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
        srgb = 12.92 * value if value <= 0.0031308 else 1.055 * value**(1.0 / 2.4) - 0.055
        output.append(min(255, int(srgb * 255.0 + 0.5)))
    return bytes(output)
