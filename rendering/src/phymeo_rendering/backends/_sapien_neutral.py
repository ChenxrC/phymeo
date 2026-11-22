"""Khronos PBR Neutral for optional CPU reference display.

Copyright 2024 The Khronos Group, Inc.
SPDX-License-Identifier: Apache-2.0

Modified for Phymeo in 2026: bounded-row FP64 NumPy adaptation, finite exposed
readback validation, and explicit nonnegative input clamping before mapping.

Pinned upstream sample implementation and its license attribution:
https://github.com/KhronosGroup/ToneMapping/blob/b5a2eed5ddf6c2227090449399de9c7affb9e4c9/PBR_Neutral/pbrNeutral.glsl
https://github.com/KhronosGroup/ToneMapping/blob/b5a2eed5ddf6c2227090449399de9c7affb9e4c9/.reuse/dep5

A full Apache-2.0 license is already distributed with this package at
phymeo_rendering/_panda_assets/LICENSE (source: rendering/assets/panda/LICENSE).
The Khronos attribution above is specific to this adaptation; the Panda
asset notices and their pinned bytes remain unchanged.
"""
from math import isfinite
from numbers import Real


SOURCE = "https://github.com/KhronosGroup/ToneMapping/blob/b5a2eed5ddf6c2227090449399de9c7affb9e4c9/PBR_Neutral/pbrNeutral.glsl"


def encode_neutral(image, exposure):
    """Nonnegative linear Rec.709 -> PBR Neutral -> one standard sRGB OETF."""
    import numpy as np

    if isinstance(exposure, bool) or not isinstance(exposure, Real):
        raise TypeError("exposure must be a real number")
    try:
        valid = isfinite(exposure) and 0 <= exposure <= 3.4028234663852886e38
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("exposure must be finite nonnegative FP32")
    rows = max(1, 65536//max(1, image.shape[1]))
    output = bytearray()
    for start in range(0, image.shape[0], rows):
        with np.errstate(over="ignore", invalid="ignore"):
            color = image[start:start+rows].astype(np.float64)*float(exposure)
        if not np.isfinite(color).all():
            raise ValueError("exposed linear RGB readback must be finite")
        np.maximum(color, 0., out=color)
        minimum = color.min(axis=-1)
        offset = np.full_like(minimum, .04)
        toe = minimum < .08
        low = minimum[toe]
        offset[toe] = low-6.25*low*low
        color -= offset[..., None]
        peak = color.max(axis=-1)
        compress = peak >= .76
        if np.any(compress):
            selected = color[compress]
            maximum = peak[compress]
            shoulder = 1.-.24*.24/(maximum-.52)
            chroma_weight = 1./(.15*(maximum-shoulder)+1.)
            selected *= (shoulder/maximum*chroma_weight)[:, None]
            selected += (shoulder*(1.-chroma_weight))[:, None]
            color[compress] = selected
        # Floating roundoff may reach an endpoint. There is no further look,
        # automatic exposure, gamut conversion or spatial/temporal filtering.
        np.clip(color, 0., 1., out=color)
        display = np.where(color <= .0031308, 12.92*color,
                           1.055*np.power(color, 1./2.4)-.055)
        output.extend(np.floor(np.clip(display, 0., 1.)*255.+.5).astype(np.uint8).tobytes())
    return bytes(output)
