"""Copy requested native attachments into immutable public image planes."""

from ..outputs import Channel, ImagePlane


def _image_planes(raw, pixels, channels, attachment_mask):
    source = memoryview(raw).cast("B")
    requested = set(channels)
    values = {}
    offset = 0
    for bit, (channel, width) in enumerate(zip(Channel, (4, 4, 4, 1), strict=True)):
        if not attachment_mask & (1 << bit):
            continue
        length = pixels * width
        if channel in requested:
            portion = source[offset:offset + length]
            if channel is Channel.RGB:
                rgb = bytearray(portion)
                del rgb[3::4]
                values[channel] = bytes(rgb)
            else:
                values[channel] = bytes(portion)
        offset += length
    if offset != len(source) or set(values) != requested:
        raise ValueError("native attachment packing does not match requested channels")
    return tuple(ImagePlane(channel, values[channel]) for channel in channels)
