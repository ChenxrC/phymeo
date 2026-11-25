"""Fixed sensor display scales and PNG encoding for the local scene viewer."""
from math import isfinite
from struct import iter_unpack, pack
import zlib

from .outputs import Channel


CHANNEL_IDS = {'rgb': Channel.RGB, 'depth': Channel.FORWARD_DEPTH,
               'id': Channel.INSTANCE_ID, 'valid': Channel.VALID}


def _id_rgb(owner):
    low, high = owner & 0xffffff, owner >> 24
    color = ((low ^ (high*0x5bd1e995))*0x9e3779b1) & 0xffffff
    return (color or (0x010101 if owner else 0)).to_bytes(3, 'big')


def channel_metadata(channels):
    metadata = {
        Channel.RGB: dict(id='rgb', label='RGB', description='相机的 RGB 显示图像。'),
        Channel.FORWARD_DEPTH: dict(id='depth', label='深度', unit='m', colors=['#ffffff','#202020'],
            invalid_color='#000000', description='相机光轴深度（米）。各镜头使用固定范围，近白、远灰，未命中为黑。'),
        Channel.INSTANCE_ID: dict(id='id', label='实例 ID', invalid_color='#000000',
            description='像素中心的 uint32 实例 ID。零为背景，同一场景中的同一实例保持同色。',
            legend=[dict(id=i, color='#'+_id_rgb(i).hex()) for i in (0,1,2,3,1001)]),
        Channel.VALID: dict(id='valid', label='有效像素', description='像素中心有命中为白，背景为黑。'),
    }
    return [metadata[ch] for ch in channels]


def _sensor_pixels(raw, channel, depth_range):
    if channel is Channel.VALID:
        return raw.translate(bytes([0]+[255]*255))
    lo, hi = depth_range
    try:
        import numpy as np
    except ImportError:
        if channel is Channel.FORWARD_DEPTH:
            return bytes(round(255-223*max(0., min(1., (d-lo)/(hi-lo)))) if isfinite(d) else 0
                         for (d,) in iter_unpack('<f',raw))
        return b''.join(_id_rgb(i) for (i,) in iter_unpack('<I',raw))
    if channel is Channel.FORWARD_DEPTH:
        # Evaluate on the exact f32 plane in double precision so display
        # rounding also agrees with the standard-library implementation.
        z = np.frombuffer(raw, '<f4').astype(np.float64)
        hit = np.isfinite(z)
        gray = np.zeros(z.shape, np.uint8)
        gray[hit] = np.rint(255-223*np.clip((z[hit]-lo)/(hi-lo),0,1)).astype(np.uint8)
        return gray.tobytes()
    ids = np.frombuffer(raw, '<u4')
    mask = np.uint32(0xffffff)
    packed = (((ids & mask) ^ ((ids >> 24)*np.uint32(0x5bd1e995)))*np.uint32(0x9e3779b1)) & mask
    packed[(ids != 0) & (packed == 0)] = 0x010101
    return np.stack((packed >> 16,packed >> 8,packed),axis=-1).astype(np.uint8).tobytes()


def _chunk(tag, data):
    return pack('>I', len(data))+tag+data+pack('>I', zlib.crc32(data, zlib.crc32(tag)))


def display_png(shot, channel, depth_range):
    """Encode one selected plane. Depth display limits are fixed across exposures."""
    if tuple(p.channel for p in shot.planes) != (channel,) or channel not in CHANNEL_IDS.values():
        raise ValueError('capture must contain only the selected viewer channel')
    width, height = shot.camera.width, shot.camera.height
    raw = shot.planes[0].data
    pixels = raw if channel is Channel.RGB else _sensor_pixels(raw, channel, depth_range)
    color = channel in (Channel.RGB, Channel.INSTANCE_ID)
    stride = width*(3 if color else 1)
    if len(pixels) != height*stride:
        raise ValueError('capture dimensions differ from the viewer plane')
    rows = b''.join(b'\x00'+pixels[offset:offset+stride] for offset in range(0,len(pixels),stride))
    header = pack('>IIBBBBB', width, height, 8, 2 if color else 0, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n'+_chunk(b'IHDR',header)+_chunk(b'IDAT',zlib.compress(rows,1))+_chunk(b'IEND',b'')
