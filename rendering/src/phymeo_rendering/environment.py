"""Pinned linear Radiance panoramas for distant lighting in world-Z-up scenes.

The middle of the panorama faces +X. Longitude increases toward +Y and the
top row is near +Z. A positive yaw rotates the light about world +Z. NumPy is
loaded when preparing a map, so declaring scenes remains dependency-free.
"""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from math import isfinite
from numbers import Real
from pathlib import Path
import re

from .mesh_data import _local_path


@dataclass(frozen=True, slots=True)
class RadianceEnvironment:
    uri: str
    sha256: str
    strength: float = 1.0
    yaw_degrees: float = 0.0
    face_size: int = 128

    def __post_init__(self):
        if not isinstance(self.uri, str) or not self.uri.strip():
            raise ValueError("environment URI must name a local HDR image")
        if not isinstance(self.sha256, str) or not re.fullmatch("[0-9a-f]{64}", self.sha256):
            raise ValueError("expected lowercase SHA256 of the HDR file")
        for value in (self.strength, self.yaw_degrees):
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError("environment gain and yaw must be finite numbers")
        if not 0 <= self.strength <= 64:
            raise ValueError("environment gain outside [0,64]")
        if type(self.face_size) is not int or self.face_size not in (16, 32, 64, 128, 256):
            raise ValueError("environment face_size must be a power of two from 16 to 256")


def _take(stream, n):
    raw = stream.read(n)
    if len(raw) != n:
        raise ValueError("truncated Radiance image")
    return raw


def _decode(raw):
    import numpy as np

    stream = BytesIO(raw)
    if stream.readline(64).rstrip(b"\r\n") not in (b"#?RADIANCE", b"#?RGBE"):
        raise ValueError("unsupported Radiance signature")
    fmt = []
    while stream.tell() < 4096:
        line = stream.readline(1024)
        if line in (b"\n", b"\r\n"):
            break
        if not line.endswith(b"\n"):
            raise ValueError("invalid Radiance header")
        key = line.strip()
        if key.startswith(b"FORMAT="):
            fmt.append(key)
        elif key and not key.startswith(b"#") and b"=" in key:
            raise ValueError("unsupported radiometric header modifier")
    else:
        raise ValueError("Radiance header exceeds resource limit")
    if fmt != [b"FORMAT=32-bit_rle_rgbe"]:
        raise ValueError("expected linear 32-bit_rle_rgbe")
    match = re.fullmatch(rb"-Y ([0-9]{1,5}) \+X ([0-9]{1,5})", stream.readline(64).strip())
    if not match:
        raise ValueError("expected Radiance dimensions in -Y +X order")
    height, width = map(int, match.groups())
    if not 2 <= height <= 1024 or width != 2*height:
        raise ValueError("panorama dimensions must be 2:1, at most 2048x1024")
    rows = np.empty((height, width, 4), dtype=np.uint8)
    for row in rows:
        first = _take(stream, 4)
        if width >= 8 and first[:2] == b"\x02\x02" and first[2] < 128:
            if int.from_bytes(first[2:], "big") != width:
                raise ValueError("RGBE scanline width differs from header")
            for channel in range(4):
                position = 0
                while position < width:
                    packet = _take(stream, 1)[0]
                    count = packet-128 if packet > 128 else packet
                    if not count or position+count > width:
                        raise ValueError("invalid RGBE run length")
                    if packet > 128:
                        values = _take(stream, 1)[0]
                    else:
                        values = np.frombuffer(_take(stream, count), dtype=np.uint8)
                    row[position:position+count,channel] = values
                    position += count
        else:
            row[:] = np.frombuffer(first+_take(stream, width*4-4), dtype=np.uint8).reshape(width,4)
            if np.any(np.all(row[:,:3] == 1,axis=1)):
                raise ValueError("legacy RGBE repeat markers are unsupported")
    if stream.read(1):
        raise ValueError("trailing bytes after Radiance image")
    exponent = rows[...,3].astype(np.int32)
    linear = np.ldexp(rows[...,:3].astype(np.float32), (exponent-136)[...,None])
    linear[exponent == 0] = 0
    return linear


def load_environment(environment):
    """Return float32 world-light input after its declared linear gain."""
    import numpy as np

    if not isinstance(environment, RadianceEnvironment):
        raise TypeError("expected RadianceEnvironment")
    path = _local_path(environment.uri, suffixes=(".hdr",))
    if not path.is_file():
        raise ValueError("environment must be a regular file")
    with path.open("rb") as stream:
        raw = stream.read(16*1024**2+1)
    if len(raw) > 16*1024**2:
        raise ValueError("environment source exceeds 16 MiB")
    if sha256(raw).hexdigest() != environment.sha256:
        raise ValueError("environment source SHA256 differs")
    values = _decode(raw)
    if float(values.max())*environment.strength > 100.:
        raise ValueError("effective HDR radiance exceeds the reference shader limit of 100")
    values *= np.float32(environment.strength)
    return values


def sample_environment(panorama, directions, *, yaw_degrees):
    """Bilinear radiance lookup at arbitrary world directions; periodic longitude."""
    import numpy as np

    d = np.asarray(directions, dtype=np.float64)
    radius = np.linalg.norm(d,axis=-1)
    if np.any(radius == 0) or not np.isfinite(radius).all():
        raise ValueError("environment sample requires finite nonzero directions")
    height, width = panorama.shape[:2]
    longitude = np.arctan2(d[...,1],d[...,0]) - np.deg2rad(yaw_degrees % 360)
    x = (longitude/(2*np.pi)+.5)*width-.5
    y = np.clip(np.arccos(np.clip(d[...,2]/radius,-1,1))/np.pi*height-.5,0,height-1)
    lo, top = np.floor(x).astype(int), np.floor(y).astype(int)
    tx, ty = (x-lo)[...,None], (y-top)[...,None]
    nextrow = np.minimum(top+1,height-1)
    left, right = lo%width,(lo+1)%width
    return ((1-ty)*((1-tx)*panorama[top,left]+tx*panorama[top,right])
            + ty*((1-tx)*panorama[nextrow,left]+tx*panorama[nextrow,right]))


def sapien_face_directions(index, size):
    """World ray per Vulkan face texel for the reference miss shader's xzy swizzle."""
    import numpy as np

    axis = 2*(np.arange(size)+.5)/size-1
    s,t = np.meshgrid(axis,axis)
    one = np.ones_like(s)
    x,y,z = ((one,-t,-s),(-one,-t,s),(s,one,t),(s,-one,-t),(s,-t,one),(-s,-t,-one))[index]
    world = np.stack((x,z,y),axis=-1)
    return world/np.linalg.norm(world,axis=-1,keepdims=True)


def _encode_face(face):
    import numpy as np

    h,w = face.shape[:2]
    maximum = face.max(axis=-1)
    exponent = np.frexp(maximum)[1]
    mantissa = np.floor(np.ldexp(face, (8-exponent)[...,None])).clip(0,255)
    encoded = np.zeros((h,w,4),dtype=np.uint8)
    encoded[...,:3] = mantissa.astype(np.uint8)
    nonzero = (maximum > 0) & (exponent > -128)
    encoded[...,3] = np.where(nonzero,exponent+128,0).astype(np.uint8)
    encoded[~nonzero] = 0
    target = BytesIO()
    target.write(f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n".encode("ascii"))
    for row in encoded:
        target.write(bytes((2,2,w//256,w%256)))
        for plane in row.T:
            for start in range(0,w,128):
                block = plane[start:start+128]
                target.write(bytes((len(block),)))
                target.write(block.tobytes())
    return target.getvalue()


def write_sapien_faces(environment, directory):
    """Prepare session-owned, separately named RGBE faces for the public six-file loader."""
    pixels = load_environment(environment)
    paths, hashes = [], []
    for index,name in enumerate(("px","nx","py","ny","pz","nz")):
        directions = sapien_face_directions(index,environment.face_size)
        sampled = sample_environment(pixels,directions,yaw_degrees=environment.yaw_degrees)
        raw = _encode_face(sampled)
        target = Path(directory)/f"environment-{name}.hdr"
        with target.open("xb") as stream:
            stream.write(raw)
        paths.append(target)
        hashes.append(sha256(raw).hexdigest())
    return tuple(paths),dict(source_kind="local equirectangular Radiance RGBE", source_sha256=environment.sha256,
        source_size=[pixels.shape[1],pixels.shape[0]], face_resolution=[environment.face_size]*2,
        face_count=6,face_sha256=hashes,peak_linear_radiance=float(pixels.max()),
        strength=float(environment.strength),yaw_degrees=float(environment.yaw_degrees),
        orientation="world +Z up, panorama centre +X, increasing longitude toward +Y; sampler receives ray.xzy")
