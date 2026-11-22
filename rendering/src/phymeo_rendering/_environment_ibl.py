"""Prepare the diffuse SH projection and specular GGX cube consumed by Vulkan."""
from dataclasses import dataclass
from hashlib import sha256
import json
import math


@dataclass(frozen=True, slots=True)
class NativeEnvironmentData:
    side: int
    mip_levels: int
    rgba_f32le: bytes
    diffuse_sh_f32le: bytes
    provenance: str | dict

    def __post_init__(self):
        if isinstance(self.provenance, dict):
            object.__setattr__(self, "provenance", json.dumps(self.provenance, allow_nan=False))

    @property
    def diagnostics(self):
        return json.loads(self.provenance)


def _diffuse_coefficients(pixels, yaw_degrees):
    """Integrate each constant lat-long texel over its exact solid angle.

    Coefficients are in the positive real SH basis in world coordinates.
    Cosine convolution divided by pi scales bands 0, 1 and 2 by 1, 2/3 and 1/4.
    """
    import numpy as np

    height, width = pixels.shape[:2]
    theta = np.linspace(0, np.pi, height+1)
    phi = np.linspace(-np.pi, np.pi, width+1)+np.deg2rad(yaw_degrees % 360)
    st, ct = np.sin(theta), np.cos(theta)
    az = np.diff(phi)
    cos_phi = np.diff(np.sin(phi))
    sin_phi = -np.diff(np.cos(phi))
    area = -np.diff(ct)
    sin_sq = np.diff(theta/2-np.sin(2*theta)/4)
    sin_cubed = np.diff(-ct+ct**3/3)
    sin_sq_cos = np.diff(st**3)/3
    c0 = math.sqrt(1/(4*math.pi))
    c1 = math.sqrt(3/(4*math.pi))*2/3
    c2 = math.sqrt(15/(4*math.pi))/4
    terms = [
        (c0,area,az),
        (c1,sin_sq,sin_phi), (c1,-np.diff(ct**2)/2,az), (c1,sin_sq,cos_phi),
        (c2,sin_cubed,np.diff(np.sin(phi)**2)/2),
        (c2,sin_sq_cos,sin_phi),
        (math.sqrt(5/(16*math.pi))/4,np.diff(ct-ct**3),az),
        (c2,sin_sq_cos,cos_phi),
        (math.sqrt(15/(16*math.pi))/4,sin_cubed,np.diff(np.sin(2*phi))/2),
    ]
    coeff = np.zeros((9,4), dtype="<f4")
    for index,(scale,polar,longitude) in enumerate(terms):
        # Contract the small colour axis without a panorama-sized SH tensor.
        integral = np.einsum("hwc,h,w->c",pixels,polar,longitude,optimize=True,dtype=np.float64)
        coeff[index,:3] = scale*integral
    return coeff


def _gloss(pixels, normals, roughness, yaw, samples=128):
    """Karis split-sum radiance prefilter with V=N, N·L weights and GGX NDF samples."""
    import numpy as np
    from .environment import sample_environment

    n = normals.reshape(-1,3)
    # Hammersley points, with half strata offsets on the radial dimension.
    u = (np.arange(samples)+.5)/samples
    rev = np.array([int(f"{i:07b}"[::-1],2)/128 for i in range(samples)])
    azimuth = 2*np.pi*rev
    a2 = roughness**4
    cos_h = np.sqrt((1-u)/(1+(a2-1)*u))
    sin_h = np.sqrt(np.maximum(0.,1-cos_h**2))
    # Reflection of -N at H in each tangent frame.
    dir_x = 2*cos_h*sin_h*np.cos(azimuth)
    dir_y = 2*cos_h*sin_h*np.sin(azimuth)
    dir_z = 2*cos_h**2-1
    keep = dir_z>0
    directions = np.stack((dir_x[keep],dir_y[keep],dir_z[keep]),axis=0)
    weights = dir_z[keep]/dir_z[keep].sum()
    values = np.empty_like(n)
    for start in range(0,len(n),384):
        chunk = n[start:start+384]
        up = np.zeros_like(chunk)
        polar = abs(chunk[:,2]) >= .99
        up[~polar,2] = 1
        up[polar,0] = 1
        tangent = np.cross(up,chunk)
        tangent /= np.linalg.norm(tangent,axis=1,keepdims=True)
        bitangent = np.cross(chunk,tangent)
        basis = np.stack((tangent,bitangent,chunk),axis=2)
        rays = (basis@directions).transpose(0,2,1)
        incoming = sample_environment(pixels,rays,yaw_degrees=yaw)
        values[start:start+len(chunk)] = np.einsum("msc,s->mc",incoming,weights)
    return values.reshape(normals.shape)


def prepare_native_environment(environment):
    import numpy as np
    from .environment import load_environment, sample_environment, sapien_face_directions

    pixels = load_environment(environment)
    side = environment.face_size
    levels = side.bit_length()
    cube = bytearray()
    for level in range(levels):
        size = side >> level
        roughness = level/(levels-1)
        rgba = np.ones((size,size,4),dtype="<f4")
        for face in range(6):
            directions = sapien_face_directions(face,size)
            if level == 0:
                rgb = sample_environment(pixels,directions,yaw_degrees=environment.yaw_degrees)
            else:
                rgb = _gloss(pixels,directions,roughness,environment.yaw_degrees)
            rgba[...,:3] = rgb
            cube.extend(rgba.tobytes())
    diffuse = _diffuse_coefficients(pixels,environment.yaw_degrees).tobytes()
    return NativeEnvironmentData(side,levels,bytes(cube),diffuse,
        dict(source_sha256=environment.sha256, input_size=[pixels.shape[1],pixels.shape[0]],
             strength=float(environment.strength),yaw_degrees=float(environment.yaw_degrees),
             peak_linear_radiance=float(pixels.max()),prefilter_roughness="mip/(levels-1)",
             ggx_directions_per_texel=128,gloss="split-sum radiance prefilter, V=N, normalized positive N.L",
             orientation="world Z up; Vulkan sampler ray.xzy; positive yaw turns +X toward +Y",
             diffuse="world XYZ real SH9; exact texel solid-angle projection; irradiance divided by pi",
             rgba_sha256=sha256(cube).hexdigest(),diffuse_sh_sha256=sha256(diffuse).hexdigest()))
