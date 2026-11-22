"""Static resident studio ABI and source-derived rectangular emitter bases."""
import ctypes as ct
import math
from hashlib import sha256

from ..scene import Pose, compose_pose
from ..studio import StudioLighting
from ._sapien_studio import studio_recipe, uniform_hdr


class StudioPanel(ct.Structure):
    _fields_ = [(name, ct.c_float * 4) for name in
                ('position', 'half_u', 'half_v', 'emission')]


class NativeStudio(ct.Structure):
    _fields_ = [('environment', ct.c_float * 4), ('panels', StudioPanel * 2)]


# Source inspection is pinned to file revisions, with installed 3.0.3 shader
# and coordinate-header hashes. No upstream runtime is imported for packing.
_BASIS_SOURCES = (
    dict(url='https://github.com/haosulab/SAPIEN/blob/8360d517770f24f00bacef7d906ddf1380b8967c/src/sapien_renderer/light_component.cpp',
         fact='internalUpdate composes entity global pose with POSE_GL_TO_ROS before setting the renderer light transform'),
    dict(url='https://github.com/haosulab/sapien-vulkan-2/blob/62e2511930e43db9fc7d384a49da4f374b35f701/src/scene/light.cpp',
         fact='centered rectangle half width along GL +X, half height along GL +Y at shape angle pi/2'),
    dict(reference='installed SAPIEN 3.0.3 public headers and shipped rt shader',
         pose_header_sha256='122446114db45d1ad78e47394a31e86cb6c01efc07f1a30bcf657d81f64dce0e',
         rt_shading_sha256='4f7421b369f75e14fa22dc623b770f346800b8c05470fc5d682ccb4f2d52efc4',
         fact='GL_TO_ROS quaternion wxyz=(-.5,-.5,.5,.5); RT shader uses cross(edge0,edge1) dotted with point-to-light'),
)


def pack_studio(studio):
    """Return actual FP32 payload and detached geometry/encoding evidence.

    GL shape +X/+Y map to entity -Y/+Z. Their cross product is entity -X,
    opposite emitted travel (+X). Reflecting the centered first half-axis
    preserves exactly the same rectangle while making U cross V point along
    emitted travel. Apply the same shortest rotation from entity +X to the
    recipe's world emission direction. Half axes include their metric sizes.
    """
    if not isinstance(studio, StudioLighting):
        raise TypeError('studio must be StudioLighting')
    if ct.sizeof(NativeStudio) != 144:
        raise RuntimeError('native studio ABI structure layout mismatch')
    recipe=studio_recipe(studio)
    payload=NativeStudio()
    hdr=uniform_hdr(studio.environment_radiance)
    environment=hdr[-4] * 2.0**(hdr[-1]-136)
    payload.environment[:]=(environment,environment,environment,1.)
    panels=[]
    for target,panel in zip(payload.panels,recipe['panels'],strict=True):
        dx,dy,dz=panel['emission_direction_world']
        quaternion=(0.,-dz,dy,1.+dx)
        length=math.hypot(*quaternion)
        if length==0:
            raise ValueError('studio recipe cannot select an ambiguous opposite-X shortest rotation')
        rotation=Pose(orientation_xyzw=tuple(value/length for value in quaternion))
        width,height=panel['half_size_m']
        half_u=compose_pose(rotation,Pose((0.,width,0.))).position_m
        half_v=compose_pose(rotation,Pose((0.,0.,height))).position_m
        target.position[:]=(*panel['position_m'],0.)
        target.half_u[:]=(*half_u,0.)
        target.half_v[:]=(*half_v,0.)
        target.emission[:]=(*panel['radiance_linear_rgb'],0.)
        if not all(math.isfinite(value) for block in (target.position,target.half_u,target.half_v,target.emission) for value in block):
            raise ValueError('native studio FP32 packing is nonfinite')
        panels.append(dict(name=panel['name'],position_m=list(target.position[:3]),
            half_u_world_m=list(target.half_u[:3]),half_v_world_m=list(target.half_v[:3]),
            emission_direction_world=panel['emission_direction_world'],
            half_size_m=panel['half_size_m'],radiance_linear_rgb=list(target.emission[:3]),
            visible_geometry=False))
    return payload,dict(preset=recipe['preset'],center_m=list(studio.center_m),scale_m=studio.scale_m,
        panel_size_scale=studio.panel_size_scale,panel_strength=studio.panel_strength,
        panels=panels,basis_sources=[dict(item) for item in _BASIS_SOURCES],
        basis='entity shortest_rotation(+X to direction); halfU=R*(0,width,0), halfV=R*(0,0,height); U cross V emits',
        rectangle_equivalence='same authored pi/2 rectangle; first GL edge reflected for emission-oriented normal',
        arithmetic_scope='FP64 host basis packed to FP32; not bitwise equivalence to SAPIEN FP32 quaternion/trigonometric evaluation',
        environment=dict(source_requested_radiance=studio.environment_radiance,
            radiance_linear_rgb=list(payload.environment[:3]),enabled=True,
            quantization='same RGBE shared 8-bit mantissa decode as reference uniform HDR',
            source_hdr_sha256=sha256(hdr).hexdigest()),
        declared_scene_lights='retained unchanged',
        units='metres; relative linear radiance, not calibrated photometric lux',
        evaluation='native resident studio shader; distinct estimator from SAPIEN reference path tracing',
        payload_bytes=ct.sizeof(payload),payload_sha256=sha256(bytes(payload)).hexdigest())
