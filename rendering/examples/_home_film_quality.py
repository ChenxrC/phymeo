"""Supersampled product-film RGB and depth-layer aperture reconstruction.

The native 4× MSAA path delivers tone-mapped sRGB8. All resampling here uses
linear display light after decoding that transfer. Hidden background radiance
is estimated from visible neighbours in the same depth layer. For reference,
capture_lens_reference traces actual off-axis aperture exposures instead.
"""
from dataclasses import replace
from functools import lru_cache
from math import ceil, cos, isfinite, pi, sin, sqrt
from time import perf_counter

import cv2
import numpy as np

from _panda_scene import pose_matrix
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath


MAX_COC_RADIUS_PX = 16.
_EDGES = np.array((-16.,-9.,-4.,-1.5,-.5,.5,1.5,4.,9.,16.), np.float32)
_SRGB = np.arange(256,dtype=np.float32)/255.
_DECODE = np.where(_SRGB<=.04045,_SRGB/12.92,((_SRGB+.055)/1.055)**2.4)


def _encode(linear):
    x = np.clip(linear,0.,1.)
    s = np.where(x<=.0031308, 12.92*x, 1.055*np.power(x,1./2.4)-.055)
    return np.rint(255*s).astype(np.uint8)


def focus_distance(camera, target_xyz):
    """Metres along camera forward, to a world-space point of interest."""
    target = np.asarray(target_xyz,dtype=float)
    if target.shape != (3,):
        raise ValueError('a focus target needs three coordinates in metres')
    m = pose_matrix(camera.world_pose)
    dist = float((target-m[:3,3]) @ -m[:3,2])
    if not isfinite(dist) or dist<=0:
        raise ValueError('focus target must be in front of the lens')
    return dist


def _geometry(camera, scale, focus, aperture):
    if type(scale) is not int or scale<1:
        raise ValueError('scale must be a positive integer')
    if not isfinite(focus) or focus<=0 or not isfinite(aperture) or aperture<0:
        raise ValueError('focus must be positive; aperture radius must be finite and nonnegative')
    return replace(camera,width=camera.width*scale,height=camera.height*scale,
                   fx_px=camera.fx_px*scale,fy_px=camera.fy_px*scale,
                   cx_px=camera.cx_px*scale,cy_px=camera.cy_px*scale)


def _capture(session, frame, camera, channels):
    result = session.capture(frame,CaptureRequest(camera,RenderPath.RASTER,channels))
    if result.frame_key!=frame.key or result.time_s!=frame.time_s or result.camera!=camera:
        raise RuntimeError('native exposure differs from the requested frame or camera')
    return {p.channel:p.data for p in result.planes}


def _area(image, w, h):
    return image if image.shape[:2]==(h,w) else cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=32)
def _disk(radius, aspect):
    """Pixel-integrated circular pupil (elliptical for nonsquare camera pixels)."""
    if radius<=.5 and radius*aspect<=.5:
        return None
    ry = radius*aspect
    y,x = np.ogrid[-ceil(ry):ceil(ry)+1,-ceil(radius):ceil(radius)+1]
    kernel = np.zeros((len(y),x.shape[1]),np.float32)
    subs = (np.arange(8)+.5)/8-.5
    for dy in subs:
        for dx in subs:
            kernel += ((x+dx)/radius)**2+((y+dy)/ry)**2<=1.
    return kernel/kernel.sum()


def _spread(image,radius,aspect):
    kernel = _disk(radius,aspect)
    if kernel is None:
        return image
    return cv2.filter2D(image,-1,kernel,borderType=cv2.BORDER_REPLICATE)


def _layered(rgb, depth, camera, scale, focus, aperture):
    """Box-resolve same-layer subpixels; spread covered colors back to front."""
    w,h = camera.width,camera.height
    signed = (1./depth - 1./focus) * (aperture*camera.fx_px)
    peak = float(np.max(np.abs(signed)))
    capped = float(np.count_nonzero(np.abs(signed)>MAX_COC_RADIUS_PX)/signed.size)
    sharp = float(np.count_nonzero(np.abs(signed)<=.5)/signed.size)
    metrics = dict(requested_peak_coc_radius_px=peak, coc_limited_pixel_fraction=capped,
                   subpixel_coc_fraction=sharp, max_coc_radius_px=MAX_COC_RADIUS_PX)
    # Return exactly the area integration for a pinhole or a wholly focused image.
    linear = _DECODE[rgb]
    if peak<=.5:
        return _area(linear,w,h),metrics
    np.clip(signed,-MAX_COC_RADIUS_PX,MAX_COC_RADIUS_PX,out=signed)
    bands = np.searchsorted(_EDGES[1:-1],signed).astype(np.uint8)
    radius = np.abs(signed)
    groups = bands.reshape(h,scale,w,scale)
    least,most = groups.min(axis=(1,3)),groups.max(axis=(1,3))
    result = np.zeros((h,w,4),np.float32)
    nearer = np.ones((h,w),np.float32)
    aspect = float(camera.fy_px/camera.fx_px)
    extend = ceil(MAX_COC_RADIUS_PX*max(1.,aspect))
    padding = 2*extend+2
    for i,(a,b) in enumerate(zip(_EDGES,_EDGES[1:])):
        x,y,dw,dh = cv2.boundingRect(((least<=i)&(most>=i)).astype(np.uint8))
        if not dw:
            continue
        x0,y0 = max(0,x-padding),max(0,y-padding)
        x1,y1 = min(w,x+dw+padding),min(h,y+dh+padding)
        out = np.s_[y0:y1,x0:x1]
        fine = np.s_[y0*scale:y1*scale,x0*scale:x1*scale]
        mask = (bands[fine]==i).astype(np.float32)
        if not mask.any():
            continue
        width,height = x1-x0,y1-y0
        rgba = np.empty((height,width,4),np.float32)
        rgba[...,3] = _area(mask,width,height)
        rgba[...,:3] = _area(linear[fine]*mask[...,None],width,height)
        lo,hi = sorted((float(abs(a)),float(abs(b))))
        # The central group covers |CoC| <= 0.5; its exact PSF is a pixel.
        weight = None if lo==hi else _area(np.clip((radius[fine]-lo)/(hi-lo),0.,1.)*mask,width,height)
        remaining = nearer[out]
        np.maximum(remaining-rgba[...,3],0.,out=remaining)
        if np.any(remaining>0):
            # A farther surface can continue underneath a nearer silhouette.
            # Estimate only those missing samples, from this layer's own color;
            # never expand it onto a known farther surface or smear a sharp edge.
            k = 2*extend+1
            support = cv2.blur(rgba,(k,k),borderType=cv2.BORDER_REPLICATE)
            extra = remaining*(support[...,3]>1e-5)
            factor = extra/np.maximum(support[...,3],1e-5)
            rgba[...,:3] += support[...,:3]*factor[...,None]
            rgba[...,3] += extra
            if weight is not None:
                weight += cv2.blur(weight,(k,k),borderType=cv2.BORDER_REPLICATE)*factor
        if weight is not None:
            # Interpolating the neighbouring pupil kernels prevents a sudden
            # radius jump when a surface moves through a depth-band boundary.
            fraction = np.clip(weight/np.maximum(rgba[...,3],1e-7),0.,1.)[...,None]
            layer = _spread(rgba*(1.-fraction),lo,aspect) + _spread(rgba*fraction,hi,aspect)
            np.maximum(layer,0.,out=layer)
            layer /= np.maximum(layer[...,3:4],1.)
        else:
            layer = rgba
        canvas = result[out]
        canvas *= 1.-layer[...,3:4]
        canvas += layer
    # At silhouettes without observable background, normalize available light.
    result[...,:3] /= np.maximum(result[...,3:4],1e-7)
    return result[...,:3],metrics


def capture_film_frame(session, frame, camera, *, focus_distance_m, aperture_radius_m=.003, scale=2, inspect=False):
    """Return output-size RGB bytes, optional true center uint32 IDs and optics metadata.

    ``aperture_radius_m`` is the radius, not diameter. Radius zero selects just
    supersampled antialiasing. The practical blur bound is 16 output pixels
    horizontally; the vertical radius follows the ratio of fy/fx.
    """
    sensor = _geometry(camera,scale,focus_distance_m,aperture_radius_m)
    channels = (Channel.RGB,Channel.FORWARD_DEPTH) if aperture_radius_m else (Channel.RGB,)
    buffers = _capture(session,frame,sensor,channels)
    rgb = np.frombuffer(buffers[Channel.RGB],np.uint8).reshape(sensor.height,sensor.width,3)
    start = perf_counter()
    if aperture_radius_m:
        depth = np.frombuffer(buffers[Channel.FORWARD_DEPTH],'<f4').reshape(sensor.height,sensor.width)
        resolved,stats = _layered(rgb,depth,camera,scale,focus_distance_m,aperture_radius_m)
    else:
        resolved = _area(_DECODE[rgb],camera.width,camera.height)
        stats = dict(requested_peak_coc_radius_px=0.,coc_limited_pixel_fraction=0.,subpixel_coc_fraction=1.,
                     max_coc_radius_px=MAX_COC_RADIUS_PX)
    pixels = _encode(resolved).tobytes()
    stats.update(method='supersampled depth layers with pixel-integrated pupil' if aperture_radius_m else 'pinhole area resolve',
                 focus_distance_m=float(focus_distance_m),aperture_radius_m=float(aperture_radius_m),
                 sampling_space='linear display light decoded from native MSAA sRGB8',
                 native_size=[sensor.width,sensor.height],subpixels_per_output=scale*scale,
                 processing_ms=1000*(perf_counter()-start))
    ids = None
    if inspect:
        center = _capture(session,frame,camera,(Channel.INSTANCE_ID,))
        ids = np.frombuffer(center[Channel.INSTANCE_ID],'<u4').reshape(camera.height,camera.width)
    return pixels,ids,stats


def _lens_camera(camera, dx, dy, focus):
    m = pose_matrix(camera.world_pose)
    origin = m[:3,3]+dx*m[:3,0]+dy*m[:3,1]
    return replace(camera,world_pose=replace(camera.world_pose,position_m=tuple(origin)),
                   cx_px=camera.cx_px+camera.fx_px*dx/focus,cy_px=camera.cy_px-camera.fy_px*dy/focus)


def capture_lens_reference(session, frame, camera, *, focus_distance_m, aperture_radius_m, samples=12, scale=1):
    """Actual translated-camera reference. Each exposure reveals its own visibility.

    Use on selected frames or smaller calibrated crops; physically traced CoCs
    are uncapped here. Opposite pupil pairs keep the optical centroid fixed.
    """
    if type(samples) is not int or samples<2 or samples%2:
        raise ValueError('lens reference requires a positive even sample count >= 2')
    sensor = _geometry(camera,scale,focus_distance_m,aperture_radius_m)
    total = np.zeros((camera.height,camera.width,3),np.float32)
    for i in range(samples//2):
        angle = pi*(3.-sqrt(5.))*i
        radius = aperture_radius_m*sqrt((i+.5)/(samples//2))
        for sign in (-1,1):
            lens = _lens_camera(sensor,sign*radius*cos(angle),sign*radius*sin(angle),focus_distance_m)
            raw = _capture(session,frame,lens,(Channel.RGB,))[Channel.RGB]
            rgb = np.frombuffer(raw,np.uint8).reshape(sensor.height,sensor.width,3)
            total += _area(_DECODE[rgb],camera.width,camera.height)
    return _encode(total/samples).tobytes(),dict(method='offset lens exposures',lens_samples=samples,scale=scale,
                aperture_radius_m=float(aperture_radius_m),focus_distance_m=float(focus_distance_m))
