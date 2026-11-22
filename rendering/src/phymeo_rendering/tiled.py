"""Offline pinhole-camera capture using bounded, disjoint sensor rectangles."""
from dataclasses import replace

from .backends import UnsupportedRequestError
from .cameras import validate_exposure
from .outputs import Capture, ImagePlane
from .pipeline import validate_capture_request


def capture_tiled(session, frame, request, tile_width, tile_height):
    """Render the same immutable pose with sub-cameras covering one full sensor.

    A crop retains the focal lengths, pose and clipping planes. Subtracting its
    top-left pixel address from the principal point preserves every optical ray.
    Planes are assembled by copying rows with no resampling or blending.
    Stochastic backends may seed their integrator from tile-local pixel IDs.
    Screen-space effects using neighbouring pixels require separate validation.
    """
    validate_capture_request(request)
    camera=request.camera
    validate_exposure(frame,camera)
    for size in (tile_width,tile_height):
        if type(size) is not int:
            raise TypeError("tile dimensions must be integers")
        if size<1:
            raise ValueError("tile dimensions must be positive")
    cols=(camera.width+tile_width-1)//tile_width
    rows=(camera.height+tile_height-1)//tile_height
    if max(camera.width,camera.height)>4096 or rows*cols>64:
        raise ValueError("tile capture is bounded to a 4096-square raster and 64 tiles")
    if not session.supports(request):
        raise UnsupportedRequestError("session does not support the full camera request")
    tiles=[]
    for y in range(0,camera.height,tile_height):
        for x in range(0,camera.width,tile_width):
            local=replace(camera,width=min(tile_width,camera.width-x),
                          height=min(tile_height,camera.height-y),cx_px=camera.cx_px-x,cy_px=camera.cy_px-y)
            local_request=replace(request,camera=local)
            if not session.supports(local_request):
                raise UnsupportedRequestError("session does not support a requested tile")
            tiles.append((x,y,local_request))
    planes={}
    for x,y,local_request in tiles:
        tile=session.capture(frame,local_request)
        local=local_request.camera
        if (not isinstance(tile,Capture) or tile.camera!=local or tile.frame_key!=frame.key
            or tile.time_s!=frame.time_s or tuple(p.channel for p in tile.planes)!=request.channels):
            raise ValueError("tile exposure does not match the requested source")
        for plane in tile.planes:
            bytes_per_pixel=len(plane.data)//(local.width*local.height)
            if plane.channel not in planes:
                planes[plane.channel]=bytearray(camera.width*camera.height*bytes_per_pixel)
            target=planes[plane.channel]
            source=memoryview(plane.data)
            stride=local.width*bytes_per_pixel
            for row in range(local.height):
                start=((y+row)*camera.width+x)*bytes_per_pixel
                target[start:start+stride]=source[row*stride:(row+1)*stride]
    return Capture(frame.key,frame.time_s,camera,
                   tuple(ImagePlane(channel,bytes(planes.pop(channel))) for channel in request.channels))
