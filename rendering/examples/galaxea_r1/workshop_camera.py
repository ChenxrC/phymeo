"""Three-part photography for the R1 transfer station and its rendered evidence."""
from itertools import product
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def camera_for_workshop(frame, layout, i, count, *, size=(1280, 720)):
    """Look from +X toward the +X-facing R1. Keep all contact in one locked shot."""
    from _panda_scene import look_at
    from phymeo_rendering.cameras import Camera

    if count < 2 or not 0 <= i < count:
        raise ValueError("expected an in-range workshop storyboard frame")
    width,height = size
    t = i / (count-1)
    x,y = (np.asarray(layout.pick_xy)+layout.place_xy)*.5
    z = layout.support_z

    if t < .16:
        # From the aisle, the side of the bench leaves the chassis visible.
        # Offset the framing towards the work rather than centring just the R1.
        camera_id = "workshop-establish"
        eye, aim = (x+2.34,y-2.48,z+.93), (x-.29,y-.14,z+.025)
        focal = 1120.
    else:
        # Face the positive X end of the payload. The fingers separate along
        # Y, while the downward viewing angle exposes the nest and its shadow.
        # Attach neither camera nor target to the wrist or moving workpiece.
        camera_id = "workshop-transfer"
        eye = np.array((x+1.48,y+.23,z+.79))
        aim = np.array((x-.025,y,z+.145))
        focal = 1320.
        if t >= .86:
            # Release completes at .84. Ease back on this same camera after
            # the retreat begins, retaining the deposited workpiece in view.
            u = (t-.86)/.14
            travel = u*u*u*(10. + u*(-15. + 6.*u))
            eye += travel*np.array((.36,.055,.11))
            aim += travel*np.array((-.045,-.015,.045))
    f = focal * height / 720.
    return Camera(camera_id,frame.key,look_at(eye,aim,(0.,0.,1.)),width,height,
                  f,f,width/2,height/2,.025,24.)


def workshop_action_box(camera, layout):
    """Project one fixed volume containing both nests, moving jaws and table shadows."""
    from _panda_scene import pose_matrix

    sites = np.asarray((layout.pick_xy,layout.place_xy))
    length,width,height = layout.payload_size
    low = sites.min(axis=0) - (.5*length+.09,.5*width+.09)
    high = sites.max(axis=0) + (.5*length+.09,.5*width+.09)
    box = np.array(list(product((low[0],high[0]),(low[1],high[1]),
                                (layout.support_z-.06,layout.support_z+height+layout.lift_m+.14))))
    camera_world = pose_matrix(camera.world_pose)
    camera_points = (box-camera_world[:3,3]) @ camera_world[:3,:3]
    depth = -camera_points[:,2]
    if np.any(depth <= camera.near_m) or np.any(depth >= camera.far_m):
        raise ValueError("work volume falls outside the camera clipping range")
    u = camera.cx_px+camera.fx_px*camera_points[:,0]/depth
    v = camera.cy_px-camera.fy_px*camera_points[:,1]/depth
    rect = (math.floor(u.min()),math.floor(v.min()),math.ceil(u.max()),math.ceil(v.max()))
    if not (0 <= rect[0] < rect[2] <= camera.width and 0 <= rect[1] < rect[3] <= camera.height):
        raise ValueError(f"work volume is cropped by {camera.id}: {rect}")
    return rect


def make_stage_board(output, panels, *, crop_box=None):
    """Place three (label, frame index, raw PNG path) entries at one common scale.

    For operation frames use workshop_action_box on their locked camera. The
    same crop in all three views keeps the part motion and shadows comparable.
    """
    panels = list(panels)
    if len(panels) != 3 or not panels[0][1] < panels[1][1] < panels[2][1]:
        raise ValueError("expected three chronological rendered workshop frames")
    gap, col_width = 15, 475
    canvas = None
    for family in ("DejaVuSans.ttf","arial.ttf"):
        try:
            title = ImageFont.truetype(family,22)
            detail = ImageFont.truetype(family,16)
            break
        except OSError:
            continue
    else:
        title = detail = ImageFont.load_default()
    for index,(label,frame,path) in enumerate(panels):
        with Image.open(path) as source:
            if source.format != "PNG" or source.mode != "RGB":
                raise ValueError(f"expected original RGB PNG: {path}")
            if index == 0:
                source_size = source.size
                bounds = crop_box if crop_box is not None else (0,0,*source_size)
                x0,y0,x1,y1 = bounds
                if not (0 <= x0 < x1 <= source.width and 0 <= y0 < y1 <= source.height):
                    raise ValueError(f"crop exceeds source raster: {bounds}")
                picture_size = (col_width,round(col_width*(y1-y0)/(x1-x0)))
                canvas = Image.new("RGB",(3*col_width+4*gap,picture_size[1]+64),"#182229")
                draw = ImageDraw.Draw(canvas)
            elif source.size != source_size:
                raise ValueError("workshop source frames differ in resolution")
            image = source.crop(bounds).resize(picture_size,Image.Resampling.LANCZOS)
        left = gap+index*(col_width+gap)
        draw.text((left,12),label,font=title,fill="#ffffff")
        frame_label = f"Frame {frame}"
        draw.text((left+col_width-draw.textlength(frame_label,font=detail),17),frame_label,
                  font=detail,fill="#c5d1d5")
        canvas.paste(image,(left,49))
    canvas.save(Path(output))
