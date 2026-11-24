"""Three fixed camera setups for the R1 conveyor sorting cell."""
from .workshop_camera import make_stage_board


def camera_for_sorting(frame, layout, i, count, *, size=(1280, 720)):
    """Show the incoming belt, preserve the handling axis, then reveal all four slots."""
    from _panda_scene import look_at
    from phymeo_rendering.cameras import Camera

    if count < 2 or not 0 <= i < count:
        raise ValueError("expected an in-range conveyor frame")
    width, height = size
    t = i / (count - 1)
    pick_x, pick_y = layout.pick_xy
    (ax, ay), (bx, by) = layout.bin_centers
    bin_x, bin_y = (ax + bx) / 2, (ay + by) / 2
    z = layout.support_z

    # All three cameras stay on the +X/-Y side. Incoming parts travel across
    # the image from right to left, and both fingers remain visible at pickup.
    if t < .062:
        name = "sorting-establish"
        aim = (.14, pick_y + .16, z + .015)
        eye = (aim[0] + 2.90, aim[1] - 1.68, aim[2] + 1.11)
        focal = 1060.
    elif t < .890:
        # First cut is during the empty approach, before the first contact.
        # The last section of the belt, fixed pickup, jaws and both bins share
        # this locked view through the first three cycles and fourth pickup.
        name = "sorting-transfer"
        aim = (pick_x + .015, pick_y - .01, z + .115)
        eye = (aim[0] + 1.30, aim[1] - .58, aim[2] + 1.02)
        focal = 1340.
    else:
        # Cut during the final elevated carry. A steep view limits perspective
        # overlap between the two slots 90 mm apart in X. The azimuth offset
        # separates them along screen X as well, and keeps the labels legible.
        name = "sorting-bin-overview"
        aim = (bin_x + .05, bin_y + .025, z + .034)
        eye = (aim[0] + .47, aim[1] - .28, aim[2] + 1.57)
        focal = 2380.

    f = focal * height / 720.
    return Camera(name, frame.key, look_at(eye, aim, (0., 0., 1.)), width, height,
                  f, f, width / 2, height / 2, .025, 24.)


def make_sorting_board(output, panels):
    """Assemble three chronological, full-raster RGB PNGs at equal panel width."""
    make_stage_board(output, panels)
