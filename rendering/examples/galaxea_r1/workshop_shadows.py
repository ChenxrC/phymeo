"""Paired native lighting measurements on registered workshop receiving surfaces."""
from dataclasses import asdict, replace
from hashlib import sha256

import numpy as np


_LUMA = np.array([.2126, .7152, .0722])
_FLOOR = 1e-5       # In unexposed scene-linear radiance units.
_MIN_RELATIVE = .01


def _read(owner, frame, camera, binding, box, payload_ids, index):
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    from preview_compact_rigid_history.run_smoke import history_dtype
    shot = owner.capture(frame, CaptureRequest(camera, RenderPath.RASTER, tuple(Channel)), presentation=binding)
    if (shot.frame_key, shot.time_s, shot.camera) != (frame.key, frame.time_s, camera):
        raise ValueError("workshop shadow capture published a different frame or camera")
    planes = {p.channel:p.data for p in shot.planes}
    total = owner.read_reconstructed_linear()
    history = owner.read_history_incoming()
    h, w = camera.height, camera.width
    if ((total["width"],total["height"],total["format"]) != (w,h,"RGBA32F_LE") or
            "scene-linear" not in total["space"] or
            (history["width"],history["height"],history["format"]) != (w,h,"F32x4_U32x4_F32x4_LE") or
            total["capture_index"] != index or history["capture_index"] != index):
        raise ValueError("native shadow-probe readback format or frame counter differs")
    x0,y0,x1,y1 = box
    roi = np.s_[y0:y1,x0:x1]
    ids = np.frombuffer(planes[Channel.INSTANCE_ID], "<u4").reshape(h,w)
    depth = np.frombuffer(planes[Channel.FORWARD_DEPTH], "<f4").reshape(h,w)
    valid = np.frombuffer(planes[Channel.VALID],"u1").reshape(h,w)
    rgb = np.frombuffer(planes[Channel.RGB],"u1").reshape(h,w,3)
    hdr = np.frombuffer(total["data"],"<f4").reshape(h,w,4)[roi][...,:3]
    cells = np.frombuffer(history["data"], history_dtype(np)).reshape(h,w)[roi]
    primary = cells["lighting"][...,:3]
    if not (np.isfinite(hdr).all() and np.isfinite(primary).all() and np.isfinite(cells["incoming"]).all()):
        raise ValueError("nonfinite native scene-linear shadow-probe signal")
    # The sealed compute shader stores the current unoccluded diffuse factor
    # in base.a as an exact 24-bit integer in FP32. Its incoming RGB already
    # includes local environment visibility; it is separate from panel lighting.
    packed = cells["lighting"][...,3].astype(np.uint32)
    rho = np.stack([(packed >> bits) & 255 for bits in (0,8,16)], axis=2) / 255.
    indirect = rho * cells["incoming"][...,:3]
    metadata = dict(key=asdict(frame.key), time_s=frame.time_s, presentation=binding,
                    payload_pixels=int(np.isin(ids,payload_ids).sum()), hdr_sha256=sha256(total["data"]).hexdigest(),
                    planes={key.value:sha256(data).hexdigest() for key,data in planes.items()})
    return dict(ids=ids[roi].copy(), depth=depth[roi].copy(), valid=valid[roi].copy(),
                rgb=rgb[roi].copy(), total=hdr @ _LUMA, primary=primary @ _LUMA, environment=indirect @ _LUMA,
                metadata=metadata)


def _interior(rows, receivers):
    ref = rows[0]
    ids, z = ref["ids"], ref["depth"]
    good = np.isin(ids,receivers) & np.isfinite(z) & (ref["valid"] == 1)
    for row in rows[1:]:
        good &= (row["ids"] == ids) & (row["depth"] == z) & (row["valid"] == 1)
    result = np.zeros_like(good)
    inside = good[1:-1,1:-1].copy()
    h,w = good.shape
    # All nine samples must survive the subtraction in *all three* captures
    # and hit this very instance, with no depth step larger than 1 cm.
    relative_z = np.where(good, z, 0.)
    for dy in range(3):
        for dx in range(3):
            area = np.s_[dy:h-2+dy,dx:w-2+dx]
            inside &= good[area] & (ids[area] == ids[1:-1,1:-1])
            inside &= np.abs(relative_z[area]-relative_z[1:-1,1:-1]) < .01
    result[1:-1,1:-1] = inside
    return result


def _metric(rows, channel, mask):
    on, off, back = (r[channel] for r in rows)
    delta = np.minimum(off-on, off-back)  # Both in-place readings must be lower.
    noise = np.abs(back-on)
    floor = max(_FLOOR, 3*float(np.max(noise[mask], initial=0.)))
    relative = delta / np.maximum(off, _FLOOR)
    affected = mask & (delta > floor) & (relative >= _MIN_RELATIVE)
    metrics = dict(absolute_threshold=floor, fractional_threshold=_MIN_RELATIVE,
                   roundtrip_max_abs=float(np.max(noise[mask], initial=0.)),
                   darkened_pixels=int(affected.sum()),
                   brightened_pixels=int(np.sum(mask & (np.maximum(off-on,off-back) < -floor))),
                   max_absolute_attenuation=float(np.max(delta[mask], initial=0.)),
                   max_relative_attenuation=float(np.max(relative[affected], initial=0.)),
                   mean_absolute_attenuation_on_darkened=float(delta[affected].mean()) if affected.any() else 0.,
                   median_relative_attenuation_on_darkened=float(np.median(relative[affected])) if affected.any() else 0.)
    return metrics, affected, delta, relative


def check_shadows(package, scene, frame, camera, studio, exposure, device_index, validation, layout):
    """Return a JSON-ready three-capture control tied to the raised film exposure."""
    from _panda_scene import pose_matrix
    from preview_compact_rigid_history.state import bind_presentations, frame_digest
    from preview_stable_lighting.runtime import StableBackend
    from .workshop_camera import workshop_action_box
    from .workshop_set import PAYLOAD_NODE, STATIC_NODE

    if camera.frame_key != frame.key or frame.key.scene_key != scene.key:
        raise ValueError("shadow control requires the original bound film frame and camera")
    payload, = [n for n in frame.nodes if n.node_id == PAYLOAD_NODE]
    bottom_z = payload.world_pose.position_m[2] - layout.payload_size[2]/2
    if bottom_z < layout.support_z + .03:
        raise ValueError("choose a film pose with the workpiece visibly raised above its fixture")
    payload_ids = [item.id for item in scene.instances if item.node_id == PAYLOAD_NODE]
    # Exclude the native mirror-reflection eligibility range. On all remaining
    # static receivers, fixed normals/view/material keep the distant specular
    # environment constant. Changes in the primary guide are emitter visibility.
    material = {m.id:m for m in scene.materials}
    receivers = {i.id:i.material_id for i in scene.instances if i.node_id == STATIC_NODE
                 and material[i.material_id].metallic < .99}
    if not receivers or not payload_ids:
        raise ValueError("cannot identify registered stationary receivers and workpiece")
    box = workshop_action_box(camera, layout)
    C = pose_matrix(camera.world_pose)
    parked = C[:3,3] + C[:3,2]*(2*camera.far_m)  # Behind the eye, away from the cell and finite emitters.
    off = replace(payload, world_pose=replace(payload.world_pose, position_m=tuple(parked.tolist())))
    frames, cameras = [], []
    for i in range(3):
        key = replace(frame.key, world_id=frame.key.world_id+"-shadow-control",
                      tick=frame.key.tick+i, revision=frame.key.revision+i)
        nodes = tuple(off if i == 1 and n.node_id == PAYLOAD_NODE else n for n in frame.nodes)
        frames.append(replace(frame,key=key,time_s=frame.time_s+i/30,nodes=nodes))
        cameras.append(replace(camera,frame_key=key))
    source = dict(original_frame_sha256=frame_digest(frame), original_camera_sha256=frame_digest(camera),
                  original_key=asdict(frame.key), scene_sha256=frame_digest(scene), studio_sha256=frame_digest(studio),
                  parked_position_m=parked.tolist(), exposure=exposure)
    bindings = bind_presentations(frames, frame_digest(source))
    owner = StableBackend(package, studio=studio, exposure=exposure, device_index=device_index,
                          validation=validation, display_transform="pbr-neutral").open_scene(scene)
    try:
        rows = [_read(owner,f,c,b,box,payload_ids,i) for i,(f,c,b) in enumerate(zip(frames,cameras,bindings))]
    finally:
        owner.close()
    diag = owner.diagnostics()["device"]

    mask = _interior(rows, list(receivers))
    # Unproject centre depth to keep only the horizontal work area and the
    # fixture surfaces below the raised part, excluding distant walls/racks.
    yy,xx = np.indices(mask.shape)
    d = np.where(mask, rows[0]["depth"], 0.)
    rays = np.stack(((xx+box[0]+.5-camera.cx_px)/camera.fx_px,
                     -(yy+box[1]+.5-camera.cy_px)/camera.fy_px, -np.ones_like(d)), axis=2)
    world = (rays*d[...,None]) @ C[:3,:3].T + C[:3,3]
    xy = np.asarray((layout.pick_xy,layout.place_xy))
    mask &= np.all((world[...,:2] >= xy.min(axis=0)-.32) & (world[...,:2] <= xy.max(axis=0)+.32), axis=2)
    mask &= (world[...,2] >= layout.support_z-.075) & (world[...,2] <= layout.support_z+.002)
    metrics = {}
    for channel in ("primary","environment","total"):
        metrics[channel], affected, delta, fraction = _metric(rows,channel,mask)
        if channel == "primary":
            primary = affected
        if channel == "total":
            common = affected & primary
    counts = {identity:int(np.sum(common & (rows[0]["ids"] == identity))) for identity in receivers}
    by_receiver = [dict(instance_id=k,material=receivers[k],pixels=v) for k,v in counts.items() if v]
    witness = None
    code_drop = np.array([], dtype=float)
    if common.any():
        displayed = [r["rgb"][common] @ _LUMA for r in rows]
        code_drop = np.minimum(displayed[1]-displayed[0],displayed[1]-displayed[2])
        y,x = np.unravel_index(np.argmax(np.where(common,delta,-np.inf)),delta.shape)
        witness = dict(pixel_xy=[int(x+box[0]),int(y+box[1])], receiver_id=int(rows[0]["ids"][y,x]),
                       receiver_world_m=world[y,x].tolist(), total_attenuation=float(delta[y,x]),
                       total_relative=float(fraction[y,x]),
                       displayed_rgb=[r["rgb"][y,x].tolist() for r in rows],
                       luminance={name:[float(r[name][y,x]) for r in rows] for name in metrics})
    md = [r["metadata"] for r in rows]
    checks = dict(workpiece_removed_from_view=md[0]["payload_pixels"] > 0 and md[1]["payload_pixels"] == 0
                  and md[2]["payload_pixels"] == md[0]["payload_pixels"],
                  normal_geometry_and_rgb_restored=md[0]["planes"] == md[2]["planes"],
                  normal_linear_restored=md[0]["hdr_sha256"] == md[2]["hdr_sha256"],
                  stationary_receivers_visible=int(mask.sum()) >= 16,
                  primary_and_total_darkening=int(common.sum()) >= 16,
                  native_closed_and_clean=diag.get("closed") is True and all(diag.get(k,1) == 0 for k in
                      ("live_contexts","live_frames","live_resources","validation_errors")))
    return dict(passed=bool(all(checks.values())), checks=checks, source=source,
                captures=md, capture_order=["film pose","workpiece parked","film pose restored"],
                work_area_rectangle_px=box, eligible_receiver_pixels=int(mask.sum()),
                primary_and_total_darkened_pixels=int(common.sum()), by_receiver=by_receiver, strongest=witness,
                metrics=metrics, displayed_on_common=dict(pixels_darker_by_one_code=int(np.sum(code_drop >= 1.)),
                    median_luma_code_drop=float(np.median(code_drop)) if common.any() else 0.,
                    maximum_luma_code_drop=float(np.max(code_drop,initial=0.))),
                linear_signal="scene-linear Rec.709 luminance before display or exposure",
                geometry_gate="same static ID and exact centre depth in all three captures; eroded 3x3; workbench height band",
                lighting_terms=dict(primary="native primary direct lighting plus fixed distant specular environment",
                                    environment="packed diffuse reflectance times visibility-modulated environment incoming",
                                    total="native reconstructed HDR including current-frame contour resolve"))
