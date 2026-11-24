"""Render a color-sorted, four-workpiece production run with a Galaxea R1."""
import argparse
from dataclasses import asdict
from pathlib import Path

from .film import FPS, capture_sequence, write_json
from .sorting_layout import ITEM_KINDS, ITEM_NODES, LAYOUT


def cell_frame(scene, model, sample, index):
    from phymeo_rendering.scene import FrameKey, NodePose, SceneFrame
    from .sorting_set import conveyor_nodes
    key = FrameKey(scene.key, "r1-indexed-conveyor", 0, index, index, "authored")
    nodes = (*model.pose_nodes(sample.joints, sample.base), *conveyor_nodes(LAYOUT, sample.belt_travel_m),
             *(NodePose(node, pose) for node, pose in zip(ITEM_NODES, sample.payloads, strict=True)))
    return SceneFrame(key, index/FPS, nodes)


def verify_final_bins(owner, scene, frame, camera, kinds):
    """Check the released poses against the cell layout and the visible ID plane."""
    import numpy as np
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    owner.reset_history()
    capture = owner.capture(frame, CaptureRequest(camera, RenderPath.RASTER, (Channel.INSTANCE_ID,)))
    if (capture.frame_key, capture.camera, capture.time_s) != (frame.key, camera, frame.time_s):
        raise ValueError("final bin exposure mismatch")
    plane, = capture.planes
    ids = np.frombuffer(plane.data,dtype="<u4").reshape(camera.height,camera.width)
    poses = {node.node_id:node.world_pose for node in frame.nodes}
    occupied = set()
    contents = []
    for node, kind in zip(ITEM_NODES, kinds, strict=True):
        x,y,z = poses[node].position_m
        cx,cy = LAYOUT.bin_centers[0 if kind == "block" else 1]
        offset = x-cx
        expected_z = LAYOUT.support_z+LAYOUT.payload_size[2]/2
        if (abs(abs(offset)-LAYOUT.slot_offset_x) > 1e-4 or abs(y-cy) > 1e-4 or abs(z-expected_z) > 1e-4):
            raise ValueError(f"{node}: pose outside its receiving slot")
        slot = (kind, offset>0)
        visible = np.isin(ids,[i.id for i in scene.instances if i.node_id == node])
        pixels = int(visible.sum())
        if slot in occupied or pixels < 64:
            raise ValueError(f"{node}: duplicate slot or hidden workpiece")
        occupied.add(slot)
        contents.append(dict(node=node, kind=kind, position_m=(x,y,z), visible_pixels=pixels))
    return contents


def run(args):
    from preview_stable_lighting.runtime import activate
    activate(args.package)
    from preview_compact_rigid_history.state import frame_digest
    from preview_stable_lighting.runtime import StableBackend
    from .model import load_model
    from .finishes import apply_finish
    from .finish_details import apply_details
    from .polished_normals import PAINTED_SHELLS, smooth_painted_shells
    from .sorting_camera import camera_for_sorting, make_sorting_board
    from .sorting_inset import make_reader_board, make_reader_inset
    from .sorting_labels import apply_sorting_labels
    from .sorting_motion import build_sort_motion
    from .sorting_reader import read_part
    from .sorting_set import append_sorting_set
    from .workshop import workshop_lighting
    from .workshop_surfaces import apply_workshop_surfaces

    kinds = tuple(args.order.split(","))
    if sorted(kinds) != sorted(ITEM_KINDS):
        raise ValueError("use two blocks and two hex collars in any order")
    if args.frames < 64:
        raise ValueError("allow at least 64 frames for the four transfers")
    if args.still is not None and not 0 <= args.still < args.frames:
        raise ValueError("still frame outside recording")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"choose an empty sorting output directory: {args.output}")
    model = load_model(args.asset_dir, args.cache_dir or args.output/"model-cache")
    scene = apply_details(apply_finish(model.scene, "polished"), args.output/"finish-assets")
    scene = smooth_painted_shells(scene, args.output/"normals", targets=PAINTED_SHELLS)
    scene = append_sorting_set(scene, args.output/"sorting-meshes", LAYOUT, kinds)
    scene = apply_workshop_surfaces(scene, args.output/"surface-assets")
    scene = apply_sorting_labels(scene, args.output/"markings", LAYOUT)
    studio = workshop_lighting()
    exposure = 8.
    observations = []
    hold = max(1, round(args.frames/18))
    reader = StableBackend(args.package, studio=studio, device_index=args.device_index,
                           validation=args.validation, exposure=exposure, display_transform="pbr-neutral").open_scene(scene)
    try:
        def inspect(sample, index):
            frame = cell_frame(scene, model, sample, index)
            reading = read_part(reader, frame, LAYOUT, args.output)
            observations.append(reading)
            return reading["decision"]

        samples, motion = build_sort_motion(model, args.frames-hold, FPS, kinds, read_station=inspect)
        final_index = args.frames-1
        final_frame = cell_frame(scene, model, samples[-1], final_index)
        final_bins = verify_final_bins(reader, scene, final_frame,
            camera_for_sorting(final_frame, LAYOUT, final_index, args.frames), kinds)
    finally:
        reader.close()
    reader_device = reader.diagnostics()["device"]
    if reader_device["live_resources"] or reader_device["validation_errors"]:
        raise RuntimeError("station camera did not close cleanly")
    samples = [*samples, *([samples[-1]]*hold)]
    trajectory = args.output/"trajectory.json"
    frames, records = [], []
    for i, sample in enumerate(samples):
        frame = cell_frame(scene, model, sample, i)
        camera = camera_for_sorting(frame, LAYOUT, i, len(samples))
        frames.append((frame,camera))
        records.append(dict(frame=i, time_s=frame.time_s, joints=sample.joints, phase=sample.phase,
                            travel_m=sample.belt_travel_m, attached=sample.attached_index,
                            placed=sample.placed_count, camera=asdict(camera),
                            poses={n.node_id:asdict(n.world_pose) for n in frame.nodes}))
    for data, route in zip(observations, motion["routes"], strict=True):
        i = data["frame"]
        if (i != route["read_frame"] or data["decision"] != route["bin"]
                or frame_digest(frames[i][0]) != data["source_pose_sha256"]):
            raise ValueError("station observation, routing and filmed exposure differ")
    if (len(observations) != len(kinds) or not motion["classification_matches"]
            or motion["placed_count"] != len(kinds) or any(len(v) != 2 for v in motion["final_slots"].values())):
        raise ValueError("sorting run did not deliver all four parts to the correct slots")
    write_json(trajectory, dict(layout=asdict(LAYOUT), orders=kinds, fps=FPS, frames=records))
    marks = [c["proof_frames"] for c in motion["cycles"]]
    selected = {0, args.frames-1, *(c[k] for c in marks for k in ("stopped","pick","transfer","release"))}
    inset = make_reader_inset(observations, args.output, FPS, (1280,720))
    make_reader_board(observations, args.output)
    report = capture_sequence(args, scene, frames, trajectory, studio, exposure, selected,
                              stem="r1-conveyor-sorting", annotate=inset)
    report.update(model_revision=model.diagnostics["revision"], motion=motion, station_reads=observations,
                  reader_device=reader_device["device_name"], reader_validation_errors=reader_device["validation_errors"],
                  end_hold_frames=hold, instance_count=len(scene.instances), final_visible_contents=final_bins)
    if args.still is None:
        panels = [(label,i,args.output/f"frame-{i:03}.png") for label,i in
                  (("Infeed",0), ("Blue collar to B",marks[kinds.index("hex")]["transfer"]),
                   ("Sorted bins",args.frames-1))]
        make_sorting_board(args.output/"sorting-sequence.png",panels)
    write_json(args.output/"render.json",report)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--asset-dir",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--cache-dir",type=Path)
    cli.add_argument("--package",type=Path,
                     default=Path(__file__).resolve().parents[3]/"artifacts/rendering/panda-stable-surface/default")
    cli.add_argument("--device-index",type=int,default=1)
    cli.add_argument("--frames",type=int,default=540)
    cli.add_argument("--order",default=",".join(ITEM_KINDS))
    cli.add_argument("--still",type=int)
    cli.add_argument("--samples",choices=("grid2","rotated4"),default="grid2")
    cli.add_argument("--validation",action="store_true")
    run(cli.parse_args())


if __name__ == "__main__":
    main()
