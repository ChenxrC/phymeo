"""Film a Galaxea R1 transferring a machined part between two workshop fixtures."""
import argparse
from dataclasses import asdict
from pathlib import Path

from .film import FPS, capture_sequence, write_json


def workshop_lighting():
    from phymeo_rendering.environment import RadianceEnvironment
    from phymeo_rendering.studio import StudioLighting
    hdr = Path(__file__).resolve().parents[1]/"materials"/"university_workshop_1k.hdr"
    environment = RadianceEnvironment(hdr.as_uri(),
        "a125ada38fc602025326f29b68df09531bb16f899f1973b03d9ac5ac98142548",
        strength=.047, yaw_degrees=25., face_size=256)
    return StudioLighting(center_m=(.42,.22,.87), scale_m=1.32, panel_size_scale=.58,
                          panel_strength=.205, environment_map=environment)


def run(args):
    from preview_stable_lighting.runtime import activate
    activate(args.package)
    from phymeo_rendering.scene import FrameKey, NodePose, Pose, SceneFrame
    from .model import load_model
    from .finishes import apply_finish
    from .finish_details import apply_details
    from .polished_normals import PAINTED_SHELLS, smooth_painted_shells
    from .workshop_camera import camera_for_workshop, make_stage_board, workshop_action_box
    from .workshop_motion import LAYOUT, build_motion
    from .workshop_set import PAYLOAD_NODE, STATIC_NODE, append_workshop
    from .workshop_surfaces import apply_workshop_surfaces
    from .workshop_markings import apply_markings

    if args.frames < 24 or (args.still is not None and not 0 <= args.still < args.frames):
        raise ValueError("allow at least 24 frames for the complete operation and select a valid still")
    args.output.mkdir(parents=True, exist_ok=True)
    trajectory = args.output/"trajectory.json"
    if trajectory.exists():
        raise FileExistsError(f"choose a fresh output directory: {args.output}")
    model = load_model(args.asset_dir, args.cache_dir or args.output/"model-cache")
    scene = apply_details(apply_finish(model.scene, "polished"), args.output/"finish-assets")
    scene = smooth_painted_shells(scene, args.output/"normals", targets=PAINTED_SHELLS)
    scene = append_workshop(scene, args.output/"workshop-meshes", LAYOUT)
    scene = apply_workshop_surfaces(scene, args.output/"surface-assets")
    scene = apply_markings(scene, args.output/"markings", LAYOUT)
    samples, evidence = build_motion(model, args.frames, FPS)
    frames, records = [], []
    for index, sample in enumerate(samples):
        key = FrameKey(scene.key, "r1-workshop-transfer", 0, index, index, "authored")
        nodes = (*model.pose_nodes(sample.joints, sample.base), NodePose(STATIC_NODE, Pose()),
                 NodePose(PAYLOAD_NODE, sample.payload))
        frame = SceneFrame(key, index/FPS, nodes)
        camera = camera_for_workshop(frame, LAYOUT, index, args.frames)
        frames.append((frame, camera))
        records.append(dict(index=index, time_s=frame.time_s, phase=sample.phase,
                            attached=sample.attached, joints=sample.joints, camera=asdict(camera),
                            nodes={n.node_id:asdict(n.world_pose) for n in nodes}))
    write_json(trajectory, dict(frames=records, fps=FPS, layout=asdict(LAYOUT), motion=evidence))
    marks = evidence["proof_frames"]
    selected = {0, args.frames-1, *(m["frame"] for m in marks.values())}
    studio = workshop_lighting()
    exposure = 8.
    if args.verify_shadows:
        from .workshop_shadows import check_shadows
        frame, camera = frames[marks["in_transit"]["frame"]]
        shadow = check_shadows(args.package, scene, frame, camera, studio, exposure,
                               args.device_index, args.validation, LAYOUT)
        write_json(args.output/"workshop-shadows.json", shadow)
        if not shadow["passed"]:
            raise RuntimeError(f"stationary receiver shadow control failed: {shadow['checks']}")
    report = capture_sequence(args, scene, frames, trajectory, studio, exposure,
                              selected, stem="r1-workshop-transfer")
    report.update(motion=evidence, layout=asdict(LAYOUT), model_revision=model.diagnostics["revision"],
                  workpiece_node=PAYLOAD_NODE, instance_count=len(scene.instances))
    if args.still is None:
        panels = []
        for label, stage in (("Pick", "closed_at_pick"), ("Transfer", "in_transit"), ("Release", "open_at_place")):
            i = marks[stage]["frame"]
            panels.append((label, i, args.output/f"frame-{i:03}.png"))
        camera = frames[panels[1][1]][1]
        box = workshop_action_box(camera, LAYOUT)
        make_stage_board(args.output/"workshop-operation.png", panels, crop_box=box)
        report["board_source_indices"] = [p[1] for p in panels]
        report["board_source_rectangle"] = box
    write_json(args.output/"render.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--package", type=Path,
        default=Path(__file__).resolve().parents[3]/"artifacts/rendering/panda-stable-surface/default")
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--still", type=int)
    parser.add_argument("--samples", choices=("grid2","rotated4"), default="grid2")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--verify-shadows", action="store_true",
                        help="Verify the raised part casts a shadow on stationary surfaces before recording")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
