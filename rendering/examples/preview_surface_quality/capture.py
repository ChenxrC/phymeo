"""Time normal 720p RGB capture; verify sensor publication in a separate replay."""
import argparse
from dataclasses import replace
import subprocess
from pathlib import Path
from time import perf_counter

from PIL import Image
from preview_compact_rigid_history import preview
from preview_compact_rigid_history.runtime import activate
from preview_lowres_filter._helpers import digest, save


def capture(args):
    from .build import check_package
    from preview_lowres_filter.runtime import check_seal
    from .runtime import SurfaceQualityBackend
    from preview_compact_rigid_history.runtime import RigidHistoryBackend

    stable = (args.package / "stable-seal.json").is_file() and not args.baseline
    if stable:
        from preview_stable_lighting.runtime import StableBackend, activate as activate_stable
        root, manifest = activate_stable(args.package)
        backend = StableBackend
    elif args.baseline:
        root, manifest = check_seal(args.package)
        backend = RigidHistoryBackend
    else:
        root, manifest = check_package(args.package)
        backend = SurfaceQualityBackend
    if not stable:
        activate(root)
    from phymeo_rendering.outputs import Channel
    delivery = Path(__file__).resolve().parents[1] / "preview_sparse_motion"
    shared = preview.load_delivery(delivery)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if stable:
        from preview_stable_lighting.scene import prepare_scene
        scene, studio, inputs, bindings, _ = prepare_scene(
            out, args.source_inputs, camera=args.camera, studio_look=args.studio_look)
    else:
        scene, studio, inputs, bindings = preview.prepare_inputs(args, out, shared, manifest["contract"])
    record = dict(camera=args.camera, lighting=args.studio_look, resolution=[1280, 720], seed=0,
                  source_inputs_sha256=digest(args.source_inputs),
                  lighting_profile="stable-environment" if stable else "sparse-bounce",
                  source_manifest_sha256=digest(root / "source-manifest.json"), rounds=[])
    saved = {}

    def open_session(validation):
        history = {} if stable else {"reuse_history": True}
        return backend(root, device_index=args.device_index, studio=studio, seed=0, exposure=8.,
                       validation=validation, **history).open_scene(scene)

    def expose(session, i, channels):
        frame, request = inputs[i]
        request = replace(request, channels=channels)
        start = perf_counter()
        shot = session.capture(frame, request, presentation=bindings[i])
        elapsed = 1000 * (perf_counter() - start)
        shared.base.verify_capture(shot, frame, request)
        return shot, elapsed

    def close_session(session):
        session.close()
        shared.base.verify_close(session.diagnostics()["device"])

    for round_index in range(args.rounds):
        session = open_session(False)
        times, movie = [], []
        try:
            for i in preview.WARMUP:
                expose(session, i, (Channel.RGB,))
            session.reset_history()
            for i in preview.POSES:
                shot, elapsed = expose(session, i, (Channel.RGB,))
                times.append(elapsed)
                rgb = shot.planes[0].data
                if round_index == 0 and i in (59, 60, 61):
                    saved[i] = rgb
                    Image.frombytes("RGB", (1280, 720), rgb).save(out / f"frame-{i:03}.png")
                if round_index == 0 and args.movie:
                    movie.append(rgb)
            result = shared.statistics(times, full=True)
            result.update(frame_ms=times, gpu=session.diagnostics()["device"]["resident_profile"])
            record["rounds"].append(result)
        finally:
            close_session(session)
        if movie:
            command = ["ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "-video_size", "1280x720", "-framerate", "30",
                       "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
                       "-movflags", "+faststart", str(out / "motion.mp4")]
            with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
                for frame in movie:
                    encoder.stdin.write(frame)
                encoder.stdin.close()
                if encoder.wait():
                    raise RuntimeError("motion video encoding failed")
            movie.clear()
        save(out / "timings.json", record)

    # Reproduce the same presentation sequence in a validation-enabled owner.
    # The preceding RGB-only timed run remains the performance result.
    session = open_session(True)
    try:
        for i in preview.WARMUP:
            expose(session, i, (Channel.RGB,))
        session.reset_history()
        for i in range(62):
            shot, _ = expose(session, i, tuple(Channel) if i >= 59 else (Channel.RGB,))
            if i >= 59:
                planes = {p.channel: p.data for p in shot.planes}
                if planes[Channel.RGB] != saved[i]:
                    raise RuntimeError(f"sensor request changed the returned RGB at frame {i}")
                if i == 60:
                    for channel, name in ((Channel.INSTANCE_ID, "id"), (Channel.FORWARD_DEPTH, "depth"),
                                          (Channel.VALID, "valid")):
                        (out / f"frame-060-{name}.bin").write_bytes(planes[channel])
                    if stable or "contours" in manifest["contract"].get("surface_quality", {}):
                        from preview_contour_aa.calibrate import counts
                        record["contour_counts_frame60"] = counts(session)
    finally:
        close_session(session)
    record.update(validation_errors=0, rgb_sensor_parity_frames=[59, 60, 61])
    save(out / "timings.json", record)


def main():
    from preview_stable_lighting.build import DEFAULT_PACKAGE
    from preview_lowres_filter._helpers import read
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", type=Path, default=DEFAULT_PACKAGE,
                     help="Defaults to the stable motion preset. Pass a sealed sparse package to replay the older lighting.")
    cli.add_argument("--source-inputs", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--device-index", type=int, default=1)
    cli.add_argument("--baseline", action="store_true")
    cli.add_argument("--movie", action="store_true")
    cli.add_argument("--rounds", type=int, choices=(1, 2), default=2)
    cli.add_argument("--camera", choices=("detail", "sequence"))
    cli.add_argument("--studio-look", choices=("recorded", "soft-studio"))
    args = cli.parse_args()
    source = read(args.source_inputs)
    args.camera = args.camera or source.get("camera", "detail")
    args.studio_look = args.studio_look or source.get("studio_look", "recorded")
    if args.package == DEFAULT_PACKAGE and not (args.package / "stable-seal.json").is_file():
        cli.error("Build the default package with python -m preview_stable_lighting.build --parent PATH_TO_SURFACE_16_AA")
    capture(args)


if __name__ == "__main__":
    main()
