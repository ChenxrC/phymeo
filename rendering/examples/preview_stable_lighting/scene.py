"""Prepare the registered Panda recording and bind the derived shading assets."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from preview_compact_rigid_history import preview
from preview_compact_rigid_history.state import bind_presentations
from preview_lowres_filter._helpers import save
from .shell_normals import scene_with_smooth_shells


def prepare_scene(out, source_inputs, smooth=True, camera="sequence", studio_look="soft-studio"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shared = preview.load_delivery(Path(__file__).resolve().parents[1] / "preview_sparse_motion")
    args = SimpleNamespace(source_inputs=Path(source_inputs), camera=camera, studio_look=studio_look)
    scene, studio, inputs, bindings = preview.prepare_inputs(args, out, shared)
    normals = dict(changed_vertices=0, meshes=[])
    if smooth:
        scene, normals = scene_with_smooth_shells(scene, out / "shells")
    if normals["changed_vertices"]:
        assets = [(m["mesh_id"], m["derived_sha256"]) for m in normals["meshes"] if m["changed_vertices"]]
        signature = sha256(preview.canonical(assets)).hexdigest()
        scene = replace(scene, key=f"{scene.key}:shell-normals:{signature}")
        for i, (frame, request) in inputs.items():
            key = replace(frame.key, scene_key=scene.key)
            inputs[i] = (replace(frame, key=key), replace(request, camera=replace(request.camera, frame_key=key)))
        bindings = bind_presentations([inputs[i][0] for i in preview.POSES], bindings[0]["source_sha256"])
    save(out / "shading-assets.json", dict(scene_key=scene.key, **normals))
    return scene, studio, inputs, bindings, normals
