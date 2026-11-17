"""Complete MuJoCo scenes, preserved as one immutable native compilation.

This is a scene input, not an articulation lowering or a merge of robot assets.
It retains model options, actuators, world geometry and native index spaces.
The existing PhySolve runtime remains the only physical state owner.
"""

from hashlib import sha256
from pathlib import Path

import numpy as np

from ..backends.physolve_initial import INTEGRATION_FIELDS, NativeInitialState
from .mujoco_compiler import compile_mujoco
from .native_assembly import NativeAssetIdentity, NativeTemplate


def compile_native_scene(
    path, *, name, initial_state_path=None, capture_visuals=False, capture_frames=False
):
    """Compile one complete scene without namespacing or rewriting its native model.

    Public names are scoped independently. An optional NPZ owns exactly the
    integration inputs plus source_sha256 and model_sha256; allow_pickle is always false.
    The explicit FP32 quaternion contract is inherited from MuJoCoCudaSolver,
    while ordinary asset/native initialization retains strict_v1 by default.
    """
    import mujoco as mj

    if not isinstance(name, str) or not name or "::" in name:
        raise ValueError("complete scene requires an unscoped public name")
    path = Path(path).resolve(strict=True)
    compilation = compile_mujoco(path, require_asset=False, instance=name)
    model = compilation.fresh_native_model()
    data = mj.MjData(model)
    if initial_state_path is not None:
        with np.load(initial_state_path, allow_pickle=False) as source:
            if set(source.files) != {*INTEGRATION_FIELDS, "source_sha256", "model_sha256"}:
                raise ValueError(
                    "scene initial snapshot requires all integration fields and source/model SHA256"
                )
            if str(source["source_sha256"].item()) != sha256(path.read_bytes()).hexdigest():
                raise ValueError("initial snapshot source identity mismatch")
            if str(source["model_sha256"].item()) != compilation.source.model_sha256:
                raise ValueError("initial snapshot compiled model identity mismatch")
            initial = NativeInitialState(
                compilation.source.model_sha256,
                tuple((key, source[key]) for key in INTEGRATION_FIELDS),
                "mujoco_cuda_fp32_v1",
            )
            initial.apply(compilation, data)
    else:
        initial = NativeInitialState.from_data(
            compilation, data, quaternion_contract="mujoco_cuda_fp32_v1"
        )
        initial.apply(compilation, data)
    prefix = name + "::"

    def label(kind, index):
        value = mj.mj_id2name(model, getattr(mj.mjtObj, "mjOBJ_" + kind.upper()), index)
        return value or f"__{kind}_{index}"

    joints = tuple(
        (prefix + label("joint", i), i, int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i]), None)
        for i in range(model.njnt)
    )
    widths = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}
    if any(int(kind) == 1 for kind in model.jnt_type):
        raise NotImplementedError("complete scene ball-joint C ABI bindings are not qualified")
    body_names = tuple(label("body", i) for i in range(1, model.nbody))
    identity = NativeAssetIdentity(
        name,
        "mjcf",
        str(compilation.mj_spec.modelname),
        tuple((label, prefix + label, i) for i, label in enumerate(body_names, 1)),
    )
    # Immutable identity metadata. No native object or writable array escapes here.
    identities = tuple(
        (kind, tuple(prefix + label(kind, i) for i in range(count)))
        for kind, count in (
            ("body", model.nbody),
            ("geom", model.ngeom),
            ("joint", model.njnt),
            ("site", model.nsite),
            ("actuator", model.nu),
        )
    )
    if any(len(names) != len(set(names)) for _, names in identities):
        raise ValueError(
            "complete scene public identities collide with generated unnamed identities"
        )
    visual = None
    if capture_visuals:
        from .native_visuals import export_model

        visual = export_model(model, prefix=prefix, identities=dict(identities))
    frames = None
    if capture_frames:
        from .native_frames import NativeFrameMetadata, _mjcf_frames

        frames = NativeFrameMetadata(
            tuple(
                (dict(identities)["body"][i], tuple(map(float, model.body_ipos[i])))
                for i in range(1, model.nbody)
            ),
            tuple(_mjcf_frames(model, name, identities=dict(identities))),
        )
    return NativeTemplate(
        compilation,
        initial,
        joints,
        tuple((prefix + label("body", i), i) for i in range(1, model.nbody)),
        tuple((prefix + label("geom", i), i) for i in range(model.ngeom)),
        ((name, str(path), sha256(path.read_bytes()).hexdigest()),),
        joint_dimensions=tuple(widths[int(kind)] for kind in model.jnt_type),
        visual_scene=visual,
        asset_identities=(identity,),
        complete_scene_contract_version=1,
        actuator_bindings=tuple((prefix + label("actuator", i), i) for i in range(model.nu)),
        native_identity=identities,
        frame_metadata=frames,
        import_diagnostics=(
            "complete_scene_v1: native model/options/actuators retained; quaternion contract mujoco_cuda_fp32_v1",
        ),
    )
