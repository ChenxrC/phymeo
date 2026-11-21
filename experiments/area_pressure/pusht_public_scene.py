"""Replay at most 69 original withdrawal commands through the public CUDA Scene.

python -m experiments.area_pressure.pusht_public_scene --trajectory ORIGINAL_DIR
    --output NEW.json [--steps 69] [--cache-root CACHE_DIR]

The initial archived last_push_entry is normalized once for public initialization.
Afterward only the original controls are replayed: Scene.step alone advances q/v.
The declared PL prism and finite T sides are the public experiment's material,
not the capsule-distance midpoint material of the older entry experiment.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from itertools import product
from pathlib import Path
from time import perf_counter

import mujoco as mj
import numpy as np

from phymeo.backends.area_friction_law import AreaFrictionPairLaw
from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.geometry.area_pressure import AreaPressureIdentity
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh, TetraPressureMesh
from phymeo.geometry.box_union_surface import axis_aligned_box_union_mesh
from phymeo.training.manipulation_tasks import manipulation_scenes

from .pusht_entry_rows import DEFAULT_INPUTS, _digest, iter_controls, load_entries

H = 0.001
MU, SLIP = 0.7, 0.01
START_TICK, LAST_TICK = 1699, 1767
MAX_STEPS = LAST_TICK - START_TICK + 1
CHANNELS = ("ctrl", "qfrc_applied", "xfrc_applied")
IDENTITY = AreaPressureIdentity(
    "tool",
    "object",
    "test-axially-extended-octagonal-PL-pressure",
    "original-T-finite-negative-y-side-faces",
)
ALL_LATERAL_IDENTITY = AreaPressureIdentity(
    IDENTITY.compliant_body,
    IDENTITY.rigid_body,
    IDENTITY.compliant_material,
    "original-T-finite-all-lateral-faces",
)


def side_identity(side_coverage="negative-y"):
    """Resolve the complete material identity; coverage is an explicit choice."""
    if type(side_coverage) is not str or side_coverage not in ("negative-y", "all-lateral"):
        raise ValueError("side_coverage must be 'negative-y' or 'all-lateral'")
    return IDENTITY if side_coverage == "negative-y" else ALL_LATERAL_IDENTITY


LATERAL_COORDINATE_POLICY = "shortest-decimal-center-halfsize-single-round-v1"


def side_geometry_provenance(model, *, side_coverage="all-lateral"):
    """Record the material coordinate convention without claiming XML recovery.

    The all-lateral material interprets each compiled center and half-size via
    its shortest round-trip decimal string, performs exact rational +/- and
    rounds each resulting material plane once to binary64. This explicit new
    material convention joins equal decimal planes; it does not weld arbitrary
    nearby bounds. Raw bounds, declared bounds and every displacement are saved.
    The low-level typed-bounds union helper retains genuinely distinct planes.
    """
    identity = side_identity(side_coverage)
    result = dict(
        side_coverage=side_coverage,
        coordinate_frame="object-body-local",
        identity=dict(
            compliant_body=identity.compliant_body,
            rigid_body=identity.rigid_body,
            compliant_material=identity.compliant_material,
            rigid_material=identity.rigid_material,
        ),
        scope="finite lateral union of direct unrotated object-body boxes; no top/bottom",
    )
    if side_coverage == "all-lateral":
        _lateral_box_bounds(model, provenance=result)
    else:
        result.update(
            coordinate_policy="legacy-negative-y-binary64-midpoint-grid-v1",
            source_interpretation="unchanged-historical-material-and-asset-hash",
        )
    return result


def _lateral_box_bounds(model, *, provenance=None):
    """Original named tool/T sources plus any other direct object-body boxes.

    Returned bounds are object-body local. The body pose is unrestricted; geom
    frames must be unrotated in their own bodies (either quaternion sign).
    Child-body geometry and non-box object geoms are unsupported, not omitted.
    """
    tool, body = int(model.body("tool").id), int(model.body("object").id)
    pusher = int(model.geom("pusher").id)

    def unrotated(geom):
        q = np.asarray(model.geom_quat[geom])
        return np.array_equal(q, (1.0, 0.0, 0.0, 0.0)) or np.array_equal(q, (-1.0, 0.0, 0.0, 0.0))

    if (
        model.geom_bodyid[pusher] != tool
        or model.geom_type[pusher] != mj.mjtGeom.mjGEOM_CAPSULE
        or not unrotated(pusher)
        or not np.isfinite(model.geom_pos[pusher]).all()
        or not np.isfinite(model.geom_size[pusher, :2]).all()
        or np.any(model.geom_size[pusher, :2] <= 0)
    ):
        raise ValueError("all-lateral requires the body-local z-axis-aligned tool capsule")
    for name in ("object-0", "object-1"):
        if model.geom_bodyid[model.geom(name).id] != body:
            raise ValueError(
                "all-lateral requires both original boxes in the object body-local frame"
            )
    # A descendant has a separate frame and may move independently. Do not
    # silently drop its geometry from a declaration of the object's sides.
    for geom in range(model.ngeom):
        owner = int(model.geom_bodyid[geom])
        if owner == body:
            continue
        while owner > 0 and owner != body:
            owner = int(model.body_parentid[owner])
        if owner == body:
            raise ValueError("all-lateral does not support child-body box geometry")
    boxes, records = [], []
    for geom in np.flatnonzero(model.geom_bodyid == body):
        if (
            model.geom_type[geom] != mj.mjtGeom.mjGEOM_BOX
            or not unrotated(geom)
            or not np.isfinite(model.geom_pos[geom]).all()
            or not np.isfinite(model.geom_size[geom]).all()
            or np.any(model.geom_size[geom] <= 0)
        ):
            raise ValueError("all-lateral requires finite axis-aligned body-local object boxes")
        center = [str(float(value)) for value in model.geom_pos[geom]]
        half_size = [str(float(value)) for value in model.geom_size[geom]]
        raw = np.array(
            (
                model.geom_pos[geom] - model.geom_size[geom],
                model.geom_pos[geom] + model.geom_size[geom],
            )
        )
        # These are declared material coordinates, not recovered XML tokens.
        # Equal decimal expressions share one plane without a gap tolerance.
        material = np.array(
            [
                [
                    float(Fraction(c) + sign * Fraction(s))
                    for c, s in zip(center, half_size, strict=True)
                ]
                for sign in (-1, 1)
            ]
        )
        boxes.append(material)
        records.append(
            dict(
                geometry_id=int(geom),
                geometry_name=model.geom(int(geom)).name,
                center_decimal_m=center,
                half_size_decimal_m=half_size,
                raw_bounds_body_local_m=raw.tolist(),
                material_bounds_body_local_m=material.tolist(),
                material_minus_raw_m=(material - raw).tolist(),
            )
        )
    if provenance is not None:
        provenance.update(
            coordinate_policy=LATERAL_COORDINATE_POLICY,
            source_interpretation="compiled-float-shortest-roundtrip-decimal-not-original-XML",
            boxes=records,
        )
    return np.asarray(boxes)


def build_side_asset(model, *, side_coverage="negative-y"):
    """28 pressure tetrahedra and explicitly selected finite object sides.

    The unchanged octagonal PL prism has 18000 Pa at its center and zero boundary
    pressure; its caps lie outside the original T side band. ``negative-y``
    preserves the historical identity, vertex/face order and asset hash.
    ``all-lateral`` uses the union exterior normal to local +/-x and +/-y of
    unrotated object-body boxes, under the named single-round decimal material
    convention recorded by ``side_geometry_provenance``. Neither mode declares top/bottom contact; the
    whole body pair is replaced, so omitted faces have no native fallback.
    """
    identity = side_identity(side_coverage)
    lateral_boxes = _lateral_box_bounds(model) if side_coverage == "all-lateral" else None
    pusher = model.geom("pusher").id
    radius, half_axis = model.geom_size[pusher, :2]
    center = model.geom_pos[pusher]
    angles = np.pi / 8 + np.arange(8) * np.pi / 4
    xy = radius / np.cos(np.pi / 8) * np.column_stack((np.cos(angles), np.sin(angles)))
    half_z = 4 * (half_axis + radius)
    vertices = (
        np.vstack(
            (
                np.column_stack((xy, np.full(8, -half_z))),
                np.column_stack((xy, np.full(8, half_z))),
                np.zeros((1, 3)),
            )
        )
        + center
    )
    boundary = []
    for i in range(8):
        j = (i + 1) % 8
        boundary.extend(((i, j, j + 8), (i, j + 8, i + 8)))
    for i in range(1, 7):
        boundary.extend(((0, i + 1, i), (8, 8 + i, 8 + i + 1)))
    pressure = TetraPressureMesh(
        tuple(f"p{i}" for i in range(17)),
        vertices,
        tuple((*face, 16) for face in boundary),
        (0.0,) * 16 + (18000.0,),
        tuple(f"cell{i}" for i in range(28)),
    )
    if lateral_boxes is not None:
        nodes, faces = axis_aligned_box_union_mesh(
            lateral_boxes, face_directions=((0, -1), (0, 1), (1, -1), (1, 1))
        )
        rigid = RigidPressureMesh(
            tuple(f"r{i}" for i in range(len(nodes))),
            nodes,
            faces,
            tuple(f"T-all-lateral-exterior-{i}" for i in range(len(faces))),
        )
        return BodyAreaPressureAsset(identity, pressure, rigid)
    # Keep this historical path's coordinates, triangulation order and IDs
    # unchanged: archived native Scene checkpoints bind its exact asset hash.
    boxes = np.asarray(
        [
            (
                model.geom_pos[model.geom(name).id] - model.geom_size[model.geom(name).id],
                model.geom_pos[model.geom(name).id] + model.geom_size[model.geom(name).id],
            )
            for name in ("object-0", "object-1")
        ]
    )
    axes = [np.unique(boxes[:, :, axis]) for axis in range(3)]
    solid = {}
    for cell in product(*(range(len(axis) - 1) for axis in axes)):
        lo = np.array([axes[a][i] for a, i in enumerate(cell)])
        hi = np.array([axes[a][i + 1] for a, i in enumerate(cell)])
        middle = (lo + hi) / 2
        if np.any(np.all((boxes[:, 0] <= middle) & (middle <= boxes[:, 1]), axis=1)):
            solid[cell] = (lo, hi)
    nodes, indices, faces = [], {}, []
    for cell, (lo, hi) in solid.items():
        for axis, side in product(range(3), (-1, 1)):
            if axis != 1 or side != -1:
                continue
            neighbor = list(cell)
            neighbor[axis] += side
            if tuple(neighbor) in solid:
                continue
            tangents = [i for i in range(3) if i != axis]
            rectangle = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                point = lo.copy()
                point[axis] = hi[axis] if side > 0 else lo[axis]
                point[tangents] = [
                    hi[tangents[0]] if u else lo[tangents[0]],
                    hi[tangents[1]] if v else lo[tangents[1]],
                ]
                key = tuple(point)
                if key not in indices:
                    indices[key] = len(nodes)
                    nodes.append(point)
                rectangle.append(indices[key])
            a, b, c, d = rectangle
            if np.cross(nodes[b] - nodes[a], nodes[c] - nodes[a])[axis] * side < 0:
                b, d = d, b
            faces.extend(((a, b, c), (a, c, d)))
    rigid = RigidPressureMesh(
        tuple(f"r{i}" for i in range(len(nodes))),
        np.asarray(nodes),
        tuple(faces),
        tuple(f"T-exterior-{i}" for i in range(len(faces))),
    )
    return BodyAreaPressureAsset(IDENTITY, pressure, rigid)


def legal_initial_q(model, archived):
    """Normalize only the initial free quaternion, keeping the public guard."""
    original = np.asarray(archived)
    if (
        original.shape != (model.nq,)
        or original.dtype.kind not in "fiu"
        or not np.isfinite(original).all()
    ):
        raise ValueError("expected one finite original qpos vector")
    q = original.astype(np.float64, copy=True)
    address = int(model.jnt_qposadr[model.joint("object-free").id]) + 3
    length = np.linalg.norm(q[address : address + 4])
    if not np.isfinite(length) or length <= 0:
        raise ValueError("initial object quaternion must be finite and nonzero")
    q[address : address + 4] /= length
    if np.max(np.abs(q - original)) >= 1e-7:
        raise ValueError("initial normalization exceeds the established 1e-7 repair bound")
    return q


def build_scene_spec(mjcf, *, key="original-RR-public-mixed-area", side_coverage="negative-y"):
    """Compile the unchanged source and bind the explicitly selected material."""
    identity = side_identity(side_coverage)
    compilation = compile_mujoco_xml(mjcf, require_asset=False)
    if side_coverage == "all-lateral":
        _lateral_box_bounds(compilation.mj_model)
    return MuJoCoTyxModelSpec(
        key,
        compilation,
        contact_capacity=64,
        constraint_capacity=512,
        area_pressure_pairs=(("tool", "object"),),
        area_friction_laws=(AreaFrictionPairLaw(identity, MU, SLIP),),
    )


def control_from_record(record):
    return MuJoCoControl(
        record["ctrl"], qfrc_applied=record["qfrc_applied"], xfrc_applied=record["xfrc_applied"]
    )


def _payload_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _state(state, free_address):
    return dict(
        local_tick=state.tick,
        time=state.time,
        q=state.q.tolist(),
        v=state.v.tolist(),
        object_position_world_m=state.q[free_address : free_address + 3].tolist(),
        object_quaternion_wxyz=state.q[free_address + 3 : free_address + 7].tolist(),
    )


def _geometry_observation(asset, receipt, geometry, pair):
    """Keep zero force, empty clipped area and strict separation distinct."""
    dyn = receipt.dynamics
    ba, bb = pair["body_ids"]
    soft = asset.pressure_local.world_vertices @ dyn.body_rotation[ba].T + dyn.body_position[ba]
    rigid = asset.rigid_local.world_vertices @ dyn.body_rotation[bb].T + dyn.body_position[bb]
    soft = soft[asset.pressure_local.cells]
    rigid = rigid[asset.rigid_local.faces]
    lo_a, hi_a = soft.min(axis=1), soft.max(axis=1)
    lo_b, hi_b = rigid.min(axis=1), rigid.max(axis=1)
    gaps = np.maximum(lo_b[None, :, :] - hi_a[:, None, :], lo_a[:, None, :] - hi_b[None, :, :])
    separating_gaps = gaps.max(axis=2)
    # A strict AABB gap is a sufficient witness only. No gap does not establish
    # intersection. An empty positive-area clip may still be boundary touching.
    scale = max(1.0, float(np.max(np.abs(soft))), float(np.max(np.abs(rigid))))
    arithmetic_guard = 32 * np.finfo(np.float64).eps * scale
    strict = bool(np.all(separating_gaps > arithmetic_guard))
    return dict(
        phase="accepted-step-input",
        triangles=len(geometry.surface.triangles),
        polygons=geometry.polygon_count,
        candidate_pairs=geometry.candidate_pairs,
        shared_face_duplicates=geometry.shared_face_duplicates,
        clipped_area_m2=float(np.sum(pair["dA"])),
        no_positive_area_intersection=not bool(geometry.surface.triangles),
        strict_disjoint_witness="all-cell-face-AABBs" if strict else None,
        strict_disjoint_certified=strict,
        minimum_pair_separating_axis_gap_m=float(np.min(separating_gaps)),
        fp64_arithmetic_guard_m=float(arithmetic_guard),
    )


def _step_record(model, asset, before, after, geometry, command, free_address, wall_s):
    receipt = after.last_step
    if (
        receipt is None
        or after.tick != before.tick + 1
        or receipt.tick != after.tick
        or not np.array_equal(receipt.input_q, before.q)
        or not np.array_equal(receipt.input_v, before.v)
        or receipt.input_time != before.time
        or receipt.output_time != after.time
        or geometry.surface.stamp.step_index != before.tick
    ):
        raise RuntimeError("accepted receipt does not continue the preceding Scene state")
    channels = {key: np.asarray(command[key]).tolist() for key in CHANNELS}
    for key in CHANNELS:
        if not np.array_equal(getattr(receipt.stored_control, key), command[key]):
            raise RuntimeError("accepted " + key + " differs from the original held command")
    (pair,) = receipt.raw_contact_fields["area_friction"]
    solver = receipt.raw_contact_fields["area_friction_solver"]
    diagnostics = solver["raw_diagnostics"]
    meta = np.asarray(diagnostics["meta"])
    calls = int(meta[2])
    if meta.shape != (4,) or meta[0] != 0 or meta[3] != 1 or not 1 <= calls <= 25:
        raise RuntimeError("expected converged resident-native diagnostics")
    normals = np.asarray(pair["normals"])
    normal_force = np.asarray(pair["normal_force"])
    tangent = np.asarray(pair["tangential_force"])
    supports = []
    for index, contact in enumerate(receipt.contacts):
        names = tuple(model.geom(int(i)).name for i in contact.geometry_ids)
        if "support" not in names:
            continue
        other = 1 if names[0] == "support" else 0
        force = contact.force_on_geometry2_world if other == 1 else contact.force_on_geometry1_world
        supports.append(
            dict(
                contact_index=index,
                geometry_names=names,
                active=contact.active,
                normal_n=float(contact.normal_force),
                force_on_supported_world_n=force.tolist(),
            )
        )
    normal_n = float(normal_force.sum())
    control = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in command.items()
    }
    control["channels_sha256"] = _payload_hash(channels)
    record = dict(
        original_input_tick=command["tick"],
        original_output_tick=command["output_tick"],
        input_state=_state(before, free_address),
        output_state=_state(after, free_address),
        control=control,
        stored_control_matches_original=True,
        geometry=_geometry_observation(asset, receipt, geometry, pair),
        area=dict(
            samples=len(normal_force),
            active_samples=int(np.count_nonzero(normal_force > 0)),
            normal_load_n=normal_n,
            zero_normal_load=bool(np.all(normal_force == 0)),
            pusher_body="tool",
            normal_force_on_pusher_world_n=(normal_force[:, None] * normals).sum(axis=0).tolist(),
            tangential_force_on_pusher_world_n=tangent.sum(axis=0).tolist(),
            tangential_net_norm_n=float(np.linalg.norm(tangent.sum(axis=0))),
            tangential_point_norm_sum_n=float(np.linalg.norm(tangent, axis=1).sum()),
            total_force_on_pusher_world_n=np.asarray(pair["force_on_a"]).sum(axis=0).tolist(),
        ),
        native=dict(
            detected_contacts=len(receipt.contacts),
            active_contacts=sum(c.active for c in receipt.contacts),
            rows=len(receipt.constraints),
            limit_rows=sum(r.constraint_type == 3 for r in receipt.constraints),
            support_normal_n=sum(s["normal_n"] for s in supports),
            object_support_normal_n=sum(
                s["normal_n"]
                for s in supports
                if any(n in ("object-0", "object-1") for n in s["geometry_names"])
            ),
            supports=supports,
        ),
        solver=dict(
            schema=solver["schema"],
            outer_iterations=int(meta[1]),
            native_solve_calls=calls,
            normalized_residual=float(solver["normalized_residual"]),
            output_velocity=np.asarray(solver["output_velocity"]).tolist(),
            meta=meta.tolist(),
            trace_meta=np.asarray(diagnostics["trace_meta"])[:calls].tolist(),
            trace=np.asarray(diagnostics["trace"])[:calls].tolist(),
        ),
        step_wall_s=wall_s,
    )
    return record


def _write_progress(stream, report):
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    stream.seek(0)
    stream.write(text)
    stream.truncate()
    stream.flush()


def _error_chain(error):
    result, seen = [], set()
    while error is not None and id(error) not in seen and len(result) < 8:
        seen.add(id(error))
        result.append(dict(type=type(error).__name__, message=str(error)[:4000]))
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return result


def run_window(trajectory, output, *, steps=MAX_STEPS, cache_root=None):
    """Write a new bounded JSON artifact after each accepted public Scene step."""
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be an integer in 1..{MAX_STEPS}")
    trajectory, output = Path(trajectory).resolve(), Path(output).resolve()
    report = dict(
        schema="phymeo-pusht-public-scene-window-v1",
        status="running",
        original_start_tick=START_TICK,
        original_last_input_tick=START_TICK + steps - 1,
        requested_steps=steps,
        accepted_scene_steps=0,
        recorded_steps=0,
        steps=[],
        interpretation=dict(
            advancement="one public Scene initialized once; later q/v come only from its accepted bank",
            command_mode="open-loop original held ctrl/qfrc_applied/xfrc_applied, not recomputed feedback",
            material="new declared PL pressure prism and finite T sides; not the older capsule midpoint law",
            force_phase="final-velocity law on the accepted input geometry",
            geometry_phase="accepted step input, not endpoint geometry",
            separation="zero normal force and empty positive-area clip are distinct; strict disjointness needs its own witness",
            physical_calibration_validated=False,
            geometric_accuracy_validated=False,
        ),
        events=dict(
            zero_normal_load_input_ticks=[],
            empty_clip_input_ticks=[],
            strict_disjoint_input_ticks=[],
        ),
    )
    stage, current_tick = "input_validation", START_TICK
    # Exclusive creation protects all previous runs. Rewrites affect only this
    # newly owned artifact; serialize before seeking so a JSON error keeps it.
    with output.open("x+", encoding="utf-8") as stream:
        _write_progress(stream, report)
        try:
            entries = load_entries()
            initial = entries["last_push_entry"]
            if initial["tick"] != START_TICK or entries["withdrawn"]["tick"] != LAST_TICK:
                raise ValueError("portable input window differs from the fixed withdrawal interval")
            commands = list(iter_controls(trajectory, START_TICK, START_TICK + steps))
            if (
                len(commands) != steps
                or commands[0]["trace_line_sha256"] != initial["trace_line_sha256"]
            ):
                raise ValueError("original control interval does not bind to last_push_entry")
            original = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
            if original.source_sha256 != initial["source_sha256"] or original.timestep_s != H:
                raise ValueError("original model/source binding differs")
            model = mj.MjModel.from_xml_string(original.mjcf)
            if (model.nq, model.nv, model.nu) != (9, 8, 2):
                raise ValueError("expected the original RR/free-object layout")
            q = legal_initial_q(model, initial["q"])
            free_address = int(model.jnt_qposadr[model.joint("object-free").id])
            spec, asset = build_scene_spec(original.mjcf), build_side_asset(model)
            world = int(initial["world"])
            declaration = MuJoCoTyxWorldSpec(
                world, spec.key, MuJoCoInitialState(q, initial["v"], time=initial["time"])
            )
            report["provenance"] = dict(
                original_trajectory_directory=str(trajectory),
                source_sha256=original.source_sha256,
                original_trace_sha256=initial["saved_trace_sha256"],
                original_manifest_sha256=_digest(trajectory / "manifest.json"),
                portable_inputs_sha256=_digest(DEFAULT_INPUTS),
                script_sha256=_digest(Path(__file__)),
                initial_trace_line_sha256=initial["trace_line_sha256"],
                original_archive_sha256=commands[0]["archive_sha256"],
                original_model_sha256=commands[0]["model_sha256"],
                original_control_window_sha256=_payload_hash(
                    [[c["tick"], c["trace_line_sha256"]] for c in commands]
                ),
                mujoco_version=mj.__version__,
                numpy_version=np.__version__,
            )
            report["initialization"] = dict(
                label="last_push_entry",
                original_tick=initial["tick"],
                original_time=initial["time"],
                archived_q=initial["q"].tolist(),
                declared_q=q.tolist(),
                archived_v=initial["v"].tolist(),
                normalization_max_abs=float(np.max(abs(q - initial["q"]))),
                normalized_once=True,
                world=world,
                model_key=spec.key,
            )
            stage = "scene_construction"
            _write_progress(stream, report)
            with MuJoCoTyxHeterogeneousBatch(
                (spec,), (declaration,), mode="cuda", cache_root=cache_root
            ) as batch:
                scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (asset,)})
                binding = batch.layout[world]
                if binding.primary_sha256 != original.source_sha256:
                    raise ValueError("public Scene did not compile the original RR source")
                report["provenance"].update(
                    compiled_primary_sha256=binding.primary_sha256,
                    compiled_model_sha256=binding.model_sha256,
                    compiled_archive_sha256=binding.archive_sha256,
                )
                report["provenance"]["scene_asset_sha256"] = scene.asset_sha256
                report["material"] = dict(
                    identity=dict(
                        compliant_body=IDENTITY.compliant_body,
                        rigid_body=IDENTITY.rigid_body,
                        compliant_material=IDENTITY.compliant_material,
                        rigid_material=IDENTITY.rigid_material,
                    ),
                    pressure_tetrahedra=len(asset.pressure_local.cells),
                    rigid_faces=len(asset.rigid_local.faces),
                    center_pressure_pa=18000.0,
                    mu=MU,
                    slip_speed_m_s=SLIP,
                    contact_capacity=spec.contact_capacity,
                    constraint_capacity=spec.constraint_capacity,
                )
                prior = batch.integration_snapshot()[world]
                report["initialization"]["actual_initial_state"] = _state(prior, free_address)
                report["initialization"]["runtime_quantization_max_abs"] = float(
                    np.max(abs(prior.q - q))
                )
                report["final_state"] = _state(prior, free_address)
                _write_progress(stream, report)
                for index, command in enumerate(commands):
                    current_tick = command["tick"]
                    if current_tick != START_TICK + index or prior.tick != index:
                        raise RuntimeError(
                            "original command and continuous local Scene tick differ"
                        )
                    stage = "set_original_controls"
                    batch.set_controls({world: control_from_record(command)})
                    before = batch.integration_snapshot()[world]
                    if (
                        before.tick != prior.tick
                        or before.time != prior.time
                        or not np.array_equal(before.q, prior.q)
                        or not np.array_equal(before.v, prior.v)
                    ):
                        raise RuntimeError("setting controls changed the preceding accepted q/v")
                    stage = "scene_step"
                    start = perf_counter()
                    accepted = scene.step(H, expected_revision=batch.checkpoint_revision)
                    elapsed = perf_counter() - start
                    if dict(accepted) != {world: True}:
                        raise RuntimeError("single-world Scene step did not accept")
                    report["accepted_scene_steps"] += 1
                    stage = "record_accepted_step"
                    after = batch.integration_snapshot()[world]
                    report["final_state"] = _state(after, free_address)
                    row = _step_record(
                        model,
                        asset,
                        before,
                        after,
                        scene.last_geometry[world][0],
                        command,
                        free_address,
                        elapsed,
                    )
                    report["steps"].append(row)
                    report["recorded_steps"] += 1
                    for field, condition in (
                        ("zero_normal_load_input_ticks", row["area"]["zero_normal_load"]),
                        (
                            "empty_clip_input_ticks",
                            row["geometry"]["no_positive_area_intersection"],
                        ),
                        (
                            "strict_disjoint_input_ticks",
                            row["geometry"]["strict_disjoint_certified"],
                        ),
                    ):
                        if condition:
                            report["events"][field].append(current_tick)
                    report["last_original_output_tick"] = command["output_tick"]
                    prior = after
                    _write_progress(stream, report)
                stage = "scene_cleanup"
            report["status"] = "complete"
        except Exception as error:
            report.update(
                status="partial_failure",
                failure_stage=stage,
                failed_original_input_tick=current_tick,
                errors=_error_chain(error),
            )
        _write_progress(stream, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--steps", type=int, default=MAX_STEPS, help="prefix length, 1..69; default 69"
    )
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= MAX_STEPS:
        parser.error("--steps must be in 1..69")
    report = run_window(args.trajectory, args.output, steps=args.steps, cache_root=args.cache_root)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
