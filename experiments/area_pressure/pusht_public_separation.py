"""Verify sampled finite-material separation in an accepted public Scene archive.

python -m experiments.area_pressure.pusht_public_separation --input WINDOW.json
    --output NEW.json [--trajectory ORIGINAL_DIR]

This CPU-only verifier uses mj_kinematics, never advances q/v, and does not reuse
the device body poses. A missing separating witness means unknown, not overlap.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import mujoco as mj
import numpy as np

from phymeo.backends.mujoco_heterogeneous_area_scene import _asset_digest
from phymeo.training.manipulation_tasks import manipulation_scenes

from .pusht_entry_rows import (
    DEFAULT_INPUTS,
    _array,
    _digest,
    _read_json,
    iter_controls,
    load_entries,
)
from .pusht_public_scene import (
    CHANNELS,
    IDENTITY,
    MAX_STEPS,
    MU,
    SLIP,
    START_TICK,
    H,
    _payload_hash,
    build_side_asset,
    legal_initial_q,
)


def _require(condition, message):
    if not condition:
        raise ValueError("public separation: " + message)


def _checked_state(state, tick, address):
    _require(
        type(state["local_tick"]) is int and state["local_tick"] == tick, "local step identity"
    )
    q, v = _array(state["q"], (9,), "q"), _array(state["v"], (8,), "v")
    _require(np.isfinite(state["time"]), "finite state time")
    _require(np.array_equal(q, q.astype(np.float32)), "accepted q is not FP32")
    _require(np.array_equal(v, v.astype(np.float32)), "accepted v is not FP32")
    _require(
        np.array_equal(state["object_position_world_m"], q[address : address + 3]),
        "object translation",
    )
    _require(
        np.array_equal(state["object_quaternion_wxyz"], q[address + 3 : address + 7]),
        "object quaternion",
    )
    return q, v


def _validate(window, trajectory):
    rows = window["steps"]
    n = len(rows)
    _require(window["schema"] == "phymeo-pusht-public-scene-window-v1", "window schema")
    _require(
        window["status"] == "complete" and 1 <= n <= MAX_STEPS, "completed bounded window required"
    )
    _require(
        all(window[k] == n for k in ("requested_steps", "accepted_scene_steps", "recorded_steps")),
        "step counts",
    )
    _require(
        window["original_start_tick"] == START_TICK
        and window["original_last_input_tick"] == START_TICK + n - 1,
        "original interval",
    )
    source = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    model = mj.MjModel.from_xml_string(source.mjcf)
    _require(
        (model.nq, model.nv, model.nu, model.nbody) == (9, 8, 2, 5) and source.timestep_s == H,
        "original model layout/timestep",
    )
    initial = load_entries()["last_push_entry"]
    commands = list(iter_controls(trajectory, START_TICK, START_TICK + n))
    _require(len(commands) == n and initial["tick"] == START_TICK, "original commands/initial tick")
    _require(
        commands[0]["trace_line_sha256"] == initial["trace_line_sha256"], "initial original line"
    )
    provenance, init = window["provenance"], window["initialization"]
    binding = dict(
        source_sha256=source.source_sha256,
        compiled_primary_sha256=source.source_sha256,
        original_trace_sha256=initial["saved_trace_sha256"],
        original_manifest_sha256=_digest(trajectory / "manifest.json"),
        portable_inputs_sha256=_digest(DEFAULT_INPUTS),
        initial_trace_line_sha256=initial["trace_line_sha256"],
        original_archive_sha256=commands[0]["archive_sha256"],
        original_model_sha256=commands[0]["model_sha256"],
        original_control_window_sha256=_payload_hash(
            [[c["tick"], c["trace_line_sha256"]] for c in commands]
        ),
    )
    _require(source.source_sha256 == initial["source_sha256"], "portable source binding")
    for key, value in binding.items():
        _require(provenance[key] == value, key + " binding")
    asset = build_side_asset(model)
    material_digest = _asset_digest({init["model_key"]: (asset,)})
    _require(
        material_digest == provenance["scene_asset_sha256"],
        "material vertices/connectivity/pressure binding",
    )
    material = window["material"]
    identity = {
        k: getattr(IDENTITY, k)
        for k in ("compliant_body", "rigid_body", "compliant_material", "rigid_material")
    }
    _require(material["identity"] == identity, "material identity")
    _require(
        material["mu"] == MU
        and material["slip_speed_m_s"] == SLIP
        and material["center_pressure_pa"] == 18000.0,
        "material law",
    )
    _require(
        material["pressure_tetrahedra"] == len(asset.pressure_local.cells)
        and material["rigid_faces"] == len(asset.rigid_local.faces),
        "material counts",
    )
    _require(
        len(asset.pressure_local.cells) * len(asset.rigid_local.faces) <= 2048,
        "bounded cell/face pairs",
    )
    declared = legal_initial_q(model, initial["q"])
    _require(
        init["label"] == "last_push_entry"
        and init["original_tick"] == START_TICK
        and init["original_time"] == initial["time"]
        and init["world"] == initial["world"]
        and init["normalized_once"] is True,
        "initial identity",
    )
    for key, value in (
        ("archived_q", initial["q"]),
        ("declared_q", declared),
        ("archived_v", initial["v"]),
    ):
        _require(np.array_equal(init[key], value), key + " binding")
    address = int(model.jnt_qposadr[model.joint("object-free").id])
    prior = init["actual_initial_state"]
    q, v = _checked_state(prior, 0, address)
    _require(
        np.array_equal(q, declared.astype(np.float32))
        and np.array_equal(v, initial["v"].astype(np.float32)),
        "initial accepted quantization",
    )
    hi = np.float32(initial["time"])
    lo = np.float32(float(hi) - initial["time"])
    _require(prior["time"] == float(hi) - float(lo), "initial compensated clock")
    states = [(START_TICK, "initial_input", prior)]
    for index, (row, command) in enumerate(zip(rows, commands, strict=True)):
        _require(
            row["original_input_tick"] == command["tick"] == START_TICK + index
            and row["original_output_tick"] == command["output_tick"] == START_TICK + index + 1,
            "original row identity",
        )
        before, after = row["input_state"], row["output_state"]
        _checked_state(before, index, address)
        _, out_v = _checked_state(after, index + 1, address)
        _require(before == prior, "q/v/time continuity with preceding accepted output")
        increment = np.float32(np.float32(H) - lo)
        total = np.float32(hi + increment)
        lo = np.float32(np.float32(total - hi) - increment)
        hi = total
        _require(
            after["time"] == float(hi) - float(lo) and after["time"] > before["time"],
            "one compensated timestep",
        )
        stored = row["control"]
        _require(set(stored) == set(command) | {"channels_sha256"}, "control fields")
        for key, value in command.items():
            expected = value.tolist() if isinstance(value, np.ndarray) else value
            _require(stored[key] == expected, "original control value/provenance: " + key)
        channels = {key: np.asarray(command[key]).tolist() for key in CHANNELS}
        _require(
            stored["channels_sha256"] == _payload_hash(channels)
            and row["stored_control_matches_original"] is True,
            "three-channel control digest",
        )
        _require(
            np.array_equal(row["solver"]["output_velocity"], out_v), "accepted solver velocity"
        )
        meta = row["solver"]["meta"]
        _require(
            len(meta) == 4
            and meta[0] == 0
            and meta[3] == 1
            and meta[2] == meta[1] + 1
            and 1 <= meta[2] <= 25,
            "accepted solver status/calls",
        )
        states.append((command["output_tick"], "accepted_output", after))
        prior = after
    _require(
        window["final_state"] == prior and window["last_original_output_tick"] == START_TICK + n,
        "final summary binding",
    )
    binding["scene_asset_sha256"] = material_digest
    return model, asset, states, binding


def _outward(value, sign):
    return np.nextafter(value, sign * np.inf)


def _world_vertices(local, rotation, position):
    """Enclose the exact transform of the supplied FP64 CPU pose/asset values."""
    center = local @ rotation.T + position
    lower, upper = np.zeros_like(center), np.zeros_like(center)
    for axis in range(3):
        product = local[:, axis, None] * rotation[None, :, axis]
        lower = _outward(lower + _outward(product, -1), -1)
        upper = _outward(upper + _outward(product, 1), 1)
    lower, upper = _outward(lower + position, -1), _outward(upper + position, 1)
    _require(
        np.isfinite(center).all() and np.isfinite(lower).all() and np.isfinite(upper).all(),
        "finite transformed geometry",
    )
    return center, lower, upper


def _projection(lower, upper, axis):
    left, right = np.zeros(len(lower)), np.zeros(len(lower))
    for k in range(3):
        lo, hi = (lower[:, k], upper[:, k]) if axis[k] >= 0 else (upper[:, k], lower[:, k])
        left = _outward(left + _outward(lo * axis[k], -1), -1)
        right = _outward(right + _outward(hi * axis[k], 1), 1)
    return left, right


def _pair_witness(tet, tri, tet_lo, tet_hi, tri_lo, tri_hi):
    """Any positive axis gap suffices; failure of these axes proves nothing."""
    axes = [("triangle_normal", np.cross(tri[1] - tri[0], tri[2] - tri[0]))]
    axes.extend(
        ("tet_face_" + "".join(map(str, f)), np.cross(tet[f[1]] - tet[f[0]], tet[f[2]] - tet[f[0]]))
        for f in combinations(range(4), 3)
    )
    for i, j in combinations(range(4), 2):
        for start, end in ((0, 1), (1, 2), (2, 0)):
            axes.append(
                (
                    f"edge_cross_{i}{j}_{start}{end}",
                    np.cross(tet[j] - tet[i], tri[end] - tri[start]),
                )
            )
    best = None
    for kind, axis in axes:
        scale = float(np.max(np.abs(axis)))
        if not np.isfinite(scale) or scale == 0:
            continue
        axis = axis / scale
        norm_lo, norm_hi = _projection(axis[None], axis[None], axis)
        norm_lower = float(_outward(np.sqrt(max(0.0, norm_lo[0])), -1))
        norm_upper = float(_outward(np.sqrt(norm_hi[0]), 1))
        if norm_lower <= 0 or not np.isfinite(norm_upper):
            continue
        a_lo, a_hi = _projection(tet_lo, tet_hi, axis)
        b_lo, b_hi = _projection(tri_lo, tri_hi, axis)
        forward = float(_outward(b_lo.min() - a_hi.max(), -1))
        reverse = float(_outward(a_lo.min() - b_hi.max(), -1))
        gap = max(forward, reverse)
        margin = float(_outward(gap / (norm_upper if gap >= 0 else norm_lower), -1))
        if best is None or margin > best["gap_lower_m"]:
            best = dict(
                kind=kind,
                axis=axis.tolist(),
                direction="tet_to_triangle" if forward >= reverse else "triangle_to_tet",
                gap_lower_m=margin,
            )
    return best


def _sample(model, data, asset, tick, phase, state):
    data.qpos[:] = state["q"]
    data.qvel[:] = state["v"]
    data.time = state["time"]
    mj.mj_kinematics(model, data)
    _require(np.array_equal(data.qpos, state["q"]), "kinematics changed accepted q")
    bodies = (model.body(IDENTITY.compliant_body).id, model.body(IDENTITY.rigid_body).id)
    meshes = (asset.pressure_local, asset.rigid_local)
    transformed = [
        _world_vertices(mesh.world_vertices, data.xmat[body].reshape(3, 3), data.xpos[body])
        for mesh, body in zip(meshes, bodies, strict=True)
    ]
    cells, faces = asset.pressure_local.cells, asset.rigid_local.faces
    ta, la, ua = (v[cells] for v in transformed[0])
    tb, lb, ub = (v[faces] for v in transformed[1])
    gaps, positive, limiting = [], 0, None
    all_witnesses = True
    for i in range(len(cells)):
        row = []
        for j in range(len(faces)):
            witness = _pair_witness(ta[i], tb[j], la[i], ua[i], lb[j], ub[j])
            gap = None if witness is None else witness["gap_lower_m"]
            row.append(gap)
            if gap is None:
                all_witnesses = False
            else:
                positive += gap > 0
                if limiting is None or gap < limiting["gap_lower_m"]:
                    limiting = dict(cell_index=i, face_index=j, **witness)
        gaps.append(row)
    strict = all_witnesses and positive == len(cells) * len(faces)
    return dict(
        original_state_tick=tick,
        phase=phase,
        local_tick=state["local_tick"],
        time=state["time"],
        state_sha256=_payload_hash(state),
        q=state["q"],
        v=state["v"],
        status="strictly_separated" if strict else "unknown",
        strict_separation_certified=bool(strict),
        pair_count=len(cells) * len(faces),
        positive_pair_count=int(positive),
        minimum_certified_margin_m=limiting["gap_lower_m"] if strict else None,
        limiting_evaluated_pair=limiting,
        pair_axis_gap_lower_m=gaps,
        cpu_body_poses=[
            dict(
                body_id=int(b),
                position=data.xpos[b].tolist(),
                rotation=data.xmat[b].reshape(3, 3).tolist(),
            )
            for b in bodies
        ],
    )


def verify_window(input_path, output, trajectory=None):
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    report = dict(
        schema="phymeo-pusht-public-sampled-separation-v1",
        status="validation_failed",
        samples=[],
        interpretation=dict(
            geometry="declared PL pressure tetrahedra versus declared finite T negative-y triangles only",
            kinematics="independent CPU mj_kinematics of saved accepted q; no dynamics, integration or CUDA",
            pose_scope="CPU reconstructed poses, not the original device pose/clip result",
            arithmetic="outward-rounded FP64 transform/projection intervals for the supplied CPU poses and local vertices; does not bound CPU FK versus device FK",
            separation="every tetrahedron/triangle pair needs its own positive axis gap; otherwise unknown",
            persistence="saved state samples only; no continuous-time/CCD or between-step guarantee",
            tick_phase="initial input tick1699 plus accepted output state ticks1700 onward; force/clip rows refer to input ticks",
            original_full_geometry_validated=False,
            physical_calibration_validated=False,
            geometric_accuracy_validated=False,
            archived_dynamics_independently_resolved=False,
        ),
    )
    with output.open("x", encoding="utf-8") as stream:
        try:
            _require(input_path.stat().st_size <= 64 * 1024 * 1024, "bounded input size")
            window = _read_json(input_path)
            trajectory = Path(
                trajectory or window["provenance"]["original_trajectory_directory"]
            ).resolve()
            model, asset, states, binding = _validate(window, trajectory)
            report["provenance"] = dict(
                input=str(input_path),
                input_sha256=_digest(input_path),
                trajectory=str(trajectory),
                verified_bindings=binding,
                verifier_sha256=_digest(Path(__file__)),
                mujoco_version=mj.__version__,
                numpy_version=np.__version__,
            )
            report["validation"] = dict(
                original_controls="all three channels and per-line provenance compared exactly",
                continuity="q/v/time/local tick/original tick/final summary checked",
                clock="exact FP32 compensated timestep recurrence",
                model="source SHA checked; compiled binary is not rebuilt",
                material="complete declared asset digest matched",
            )
            data = mj.MjData(model)
            report["status"] = "partial_failure"
            for tick, phase, state in states:
                report["samples"].append(_sample(model, data, asset, tick, phase, state))
            strict = [
                s["original_state_tick"]
                for s in report["samples"]
                if s["strict_separation_certified"]
            ]
            suffix = len(states)
            while suffix and report["samples"][suffix - 1]["strict_separation_certified"]:
                suffix -= 1
            report["events"] = dict(
                first_strict_state_tick=strict[0] if strict else None,
                strict_state_ticks=strict,
                first_sustained_sampled_state_tick=states[suffix][0]
                if suffix < len(states)
                else None,
                sustained_sample_count=len(states) - suffix,
                final_state_tick=states[-1][0],
            )
            report["status"] = "complete"
        except Exception as error:
            report["error"] = dict(type=type(error).__name__, message=str(error)[:4000])
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--trajectory", type=Path, help="optional relocation of the original bound archive"
    )
    args = parser.parse_args()
    report = verify_window(args.input, args.output, args.trajectory)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
