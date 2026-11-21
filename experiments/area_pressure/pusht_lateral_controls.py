"""Fresh, one-step public CUDA controls at original A face4/face13 inputs.

Selection is predeclared and uses only saved state plus independent CPU geometry:
first entry/push input reaching 0.1 mm PL depth, required to remain at most 1 mm.
The source route's first input is the separation control. No rollout is resumed,
no previous receipt is inherited, and no device force chooses the input.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

MIN_DEPTH, MAX_DEPTH = 0.0001, 0.001
COVERAGES = ("negative-y", "all-lateral")


def _require(ok, message):
    if not ok:
        raise ValueError("lateral controls: " + message)


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream)


def select_shallow(rows, measure):
    """Never skip an invalid first crossing to select a later successful force."""
    for row in rows:
        if row["controller"]["stage"] not in ("entry", "push"):
            continue
        evidence = measure(row)
        depth = evidence["depth_m"]
        _require(np.isfinite(depth) and depth >= 0, "finite nonnegative selection depth")
        if depth < MIN_DEPTH:
            continue
        _require(
            depth <= MAX_DEPTH and evidence["area_m2"] > 0 and evidence["normals_match"],
            "first crossing is deep or on an unintended face",
        )
        return row, evidence
    raise ValueError("lateral controls: no predeclared shallow crossing in the saved route")


def load_source(diagnostic_path, expected_face):
    """Bind the existing completed-route audit and exact submitted candidate bytes."""
    diagnostic_path = Path(diagnostic_path).resolve()
    diagnostic = _read(diagnostic_path)
    source = diagnostic["source"]
    summary_path, audit_path = Path(source["summary"]), Path(source["audit"])
    summary, audit = _read(summary_path), _read(audit_path)
    if audit["schema"] == "phymeo-pusht-pose-replan-comparison-v1":
        _require(audit["status"] == "verified-saved-prefixes", "verified source comparison")
        audit = audit["cases"]["A"]
    _require(
        diagnostic["schema"] == "phymeo-pusht-a-contact-miss-v1"
        and diagnostic["selected_contact"] == expected_face
        and summary["schema"] == "phymeo-pusht-pose-replan-case-v1"
        and summary["key"] == "A"
        and summary["status"] == "route_complete"
        and _digest(summary_path) == source["summary_sha256"]
        and audit["schema"] == "phymeo-pusht-pose-replan-case-audit-v1"
        and audit["status"] == "verified-saved-prefix"
        and audit["source"]["sha256"] == source["summary_sha256"]
        and audit["source"]["journal_sha256"]
        == summary["journal"]["sha256"]
        == source["journal_prefix_sha256"],
        "actual original route/source audit",
    )
    route_path = (summary_path.parent / summary["ordinary_route"]["route_file"]).resolve()
    route = _read(route_path)
    _require(
        _digest(route_path) == source["route_sha256"] == summary["active_route"]["sha256"]
        and route["route"]["proposal"]["index"] == expected_face,
        "selected face/route bytes",
    )
    journal_path = (summary_path.parent / summary["journal"]["file"]).resolve()
    retained, pending, rows, accepted = summary["initial_state"], None, [], 0
    digest = hashlib.sha256()
    with journal_path.open("rb") as stream:
        for _ in range(summary["journal"]["records"]):
            line = stream.readline()
            _require(bool(line) and line.endswith(b"\n"), "complete declared event prefix")
            digest.update(line)
            event = json.loads(line)
            payload = event["payload"]
            if event["event"] == "candidate":
                _require(
                    pending is None and payload["input_state"] == retained, "source state chain"
                )
                if payload["selected"]:
                    _require(payload["restored"] is False, "selected source was restored")
                    pending = (event["trial_index"], payload, hashlib.sha256(line).hexdigest())
                    retained = payload["output_state"]
                else:
                    _require(
                        payload["restored"] and payload["rollback_verification"]["passed"],
                        "unselected source rollback",
                    )
            elif event["event"] == "accepted_callback":
                _require(
                    pending is not None and pending[0] == event["trial_index"], "source callback"
                )
                row, line_sha = pending[1:]
                _require(
                    row["measurement"] == payload["feedback_accepted"]["measurement"],
                    "source accepted measurement",
                )
                if row["phase"] == "regular":
                    command = row.get("submitted_control", row.get("control"))
                    _require(
                        command["tick"] == row["original_input_tick"]
                        and command["output_tick"] == row["original_output_tick"],
                        "source command phase",
                    )
                    rows.append(
                        dict(
                            original_input_tick=row["original_input_tick"],
                            original_output_tick=row["original_output_tick"],
                            input_state=row["input_state"],
                            control=command,
                            controller={"stage": row["controller"]["stage"]},
                            source_candidate_line_sha256=line_sha,
                        )
                    )
                accepted += 1
                pending = None
    _require(
        pending is None
        and digest.hexdigest() == source["journal_prefix_sha256"]
        and accepted == summary["recorded_steps"] == audit["recorded_steps"]
        and retained == summary["final_state"]
        and len(rows) == source["complete_route_steps"]
        and rows[0]["original_input_tick"] == summary["active_route"]["start_tick"]
        and rows[0]["input_state"]["q"] == route["q"]
        and rows[0]["input_state"]["v"] == route["v"],
        "complete saved physical route prefix",
    )
    return dict(
        rows=rows,
        route=route,
        summary=summary,
        diagnostic=diagnostic,
        provenance=dict(
            diagnostic_path=str(diagnostic_path),
            diagnostic_sha256=_digest(diagnostic_path),
            summary_path=str(summary_path),
            summary_sha256=_digest(summary_path),
            audit_path=str(audit_path),
            audit_sha256=_digest(audit_path),
            journal_path=str(journal_path),
            journal_sha256=digest.hexdigest(),
            route_path=str(route_path),
            route_sha256=_digest(route_path),
            selected_contact=expected_face,
        ),
    )


def _fk(model, state):
    import mujoco as mj

    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = state["q"], state["v"], state["time"]
    mj.mj_kinematics(model, data)
    mj.mj_comPos(model, data)
    return data


def _geometry_measure(model, asset, row, normal_local, stats):
    from .pusht_release_material import _cpu_geometry, _world

    bodies = (model.body("tool").id, model.body("object").id)
    data = _fk(model, row["input_state"])
    transformed = _world(asset, data, bodies)
    # Outward-rounded transform intervals enclose every tetrahedron and face.
    # AABB disjointness is sufficient to skip exact clipping, never to infer load.
    soft_lo, soft_hi = transformed[0][1:]
    face_lo, face_hi = transformed[1][1:]
    cells, faces = asset.pressure_local.cells, asset.rigid_local.faces
    lo_a, hi_a = soft_lo[cells].min(axis=1), soft_hi[cells].max(axis=1)
    lo_b, hi_b = face_lo[faces].min(axis=1), face_hi[faces].max(axis=1)
    overlaps = np.all(np.maximum(lo_a[:, None], lo_b) <= np.minimum(hi_a[:, None], hi_b), axis=2)
    stats["inputs_examined"] += 1
    if not np.any(overlaps):
        stats["certified_aabb_skips"] += 1
        return dict(
            depth_m=0.0, area_m2=0.0, normals_match=True, certificate="outward-cell-face-AABBs"
        )
    stats["exact_clips"] += 1
    geometry, _ = _cpu_geometry(asset, data, bodies)
    normal = data.xmat[bodies[1]].reshape(3, 3) @ normal_local
    depth, area, matches = 0.0, 0.0, True
    for triangle in geometry.surface.triangles:
        cross = np.cross(
            triangle.vertices[1] - triangle.vertices[0], triangle.vertices[2] - triangle.vertices[0]
        )
        magnitude = np.linalg.norm(cross)
        _require(magnitude > 0, "positive clipped triangle area")
        outward = cross / magnitude
        effective = triangle.pressure_gradient @ outward
        _require(magnitude > 0 and effective > 0, "unsupported material area/gradient")
        depth = max(depth, float(np.max(triangle.vertex_pressure) / effective))
        area += 0.5 * magnitude
        matches = matches and float(outward @ normal) >= 1 - 1e-6
    return dict(
        depth_m=depth,
        area_m2=float(area),
        normals_match=bool(matches),
        certificate="independent-CPU-finite-PL-clipping",
        triangles=len(geometry.surface.triangles),
    )


def _oracle(model, asset, before, after):
    """CPU mj_forward/FK/J and an independent FP64 common physical-acceleration root."""
    import mujoco as mj

    from phymeo.geometry.area_pressure import quadrature_area_pressure

    from .pusht_entry_rows import assemble_entry
    from .pusht_entry_working_set import _reference
    from .pusht_public_scene import MU, SLIP, H
    from .pusht_release_material import _cpu_geometry

    receipt = after.last_step
    command = before.held_control
    source = assemble_entry(
        model,
        dict(
            q=before.q,
            v=before.v,
            time=before.time,
            ctrl=command.ctrl,
            qfrc_applied=command.qfrc_applied,
            xfrc_applied=command.xfrc_applied,
        ),
    )
    data = source["data"]
    bodies = (model.body("tool").id, model.body("object").id)
    geometry, _ = _cpu_geometry(asset, data, bodies)
    quad = quadrature_area_pressure(geometry.surface)
    ja, jb = (np.zeros((len(quad.points), 3, model.nv)) for _ in range(2))
    for index, point in enumerate(quad.points):
        mj.mj_jac(model, data, ja[index], None, point, bodies[0])
        mj.mj_jac(model, data, jb[index], None, point, bodies[1])
    _require(MU == 0.7 and SLIP == 0.01, "existing independent reference law")
    h = float(np.float32(H))
    area = dict(
        jac=ja - jb,
        normals=quad.normals,
        stiffness=quad.dA * quad.gradient,
        initial_gap=-quad.pressure / quad.gradient,
    )
    expected_v, native_force, force = _reference(
        source, area, before.v, source["mass"] + h * np.diag(model.dof_damping), h
    )
    expected_a = (
        np.linalg.solve(
            source["mass"],
            (source["mass"] + h * np.diag(model.dof_damping)) @ (expected_v - before.v),
        )
        / h
    )
    (pair,) = receipt.raw_contact_fields["area_friction"]
    _require(
        pair["identity"] == asset.identity and tuple(pair["body_ids"]) == bodies,
        "actual material/body identity",
    )
    _require(len(receipt.constraints) == len(source["owners"]), "all original retained native rows")
    # Match native rows by original contact geometry and finite position, not compact index.
    contacts = [c for c in source["contacts"] if c["retained"]]
    native_by_source = {}
    used_contacts = set()
    for index, contact in enumerate(receipt.contacts):
        choices = [
            c
            for c in contacts
            if tuple(c["geometry_ids"]) == tuple(contact.geometry_ids)
            and c["contact_index"] not in used_contacts
        ]
        _require(bool(choices), "missing independent retained contact")
        match = min(choices, key=lambda c: np.linalg.norm(c["position"] - contact.position))
        np.testing.assert_allclose(contact.position, match["position"], atol=5e-5, rtol=2e-4)
        np.testing.assert_allclose(contact.frame, match["frame"], atol=5e-5, rtol=2e-4)
        _require(
            contact.dimension == 3 and contact.active == (match["efc_address"] >= 0),
            "native contact dimension/activation",
        )
        used_contacts.add(match["contact_index"])
        for row in receipt.constraints:
            if row.constraint_type == 6 and row.source_id == index:
                native_by_source[match["efc_address"] + row.side_or_edge] = row
    for row in receipt.constraints:
        if row.constraint_type == 3:
            owner = next(
                o
                for o in source["owners"]
                if o["kind"] == "limit"
                and o["joint_id"] == row.source_id
                and o["side"] == row.side_or_edge
            )
            native_by_source[owner["source_row"]] = row
    _require(
        used_contacts == {c["contact_index"] for c in contacts},
        "complete retained contact bijection",
    )
    _require(set(native_by_source) == set(source["retained_indices"]), "native ownership bijection")
    actual_native = np.array([native_by_source[int(i)].force for i in source["retained_indices"]])
    for i, original in enumerate(source["retained_indices"]):
        row = native_by_source[int(original)]
        np.testing.assert_allclose(row.jacobian, source["jacobian"][i], rtol=2e-4, atol=5e-5)
        np.testing.assert_allclose(
            row.regularization, source["regularization"][i], rtol=3e-4, atol=5e-5
        )
        np.testing.assert_allclose(
            row.reference_acceleration, source["reference_acceleration"][i], rtol=3e-4, atol=5e-5
        )
    np.testing.assert_allclose(actual_native, native_force, rtol=0.003, atol=0.002)
    np.testing.assert_allclose(after.v, expected_v, rtol=5e-4, atol=5e-5)
    np.testing.assert_allclose(h * receipt.qacc, h * expected_a, rtol=5e-4, atol=5e-5)
    observed_a, observed_b = np.asarray(pair["force_on_a"]), np.asarray(pair["force_on_b"])
    np.testing.assert_array_equal(observed_b, -observed_a)
    fn = np.asarray(pair["normal_force"])
    _require(np.all(fn >= 0), "nonnegative actual normal load")
    reference_fn = np.sum(force * quad.normals, axis=1)
    np.testing.assert_allclose(fn.sum(), reference_fn.sum(), rtol=0.003, atol=0.002)
    np.testing.assert_allclose(np.sum(pair["dA"]), quad.total_area_m2, rtol=2e-3, atol=2e-8)
    np.testing.assert_allclose(observed_a.sum(axis=0), force.sum(axis=0), rtol=0.003, atol=0.002)
    points = np.asarray(pair["points"])
    moment = np.cross(points - receipt.dynamics.body_com[bodies[1]], observed_b).sum(axis=0)
    reference_moment = np.cross(quad.points - data.xipos[bodies[1]], -force).sum(axis=0)
    np.testing.assert_allclose(moment, reference_moment, rtol=0.003, atol=2e-4)
    np.testing.assert_allclose(
        pair["torque_on_a_world_origin"], np.cross(points, observed_a), rtol=3e-4, atol=2e-6
    )
    np.testing.assert_array_equal(
        pair["torque_on_b_world_origin"], -pair["torque_on_a_world_origin"]
    )
    total = source["jacobian"].T @ native_force + np.einsum("sij,si->j", ja - jb, force)
    np.testing.assert_allclose(receipt.qfrc_constraint, total, rtol=0.003, atol=0.002)
    integrated = before.q.copy()
    mj.mj_integratePos(model, integrated, expected_v, h)
    np.testing.assert_allclose(after.q, integrated, rtol=5e-6, atol=2e-7)
    return dict(
        fp64_oracle_verified=True,
        opposite_reaction_verified=True,
        normal_load_n=float(fn.sum()),
        reference_normal_load_n=float(reference_fn.sum()),
        clipped_area_m2=float(np.sum(pair["dA"])),
        reference_area_m2=float(quad.total_area_m2),
        force_on_tool_world_n=observed_a.sum(axis=0).tolist(),
        force_on_object_world_n=observed_b.sum(axis=0).tolist(),
        moment_on_object_com_world_nm=moment.tolist(),
        reference_moment_on_object_com_world_nm=reference_moment.tolist(),
        max_velocity_error=float(np.max(abs(after.v - expected_v))),
        max_native_force_error=float(np.max(abs(actual_native - native_force), initial=0)),
        native_rows=len(actual_native),
        samples=len(fn),
    )


def _variant(original, model, row, coverage, cache_root, report):
    from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaPressureScene,
    )
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState

    from .pusht_public_scene import (
        CHANNELS,
        H,
        _state,
        _step_record,
        build_scene_spec,
        build_side_asset,
        control_from_record,
        legal_initial_q,
    )

    state, command = row["input_state"], row["control"]
    q = legal_initial_q(model, state["q"])
    spec = build_scene_spec(original.mjcf, side_coverage=coverage)
    asset = build_side_asset(model, side_coverage=coverage)
    initial = MuJoCoInitialState(q, state["v"], time=state["time"])
    declaration = MuJoCoTyxWorldSpec(6, spec.key, initial)
    qa = int(model.jnt_qposadr[model.joint("object-free").id])
    report.update(
        status="initializing",
        source_q=list(state["q"]),
        declared_legal_q=q.tolist(),
        declared_normalization_delta=(q - np.asarray(state["q"])).tolist(),
        side_coverage=coverage,
        original_model_sha256=original.source_sha256,
        control={name: np.asarray(command[name]).tolist() for name in CHANNELS},
        physical_step_s=H,
    )
    with MuJoCoTyxHeterogeneousBatch(
        (spec,), (declaration,), mode="cuda", cache_root=cache_root
    ) as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (asset,)})
        batch.set_controls({6: control_from_record(command)})
        before = batch.integration_snapshot()[6]
        _require(before.tick == 0 and before.last_step is None, "fresh independent episode")
        np.testing.assert_array_equal(before.q, q.astype(np.float32).astype(float))
        np.testing.assert_array_equal(before.v, np.asarray(state["v"], np.float32).astype(float))
        for name in CHANNELS:
            np.testing.assert_array_equal(getattr(before.held_control, name), command[name])
        report.update(
            status="attempting-physical-step",
            initial_receipt=None,
            scene_asset_sha256=scene.asset_sha256,
            input_state=_state(before, qa),
            actual_input_minus_archived=(before.q - np.asarray(state["q"])).tolist(),
            initial_time_minus_archived=before.time - state["time"],
        )
        started = perf_counter()
        _require(
            scene.step(H, expected_revision=batch.checkpoint_revision) == {6: True},
            "public step accepted",
        )
        wall = perf_counter() - started
        after = batch.integration_snapshot()[6]
        receipt = after.last_step
        report.update(
            status="accepted-awaiting-independent-audit",
            output_state=_state(after, qa),
            receipt_tick=receipt.tick,
            step_wall_s=wall,
        )
        _require(
            receipt.input_time == before.time
            and receipt.output_time == after.time
            and receipt.tick == after.tick == 1,
            "new episode force/state phase",
        )
        # Store the real accepted record before the independent audit. An audit
        # failure must not erase the physical step that actually occurred.
        report["record"] = _step_record(
            model, asset, before, after, scene.last_geometry[6][0], command, qa, wall
        )
        (pair,) = receipt.raw_contact_fields["area_friction"]
        report["raw_normal_force_n"] = np.asarray(pair["normal_force"]).tolist()
        report.update(_oracle(model, asset, before, after))
        report["status"] = "verified-independent-one-step"
    return report


def run(face4, face13, output, *, cache_root=None):
    import mujoco as mj

    from phymeo.backends.mujoco_heterogeneous_area_scene import _asset_digest
    from phymeo.training.manipulation_tasks import manipulation_scenes

    from .pusht_public_scene import _error_chain, _write_progress, build_side_asset

    original = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    model = mj.MjModel.from_xml_string(original.mjcf)
    _require((model.nq, model.nv, model.nu) == (9, 8, 2), "original RR free-object coordinates")
    old_asset_sha = _asset_digest({"original-RR-public-mixed-area": (build_side_asset(model),)})
    asset = build_side_asset(model, side_coverage="all-lateral")
    report = dict(
        schema="phymeo-pusht-lateral-controls-v1",
        status="running",
        cases={},
        sources={},
        selection=dict(
            minimum_depth_m=MIN_DEPTH,
            maximum_depth_m=MAX_DEPTH,
            first_crossing=True,
            state_source="accepted original q/v and submitted action",
            separation="first original route input",
            new_force_selection=False,
        ),
        scope="eight independent one-step resets; no inherited checkpoint/receipt, no trajectory or Goal claim",
    )
    with Path(output).open("x+", encoding="utf-8") as stream:
        for face, path in ((4, face4), (13, face13)):
            try:
                source = load_source(path, face)
                _require(
                    source["diagnostic"]["original_model_sha256"] == original.source_sha256
                    and source["diagnostic"]["material_sha256"]
                    == old_asset_sha
                    == source["summary"]["provenance"]["scene_asset_sha256"],
                    "original source model and unchanged negative-y material",
                )
                report["sources"][str(face)] = source["provenance"]
                origin = _fk(model, dict(q=source["route"]["q"], v=source["route"]["v"], time=0.0))
                rotation = origin.xmat[model.body("object").id].reshape(3, 3)
                normal_local = (
                    rotation.T @ source["route"]["route"]["proposal"]["outward_normal_world"]
                )
                stats = dict(inputs_examined=0, certified_aabb_skips=0, exact_clips=0)

                def measure(row):
                    return _geometry_measure(model, asset, row, normal_local, stats)

                separated = source["rows"][0]
                separated_evidence = measure(separated)
                _require(
                    separated_evidence["depth_m"] == separated_evidence["area_m2"] == 0,
                    "registered route-entry separation control",
                )
                start = perf_counter()
                shallow, evidence = select_shallow(source["rows"], measure)
                report["sources"][str(face)]["selection_cost"] = dict(
                    stats, wall_s=perf_counter() - start
                )
                for label, row, selection in (
                    ("separated", separated, separated_evidence),
                    ("shallow", shallow, evidence),
                ):
                    key = f"face{face}-{label}"
                    case = dict(
                        source_input_tick=row["original_input_tick"],
                        source_input_time=row["input_state"]["time"],
                        source_candidate_line_sha256=row["source_candidate_line_sha256"],
                        selection=selection,
                        variants={},
                    )
                    report["cases"][key] = case
                    for coverage in COVERAGES:
                        variant = case["variants"][coverage] = {}
                        _variant(original, model, row, coverage, cache_root, variant)
                        _write_progress(stream, report)
                    old, new = (case["variants"][coverage] for coverage in COVERAGES)
                    _require(
                        old["input_state"] == new["input_state"]
                        and old["declared_legal_q"] == new["declared_legal_q"]
                        and old["control"] == new["control"],
                        "cross-asset exact actual entry/control",
                    )
                    case["paired_input_exact"] = True
                    _require(
                        label != "separated" or old["normal_load_n"] == new["normal_load_n"] == 0,
                        "separated control produced load",
                    )
                    if label == "shallow":
                        _require(
                            new["normal_load_n"] > 0
                            and new["reference_normal_load_n"] > 0
                            and new["clipped_area_m2"] > 0
                            and new["reference_area_m2"] > 0,
                            "selected shallow new material must really load",
                        )
                        difference = (
                            np.asarray(new["output_state"]["v"])[2:] - old["output_state"]["v"][2:]
                        )
                        case["object_dynamic_difference"] = float(np.linalg.norm(difference))
                        if face == 4:
                            _require(
                                old["normal_load_n"] == old["clipped_area_m2"] == 0
                                and case["object_dynamic_difference"] > 0,
                                "new face actual dynamic positive control",
                            )
                        else:
                            _require(old["normal_load_n"] > 0, "old face retained actual load")
                            for field in (
                                "normal_load_n",
                                "force_on_object_world_n",
                                "moment_on_object_com_world_nm",
                            ):
                                np.testing.assert_allclose(
                                    new[field], old[field], rtol=0.003, atol=0.002
                                )
                            np.testing.assert_allclose(
                                new["output_state"]["v"],
                                old["output_state"]["v"],
                                rtol=5e-5,
                                atol=5e-5,
                            )
                            case["old_side_preserved"] = True
            except Exception as error:
                report.setdefault("failures", {})[str(face)] = _error_chain(error)
            _write_progress(stream, report)
        report["status"] = (
            "partial_failure" if report.get("failures") else "verified-short-public-controls"
        )
        _write_progress(stream, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("face4", "face13", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    report = run(args.face4, args.face13, args.output, cache_root=args.cache_root)
    print(
        json.dumps(
            dict(
                status=report["status"], cases=len(report["cases"]), failures=report.get("failures")
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
