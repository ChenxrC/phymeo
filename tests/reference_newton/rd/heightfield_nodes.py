"""Read the material-node subset of RD-HEIGHTFIELD without requesting a full cloth surface."""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN_ID = "RD-HEIGHTFIELD"
ARCHIVE = HERE.parents[2] / "docs" / "benchmarks" / "newton-1.7-reference-observations.json"


def measure_heightfield_nodes(case, *, device, cache_root=None):
    """Query the frozen material centers/radii and report only node-to-terrain geometry."""
    import phymeo as ph

    if case["id"] != FROZEN_ID or device not in ("cpu", "cuda:0"):
        raise ValueError("use RD-HEIGHTFIELD with cpu or cuda:0")
    p = case["inputs"]
    if (p["soft_gap_m"], p["shape_margin_m"]) != (0, 0):
        raise ValueError("this frozen node reference uses zero collision margin")
    heights = np.asarray(p["heights_m"], dtype=float)
    rows, cols = heights.shape
    hx, hy = p["half_extents_xy_m"]
    field = ph.Heightfield(
        heights, (2 * hx / (cols - 1), 2 * hy / (rows - 1)), origin=(-hx, -hy, 0)
    )
    terrain = ph.CollisionShape(
        ph.CollisionKey(0, 0, 0), "heightfield", tuple(range(field.vertex_count)), heightfield=field
    )
    vertices = field.world_vertices()
    paths = {terrain.key: ph.CollisionMotion(vertices)}
    nodes = []
    for i, xyz in enumerate(p["vertices_m"]):
        node = ph.CollisionShape(
            ph.CollisionKey(0, 1, i), "rounded_particle", (i,), p["particle_radius_m"]
        )
        paths[node.key] = ph.CollisionMotion([xyz])
        nodes.append(node)
    model = ph.CollisionModel([terrain, *nodes])
    backend = "host" if device == "cpu" else "cuda"
    with ph.CollisionWorkspace(model, backend, cache_root=cache_root) as work:
        result = ph.detect(model, paths, ph.CollisionPolicy(margin=0.0), work)
        stats = work.stats()
    if backend == "cuda" and (
        stats["backend"] != "cuda"
        or stats["completed_launches"] < 2
        or stats["dynamic_uploads"] == 0
        or stats["downloads"] == 0
    ):
        raise AssertionError("missing actual TYX CUDA broad-phase execution and readback")
    if result.status != "ok":
        raise AssertionError(f"terrain node query did not complete: {result.status}")
    pairs = {pair.second: pair for pair in result.pairs if pair.first == terrain.key}
    if set(pairs) != {node.key for node in nodes}:
        raise AssertionError("terrain-node rows do not cover every registered material point")
    tol = case["tolerances"]
    records = []
    for node in nodes:
        pair = pairs[node.key]
        if pair.family != "RD" or pair.status not in ("separated", "touching", "overlap"):
            raise AssertionError(f"unexpected terrain node result {pair}")
        index = node.vertex_ids[0]
        center = paths[node.key].start[0]
        row = dict(
            material_id=index,
            center_m=center.tolist(),
            status=pair.status,
            candidate=pair.candidate,
            reason=pair.reason,
        )
        if pair.witness is not None:
            w, address = pair.witness, pair.heightfield
            if w.second_weights != ((index, 1.0),) or w.second_offset is None or address is None:
                raise AssertionError("terrain witness lost its material node, skin offset or face")
            ground = np.asarray(w.first_point)
            skin = np.asarray(w.second_point)
            direction = np.asarray(w.normal)
            offset = np.asarray(w.second_offset)
            np.testing.assert_allclose(center + offset, skin, rtol=0, atol=tol["newton_point_m"])
            np.testing.assert_allclose(
                offset, -node.radius * direction, rtol=0, atol=tol["newton_point_m"]
            )
            np.testing.assert_allclose(
                address.barycentric @ vertices[list(address.vertex_ids)],
                ground,
                rtol=0,
                atol=tol["newton_point_m"],
            )
            if set(i for i, _ in w.first_weights) != set(address.vertex_ids):
                raise AssertionError("terrain witness lost its triangle material IDs")
            np.testing.assert_allclose(
                sum(vertices[i] * a for i, a in w.first_weights),
                ground,
                rtol=0,
                atol=tol["newton_point_m"],
            )
            row.update(
                terrain_point_m=ground.tolist(),
                skin_point_m=skin.tolist(),
                skin_offset_m=offset.tolist(),
                normal_terrain_to_node=direction.tolist(),
                gap_m=w.gap_m,
                cell=address.cell,
                triangle=address.triangle,
                terrain_weights=w.first_weights,
                node_weights=w.second_weights,
                witness_source=w.source,
            )
        elif pair.status != "separated":
            raise AssertionError("a contacting material node requires a geometric witness")
        records.append(row)
    expected = case["expected"]
    hits = [r for r in records if r["status"] in ("touching", "overlap")]
    features = [[r["material_id"]] for r in hits]
    if features != expected["particle_features"]:
        raise AssertionError(f"material features {features!r} != {expected['particle_features']!r}")
    for r in hits:
        np.testing.assert_allclose(r["gap_m"], expected["gap_m"], atol=tol["newton_gap_m"], rtol=0)
        np.testing.assert_allclose(
            r["normal_terrain_to_node"], expected["normal"], atol=tol["newton_normal"], rtol=0
        )
        np.testing.assert_allclose(
            r["terrain_point_m"],
            [*r["center_m"][:2], expected["top_z_m"]],
            atol=tol["newton_point_m"],
            rtol=0,
        )
    return dict(
        id="RD-HEIGHTFIELD-NODES",
        source_case=case["id"],
        device=device,
        particle_features=features,
        nodes=records,
        stats=stats,
        unmeasured_source_fields=["full_features", "warning_contains"],
    )


def archived_particle_comparison(case, measured):
    """Compare the common node fields against existing Newton reports; never start Newton."""
    archive = json.loads(ARCHIVE.read_text(encoding="utf-8"))
    node_rows = {r["material_id"]: r for r in measured["nodes"]}
    out = {}
    tol = case["tolerances"]
    for run in archive["runs"]:
        if run["engine"] != "newton" or run["phase"] != "initial_matrix":
            continue
        for row in run["records"]:
            if row["id"] != case["id"]:
                continue
            if row["status"] != "pass":
                raise AssertionError(f"{run['run']} did not publish a completed observation")
            particles = row["observed"]["particle_default"]["records"]
            features = [[record["particle"]] for record in particles]
            if features != measured["particle_features"]:
                raise AssertionError("archived Newton node features differ")
            for record in particles:
                local = node_rows[record["particle"]]
                for actual, archived, budget in (
                    ("center_m", "soft_point_m", "newton_point_m"),
                    ("terrain_point_m", "rigid_point_m", "newton_point_m"),
                    ("normal_terrain_to_node", "normal", "newton_normal"),
                    ("gap_m", "gap_m", "newton_gap_m"),
                ):
                    np.testing.assert_allclose(
                        local[actual], record[archived], atol=tol[budget], rtol=0
                    )
            out[run["device"]] = dict(
                run=run["run"],
                original_report_sha256=run["original_report_sha256"],
                compared="particle_default",
                particle_features=features,
            )
    if set(out) != {"cpu", "cuda:0"}:
        raise AssertionError("expected both archived Newton particle-default readbacks")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--out", type=Path)
    options = parser.parse_args()
    rows = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]
    case = next(row for row in rows if row["id"] == FROZEN_ID)
    result = measure_heightfield_nodes(case, device=options.device, cache_root=options.cache_root)
    result["archived_newton_subset"] = archived_particle_comparison(case, result)
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    if options.out is not None:
        with options.out.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
