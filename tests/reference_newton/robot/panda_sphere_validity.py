"""Find the first unforced Panda structure on an already integrated sphere trajectory."""

import argparse
import json
from pathlib import Path

import numpy as np

from phymeo.builders.panda import load_panda
from phymeo.dynamics.robot_kinematics import forward_kinematics
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.experiments.panda_cube import _PassiveGeometry
from phymeo.experiments.panda_payload import SampledPayload


def audit(spec, run):
    asset = load_panda()
    geometry = SampledPayload.from_manifest(spec)
    assert geometry.descriptor() == run["payload_geometry"] and geometry.radius is not None
    monitor = _PassiveGeometry(asset)
    q, c, quat = spec["initial_q"], spec["body_position_m"], spec["body_xyzw"]
    cn, half = np.array(spec["table_center_m"]), np.array(spec["table_full_size_m"]) / 2
    first, edge, last_table = None, None, None
    for i, row in enumerate(run["trace"]):
        if first is None:
            gaps = monitor.clearances(
                forward_kinematics(asset.model, np.array(q)),
                np.array(c),
                quaternion_matrix(np.array([quat]))[0],
                geometry.half,
                cn,
                half,
                sphere_radius=geometry.radius,
            )
            first = monitor.unresolved(i, i * spec["dt_s"], *gaps)
        if row["table_N"] > 0:
            last_table = row["t"]
        if edge is None and np.any(np.abs(np.array(c)[:2] - cn[:2]) > half[:2]):
            edge = i * spec["dt_s"]
        q, c, quat = row["q"], row["cube"], row["xyzw"]
    return dict(
        first_unresolved_structure=first,
        first_center_projection_outside_s=edge,
        last_table_load_step_end_s=last_table,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inputs", type=Path, required=True)
    cli.add_argument("--motion", type=Path, nargs="+", required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    spec = json.loads(args.inputs.read_text(encoding="utf8"))
    result = {p.name: audit(spec, json.loads(p.read_text(encoding="utf8"))) for p in args.motion}
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
    print(json.dumps(result, allow_nan=False))
