"""Independently load all full P6 cases, exact repeats and matched FP64 references."""

import argparse
import json
from pathlib import Path

import numpy as np

from phymeo.experiments.articulated_device import load_articulated_device_run


def compare(device, repeated, reference):
    for run in (device, repeated, reference):
        if any(not np.isfinite(value).all() for value in run.arrays.values()):
            raise ValueError("reference/repeat comparison requires finite arrays")
    if device.arrays.keys() != repeated.arrays.keys():
        raise ValueError("repeat schema differs")
    for name, value in device.arrays.items():
        if not np.array_equal(value, repeated.arrays[name]):
            raise ValueError("non-deterministic repeated array: " + name)
    position_error, velocity_error, q_error = 0.0, 0.0, 0.0
    for prefix in ("", "substep_"):
        q_error = max(
            q_error,
            float(
                np.max(
                    np.abs(device.arrays[prefix + "robot_q"] - reference.arrays[prefix + "robot_q"])
                )
            ),
        )
        velocity_error = max(
            velocity_error,
            float(
                np.max(
                    np.abs(device.arrays[prefix + "robot_v"] - reference.arrays[prefix + "robot_v"])
                )
            ),
        )
        for domain in device.config["node_domains"]:
            for field in ("position", "velocity"):
                name = prefix + domain + "_" + field
                error = float(
                    np.max(np.linalg.norm(device.arrays[name] - reference.arrays[name], axis=-1))
                )
                if field == "position":
                    position_error = max(position_error, error)
                else:
                    velocity_error = max(velocity_error, error)
    if q_error > 5e-4 or position_error > 1e-3 or velocity_error > 2e-2:
        raise ValueError(
            f"P6 reference gate failed: q={q_error}, nodes={position_error}, v={velocity_error}"
        )
    return dict(
        exact_repeat_arrays=len(device.arrays),
        max_q_error_m=q_error,
        max_node_position_error_m=position_error,
        max_velocity_error_m_s=velocity_error,
        contact_substeps=device.metrics["contact_substeps"],
    )


def verify(device_root, repeat_root, reference_root):
    roots = [Path(root).resolve() for root in (device_root, repeat_root, reference_root)]
    if len(set(roots)) != 3:
        raise ValueError("device/repeat/reference must be separate artifact roots")
    result = {}
    baseline_source = None
    for scene in ("S2", "S3", "S4"):
        runs = [
            load_articulated_device_run(
                root / scene, require_complete=True, expected_scene=scene, expected_mode=mode
            )
            for root, mode in zip(roots, ("cuda", "cuda", "reference"), strict=True)
        ]
        result[scene] = compare(*runs)
        result[scene]["artifacts"] = [str(root / scene) for root in roots]
        result[scene]["sources"] = [
            json.loads((root / scene / "run.json").read_text(encoding="utf-8"))["provenance"]
            for root in roots
        ]
        for source in result[scene]["sources"]:
            if baseline_source is None:
                baseline_source = source
            if any(
                source.get(key) != baseline_source.get(key)
                for key in ("source_sha256", "python", "numpy", "platform")
            ):
                raise ValueError("source or numerical environment differs across acceptance runs")
    return dict(
        status="complete",
        scenes=result,
        limits=dict(q_m=5e-4, node_position_m=1e-3, velocity_m_s=2e-2),
        scope="trusted-producer numeric/provenance audit; not a certificate against a malicious trace producer",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = verify(args.device_root, args.repeat_root, args.reference_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["scenes"], indent=2))


if __name__ == "__main__":
    main()
