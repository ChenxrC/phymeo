"""Serial local friction observations; not a contact graph or time-step benchmark."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from benchmark_surface_foundation import observe

from phymeo import solve_friction_disk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = [
        "src/phymeo/constraints/friction.py",
        "scripts/benchmark_friction_disk.py",
        "scripts/benchmark_surface_foundation.py",
    ]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": hashes,
        "scope": "host FP64 local disk + independent KKT/energy check",
        "not_measured": ["full step", "cold process", "peak memory", "Warp"],
        "cases": {},
    }
    for name, velocity, expected in (
        ("sticking", np.array([0.2, -0.4]), np.array([-0.2, 0.1])),
        ("sliding", np.array([1.8, 4.8]), np.array([-0.6, -0.8])),
    ):
        k = np.diag([1.0, 4.0])
        iterations = []

        def operation():
            value = solve_friction_disk(k, velocity, 2.0, mu_static=0.6, mu_dynamic=0.5)
            np.testing.assert_allclose(value.impulse, expected, atol=2e-10, rtol=2e-10)
            eta = value.lagrange_multiplier
            np.testing.assert_allclose(
                k @ value.impulse + velocity + eta * value.impulse, 0, atol=1e-10
            )
            if value.energy_change > 0 or value.regime != name:
                raise RuntimeError("friction physical validation failed")
            iterations.append(value.iterations)

        measurement = observe(operation, 100, 3)
        measurement["iterations_min_max"] = [min(iterations), max(iterations)]
        result["cases"][name] = measurement
    for name, digest in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("source changed during observation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved serial friction observations to {args.output}")


if __name__ == "__main__":
    main()
