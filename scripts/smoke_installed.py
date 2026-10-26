"""Run with an isolated, non-editable wheel environment: python -I this_file.py."""

import sys
from pathlib import Path

import numpy as np

import phymeo


def main():
    installed = Path(phymeo.__file__).resolve()
    if not installed.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Expected a wheel inside {sys.prefix}, imported {installed}")
    model = phymeo.ParticleModel(np.array([0.0, 1.0]))
    state = phymeo.ParticleState(np.zeros((2, 3)), np.zeros((2, 3)))
    inputs = phymeo.StepInputs.zeros(model.particle_count)
    if state.particle_count != 2 or inputs.force.shape != (2, 3):
        raise RuntimeError("Installed data contract failed")
    print(f"Installed-wheel smoke passed: {installed}")


if __name__ == "__main__":
    main()
