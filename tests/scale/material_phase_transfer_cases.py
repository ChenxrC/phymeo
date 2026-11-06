"""Frozen inputs for the material-page measurement; no new physical solver."""

from dataclasses import dataclass, replace

H = 1 / 256
STEPS = 4
WORLD_KEYS = (503, 91) + tuple(1009 + 17 * i for i in range(30))
MATRIX = ((1, 2, "uniform"), (1, 8, "uniform"), (32, 2, "uniform"),
          (32, 8, "uniform"), (3, 8, "heterogeneous"))


@dataclass(frozen=True)
class WorldInput:
    key: int
    physical_index: int
    faces: int
    cloth_shift_z_m: float = 0.0


def frozen_layout(worlds, faces, case="uniform"):
    """Declare order and geometry before obtaining any device result."""
    if type(worlds) is not int or type(faces) is not int:
        raise ValueError("world and face counts must be integers")
    if case == "heterogeneous":
        if (worlds, faces) != (3, 8):
            raise ValueError("heterogeneous case requires worlds=3, faces=8")
        return (WorldInput(503, 0, 2), WorldInput(91, 0, 8), WorldInput(1009, 0, 8, -0.5))
    if case != "uniform" or worlds not in (1, 32) or faces not in (2, 8):
        raise ValueError("expected a frozen uniform or heterogeneous case")
    return tuple(WorldInput(WORLD_KEYS[i], i % 4, faces) for i in range(worlds))


def build_worlds(worlds, faces, case="uniform", *, operator=None):
    """Reuse the accepted freely evolving strip and finite-mass shaft inputs."""
    layout = frozen_layout(worlds, faces, case)
    import phymeo as ph
    from multiface_load_cases import make_world

    result = {}
    for spec in layout:
        world = make_world(spec.physical_index, spec.faces, operator)
        if spec.cloth_shift_z_m:
            state = world.initial
            cloth = state.domains["cloth"]
            world = replace(world, initial=ph.SceneState({
                **state.domains,
                "cloth": ph.ParticleState(
                    cloth.position + [0.0, 0.0, spec.cloth_shift_z_m], cloth.velocity
                ),
            }))
        result[spec.key] = world
    return result


# Exact dyadic rows: normal, relative velocity, width, gap, k, fraction, damping,
# friction, slip speed. Expected forces below are independent scalar calculations.
_LAW_ROWS = (
    (0, 0, 1, .25, 0, -.5, .125, .09375, 64, .5, .5, .25, .125),
    (1, 0, 0, .25, 0, 0, .125, .09375, 128, .25, 0, 0, .125),
    (0, 1, 0, 0, 1, 1, .125, .1875, 32, 1, 0, .5, .25),
    (0, 0, 1, 0, 0, -.25, .125, .09375, 64, 0, .5, .25, .125),
    (0, 1, 0, .0625, 0, .0625, .125, .09375, 128, .5, 0, .5, .25),
    (0, 0, 1, .125, 0, 4, .125, .09375, 64, 1, .5, .25, .125),
)
_LAW_RESULTS = (
    (-.5, 1.25, -.3125, 0, 1.25),
    (.25, 1, 1, 0, 0),
    (1, 0, 0, 0, 0),
    (-.25, 0, 0, 0, 0),
    (0, 2, -.25, 2, -.25),
    (4, 0, 0, 0, 0),
)


def law_protocol_rows(count):
    """Arithmetic protocol cases are explicitly not physical trajectories."""
    if type(count) is not int or count not in (0, 127, 128, 129):
        raise ValueError("law protocol count must be 0, 127, 128 or 129")
    import numpy as np

    rows = np.array([_LAW_ROWS[i % 6] for i in range(count)], dtype=np.float64).reshape(-1, 13)
    expected = np.array([_LAW_RESULTS[i % 6] for i in range(count)], dtype=np.float64).reshape(-1, 5)
    return rows, expected
