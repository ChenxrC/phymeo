"""Persist the geometry, forces and actual states from one run, without replaying it."""

import json
from dataclasses import asdict

import numpy as np


def save_history(path, family, frames, stats):
    def json_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj))

    rows = [asdict(frame) for frame in frames]
    path.write_text(
        json.dumps(
            dict(
                family=family,
                dt_s=1 / 64,
                dtype=str(frames[0].state.particles.position.dtype),
                stats=stats,
                steps=rows,
            ),
            allow_nan=False,
            default=json_default,
        ),
        encoding="utf8",
    )
