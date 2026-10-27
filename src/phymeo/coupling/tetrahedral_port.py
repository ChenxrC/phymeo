"""Lumped solid node response using the common impulse-only point protocol."""

from dataclasses import replace

import numpy as np

from ..core.scene import BackendContext, _name
from ..core.state import StepInputs
from ..core.tetrahedral import TetrahedralModel
from ..runtime.tetrahedral_family import tetrahedral_family
from .ports import ParticlePointPort


class TetrahedralPointPort(ParticlePointPort):
    """Same nodal response algebra, with a distinct solid model/state boundary."""

    def __init__(self, model, state, indices, *, domain="soft"):
        if not isinstance(model, TetrahedralModel):
            raise TypeError("solid point port requires TetrahedralModel")
        self._model = replace(model)
        self._domain = _name(domain, "port domain")
        self._family, self._context = tetrahedral_family(), BackendContext()
        self._state, _ = self._family.snapshot(
            self._model, state, StepInputs.zeros(model.particle_count), self._context
        )
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise TypeError("solid port indices must be rank-one integers")
        if np.any(indices < 0) or np.any(indices >= model.particle_count):
            raise ValueError("solid port index outside model")
        self._indices = np.array(indices, dtype=np.int64, copy=True)
        self._indices.flags.writeable = False
