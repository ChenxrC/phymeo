"""Workspace-owned fixed-size terrain masks for TYX interpret and CUDA."""

import numpy as np


class HeightfieldCandidateBuffer:
    def __init__(self, engine, offsets):
        self.engine = engine
        self.offsets = tuple(map(int, offsets))
        # A grid has strictly fewer cells than registered vertices. The largest
        # single shape is enough for any terrain in this immutable workspace.
        self.capacity = max((b - a for a, b in zip(self.offsets, self.offsets[1:])), default=0)
        self.output = None
        self.counts = dict(
            heightfield_launches=0,
            heightfield_completed_launches=0,
            heightfield_cells_tested=0,
            heightfield_faces_kept=0,
            heightfield_downloads=0,
            heightfield_downloaded_bytes=0,
            heightfield_synchronizations=0,
        )

    def _read(self):
        if self.engine._mode != "cuda":
            return self.output
        self.engine._runtime.sync()
        self.counts["heightfield_synchronizations"] += 1
        self.engine._counts["synchronizations"] += 1
        result = self.engine._read(self.output)
        self.counts["heightfield_downloads"] += 1
        self.counts["heightfield_downloaded_bytes"] += result.nbytes
        return result

    def select(self, terrain, sphere, rows, columns):
        from ._tyx_heightfield_kernels import heightfield_masks

        n = len(self.offsets) - 1
        if not (0 <= terrain < n and 0 <= sphere < n and terrain != sphere):
            raise RuntimeError("invalid heightfield candidate owners")
        cells = (rows - 1) * (columns - 1)
        if (
            rows < 2
            or columns < 2
            or rows * columns != self.offsets[terrain + 1] - self.offsets[terrain]
            or self.offsets[sphere + 1] - self.offsets[sphere] != 1
        ):
            raise RuntimeError("invalid heightfield candidate layout")
        if cells > self.capacity:
            raise RuntimeError("heightfield candidate capacity exceeded")
        generation = self.counts["heightfield_launches"] + 1
        if generation > np.iinfo(np.int32).max:
            raise RuntimeError("heightfield candidate generation exhausted")
        e = self.engine
        with e._tx.use_runtime(e._runtime):
            if self.output is None:
                # Generation zero cannot be mistaken for a completed first run.
                self.output = e._static(np.zeros((self.capacity, 2), dtype=np.int32))
            self.counts["heightfield_launches"] += 1
            e._counts["launches"] += 1
            heightfield_masks(
                cells,
                columns,
                self.offsets[terrain],
                int(sphere),
                generation,
                e._envelopes,
                e._bounds,
                self.output,
                mode=e._mode,
            )
            raw = self._read()
        if (
            raw.shape != (self.capacity, 2)
            or raw.dtype != np.int32
            or not np.all(raw[:cells, 0] == generation)
        ):
            raise RuntimeError("incomplete heightfield candidate output")
        masks = raw[:cells, 1]
        if np.any((masks < 0) | (masks > 3)):
            raise RuntimeError("invalid heightfield candidate mask")
        faces = tuple(
            2 * int(cell) + face
            for cell in np.flatnonzero(masks)
            for face in (0, 1)
            if int(masks[cell]) & (1 << face)
        )
        self.counts["heightfield_completed_launches"] += 1
        self.counts["heightfield_cells_tested"] += cells
        self.counts["heightfield_faces_kept"] += len(faces)
        e._counts["completed_launches"] += 1
        return faces
