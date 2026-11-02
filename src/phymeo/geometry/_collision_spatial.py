"""Deterministic sweep order and lossless publication of spatial pair candidates."""

import numpy as np

from ..core.capabilities import SolverFailure

WORD_BITS = 30


class SpatialPairs:
    def __init__(self, tags, pairs):
        self.tags = tags
        self.count = len(tags)
        self.words = (len(tags) + WORD_BITS - 1) // WORD_BITS
        self.defaults = np.full(len(pairs), 5, dtype=np.int32)
        a, b, excluded = pairs.T
        self.defaults[excluded != 0] = 4
        self.defaults[((tags[a, 3] & tags[b, 4]) == 0) | ((tags[b, 3] & tags[a, 4]) == 0)] = 3
        self.defaults[(tags[a, 2] != 0) & (tags[a, 1] == tags[b, 1])] = 2
        self.defaults[tags[a, 0] != tags[b, 0]] = 1
        eligible = self.defaults == 5
        a, b = a[eligible], b[eligible]
        bits = 1 << (np.arange(len(tags), dtype=np.int32) % WORD_BITS)
        self.meta = np.column_stack((tags[:, 0], tags[:, 5], bits)).astype(np.int32)
        self.allowed = np.zeros((len(tags), self.words), dtype=np.int32)
        for u, v in ((a, b), (b, a)):
            np.bitwise_or.at(self.allowed, (u, v // WORD_BITS), bits[v])
        self.worlds = tuple(np.flatnonzero(tags[:, 0] == world) for world in np.unique(tags[:, 0]))
        self.window_pairs = 0
        self.box_tests = 0

    def order(self, bounds):
        """Choose the axis with the fewest closed interval overlaps in each world."""
        order = np.empty((len(bounds), 2), dtype=np.int32)
        pos = 0
        for world in self.worlds:
            plane = self.tags[world, 5] != 0
            finite = world[~plane]
            best = None
            for axis in range(3):
                sorted_ids = finite[np.argsort(bounds[finite, axis], kind="stable")]
                starts = bounds[sorted_ids, axis]
                after = np.searchsorted(starts, bounds[sorted_ids, axis + 3], side="right")
                projected = int(after.sum()) - len(finite) * (len(finite) + 1) // 2
                if best is None or projected < best[0]:
                    best = projected, axis, sorted_ids
            _, axis, ids = best
            n = len(world)
            order[pos : pos + n, 0] = np.concatenate((world[plane], ids))
            order[pos : pos + n, 1] = axis
            pos += n
        return order

    def host_sap(self, bounds, order):
        out = np.zeros((self.count, self.words + 2), np.int32)
        for pos, (src, axis) in enumerate(order):
            src, axis = int(src), int(axis)
            world, plane, _ = self.meta[src]
            for dst in order[pos + 1 :, 0]:
                if self.meta[dst, 0] != world or (
                    not plane and bounds[dst, axis] > bounds[src, axis + 3]
                ):
                    break
                out[src, -2] += 1
                mask = self.meta[dst, 2]
                word = dst // WORD_BITS
                if not self.allowed[src, word] & mask:
                    continue
                if not plane and not self.meta[dst, 1]:
                    out[src, -1] += 1
                    if np.any(bounds[src, :3] > bounds[dst, 3:]) or np.any(
                        bounds[dst, :3] > bounds[src, 3:]
                    ):
                        continue
                out[src, word] |= mask
        return out

    def flags(self, raw):
        if raw.shape != (self.count, self.words + 2) or raw.dtype != np.int32 or np.any(raw < 0):
            raise SolverFailure("invalid packed spatial output")
        flags = self.defaults.copy()
        for i, row in enumerate(raw):
            for w in np.flatnonzero(row[: self.words]):
                word = int(row[w])
                if word & ~int(self.allowed[i, w]):
                    raise SolverFailure("invalid spatial candidate bit")
                while word:
                    bit = word & -word
                    j = int(w) * WORD_BITS + bit.bit_length() - 1
                    a, b = sorted((i, j))
                    slot = a * (2 * self.count - a - 1) // 2 + b - a - 1
                    if flags[slot] != 5:
                        raise SolverFailure("duplicate spatial candidate")
                    flags[slot] = 0
                    word -= bit
        return flags

    def decode_sap(self, raw):
        flags = self.flags(raw)
        self.window_pairs += int(raw[:, -2].sum())
        self.box_tests += int(raw[:, -1].sum())
        return flags
