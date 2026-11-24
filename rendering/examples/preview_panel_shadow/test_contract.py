"""Coverage of the emitted quadrature and compact/full shader selection."""
from math import cos, floor, pi, sin
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from preview_panel_shadow.build_source import apply, STUDIO


def points(samples, batch=0):
    # Inspect the actual GLSL permutation rather than a second copy of it.
    text = STUDIO.read_text(encoding='utf-8')
    pattern = rf'yRank\[{samples}\]=int\[{samples}\]\(([^)]+)\)'
    rows = [int(rank) for rank in re.search(pattern, text).group(1).split(',')]
    result = []
    for x, y in enumerate(rows):
        point = []
        for rank, divisions, step in ((x, 4, 0.754877666),
                                     (y, samples // 4, 0.569840291)):
            position = (rank + 0.5) * divisions / samples
            cell = floor(position)
            point.append((cell + ((position - cell + batch * step) % 1)) / divisions)
        result.append(tuple(point))
    return result


def cdf(cut, a, b):
    """Exact fraction of the unit panel where a*x+b*y <= cut."""
    cut -= min(a, 0.0) + min(b, 0.0)
    a, b = abs(a), abs(b)
    if min(a, b) < 1e-10:
        return max(0.0, min(1.0, cut / max(a, b)))
    return (max(cut, 0)**2 - max(cut - a, 0)**2 - max(cut - b, 0)**2
            + max(cut - a - b, 0)**2) / (2 * a * b)


def half_plane_error(pattern, angle):
    a, b = cos(angle), sin(angle)
    projected = sorted(a*x + b*y for x, y in pattern)
    count = len(projected)
    return max(abs(cdf(cut, a, b) - reached / count)
               for i, cut in enumerate(projected) for reached in (i, i+1))


class PanelShadows(unittest.TestCase):
    def test_motion_samples_stratify_both_dimensions_and_each_coarse_cell(self):
        for samples in (8, 16):
            pattern = points(samples)
            for axis in (0, 1):
                self.assertEqual(sorted(int(p[axis] * samples) for p in pattern),
                                 list(range(samples)))
            cells = {(int(x*4), int(y*(samples//4))) for x, y in pattern}
            self.assertEqual(len(cells), samples)

    def test_straight_edge_area_matches_continuous_panel_more_closely(self):
        baseline = [(x, y) for x in (0.25, 0.75) for y in (0.25, 0.75)]
        self.assertEqual(half_plane_error(baseline, 0), 0.25)
        for samples, tolerance in ((8, 0.18), (16, 0.13)):
            worst = max(half_plane_error(points(samples), i*pi/32) for i in range(32))
            self.assertLess(worst, tolerance)

    def test_progressive_batches_cover_each_cell_and_replay_initial_pattern(self):
        for samples in (8, 16):
            initial = points(samples)
            later = points(samples, 1)
            self.assertNotEqual(initial, later)
            self.assertEqual(initial, points(samples, 0))
            for batch in (1, 15, 63, 4095):
                pattern = points(samples, batch)
                self.assertTrue(all(0 <= v < 1 for p in pattern for v in p))
                self.assertEqual([(int(x*4), int(y*(samples//4))) for x, y in initial],
                                 [(int(x*4), int(y*(samples//4))) for x, y in pattern])

    def test_build_selects_primary_draw_only_and_rejects_unknown_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root / '_shaders' / STUDIO.name
            dest.parent.mkdir()
            dest.write_bytes(STUDIO.read_bytes())
            for bad in (True, '16', 4, 32):
                with self.assertRaises(ValueError):
                    apply(root, bad)
            info = apply(root, 8)
            self.assertEqual(info['marker_id'], 107)
            self.assertEqual(info['primary_samples_per_panel'], 8)
            self.assertEqual(info['secondary_samples_per_panel'], 4)
            text = dest.read_text(encoding='utf-8')
            self.assertIn('#define PR_PRIMARY_PANEL_SEQUENCE 8', text)
            self.assertIn('defined(PR_PREVIEW_SPARSE_MAIN)', text.splitlines()[0])
            self.assertIn('!defined(PR_PREVIEW_TILE)', text.splitlines()[0])
            self.assertIn('&& !defined(PR_PREVIEW_SPARSE_MAIN)', text)
            self.assertIn('occluded(biasP,L,dist)', text)
            self.assertIn('float(studioAreaSamples)*pdfArea', text)
            with self.assertRaisesRegex(ValueError, 'unknown or already configured'):
                apply(root)


if __name__ == '__main__':
    unittest.main()
