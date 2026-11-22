"""Verify the independently timed control clock used by a real-time Panda film."""

import unittest

import numpy as np

from phymeo.experiments.panda_cube import _motion_clock, _open_return


class FilmControlClockTests(unittest.TestCase):
    def setUp(self):
        self.original = np.array([0., 0.32, 0.55, 0.92, 1.35, 1.72, 1.95, 2.25])

    def test_default_preserves_original_controls_and_carry_window(self):
        knots, interval = _motion_clock(self.original, 2.25, 1600)
        np.testing.assert_array_equal(knots, self.original)
        self.assertEqual(interval, (0.95, 1.35))

    def test_ten_second_command_schedule_and_physical_step_count(self):
        knots, interval = _motion_clock(self.original, 10.0, 1600)
        np.testing.assert_allclose(knots / 10.0, self.original / 2.25)
        self.assertEqual(knots[-1] * 1600, 16000)
        self.assertAlmostEqual(interval[0], 0.95 * 10 / 2.25)
        self.assertEqual(interval[1], knots[4])
        self.assertEqual(self.original[-1], 2.25)

    def test_unrepresentable_duration_is_rejected_before_stepping(self):
        for duration in (0, -1, True, float("nan"), 0.0001, 10.000219):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                _motion_clock(self.original, duration, 1600)

    def test_open_return_keeps_the_loaded_motion_and_uses_the_open_home_target(self):
        knots = np.arange(8 * 9, dtype=float).reshape(8, 9)
        times, command = _open_return(self.original, knots, 10, 1600)
        np.testing.assert_array_equal(times[:-1], self.original)
        np.testing.assert_array_equal(command[:-1], knots)
        np.testing.assert_array_equal(command[-1], knots[0])
        self.assertEqual(times[-1], 10.0)
        for invalid in (float("nan"), True, 2.20, 2.25, 10.000219):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _open_return(self.original, knots, invalid, 1600)
