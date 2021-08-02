import unittest
from pyrayt.utils import wavelength_to_rgb
import numpy as np

class TestWaveToRGB(unittest.TestCase):
    def test_inflection_points(self):
        inflection_points = np.asarray([0.44, 0.49, 0.51,0.58, 0.645])
        rgb = wavelength_to_rgb(inflection_points, gamma=1.0)

        # the shape should be the same as wavelengths but 3-colors
        self.assertEqual(rgb.shape, (len(inflection_points), 3))

        expected_rgb = np.asarray([
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0]
        ])

        for expected, actual in zip(expected_rgb, rgb):
            self.assertTrue(np.allclose(expected, actual), f"expected {expected} but got {actual}")