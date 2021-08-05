import unittest
import numpy as np

import pyrayt
import tinygfx
import tinygfx.g3d.primitives as primitives

class TestThickLensAperture(unittest.TestCase):
    
    def test_balanced_concave_aperture(self):
        r1 = 1
        r2 = -1
        thickness = 1.0
        aperture = 1
        total_thickness, offset = pyrayt.components._lens_full_thickness(r1, r2, thickness, aperture)
        self.assertEqual(total_thickness, thickness)
        self.assertAlmostEqual(offset, 0.0)
    
    def test_balanced_convex_aperture(self):
        r1 = -100
        r2 = 100
        thickness = .1
        aperture = 1
        expected_thickness = thickness + 2*(aperture/2)**2/(8*r2)
        total_thickness, offset = pyrayt.components._lens_full_thickness(r1, r2, thickness, aperture)
        self.assertAlmostEqual(total_thickness, expected_thickness)
        self.assertAlmostEqual(offset, 0.0)
    
    def test_imbalanced_convex_aperture(self):
        r1 = -100
        r2 = 50
        thickness = .1
        aperture = 1
        expected_thickness = thickness + (aperture/2)**2/(8*np.abs(r1)) + (aperture/2)**2/(8*np.abs(r2))
        expected_offset = (aperture/2)**2/(8*np.abs(r2)) - (aperture/2)**2/(8*np.abs(r1))
        total_thickness, offset = pyrayt.components._lens_full_thickness(r1, r2, thickness, aperture)
        self.assertAlmostEqual(total_thickness, expected_thickness)
        self.assertAlmostEqual(offset, expected_offset)

