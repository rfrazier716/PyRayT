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
    
    def test_balanced_concave_aperture(self):
        r1 = -100
        r2 = 100
        thickness = .1
        aperture = 1
        expected_thickness = thickness + (aperture/2)**2/(r2)
        total_thickness, offset = pyrayt.components._lens_full_thickness(r1, r2, thickness, aperture)
        self.assertAlmostEqual(total_thickness, expected_thickness)
        self.assertAlmostEqual(offset, 0.0)
    
    def test_imbalanced_concave_aperture(self):
        r1 = -100
        r2 = 50
        thickness = .1
        aperture = 1
        expected_thickness = thickness + 0.5*(aperture/2)**2/(np.abs(r1)) + 0.5*(aperture/2)**2/(np.abs(r2))
        expected_offset = 0.5*(aperture/2)**2/(np.abs(r2)) - 0.5*(aperture/2)**2/(np.abs(r1))
        total_thickness, offset = pyrayt.components._lens_full_thickness(r1, r2, thickness, aperture)
        self.assertTrue(np.isclose(total_thickness, expected_thickness, rtol=1E-3))
        self.assertTrue(np.isclose(offset, expected_offset, rtol=1E-3))

