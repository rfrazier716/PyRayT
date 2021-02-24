import unittest
import pyrayt.simple_cg as cg
import pyrayt.surfaces as surf
import pyrayt.shaders.analytic as mat

import numpy as np


class TestRefractiveShader(unittest.TestCase):
    def setUp(self):
        self.cube = surf.Cuboid(material=mat.NKShader(mat.Material.REFRACTIVE, n=1.5))
        self.cube.invert_normals()
        self.ray = cg.Ray(origin=cg.Point(1, 0, 0), direction=cg.Vector(1, 0, 0))

    def test_refraction_at_normal_incidence(self):
        rays, index = self.cube.shade(self.ray, 0.63, 1.0)

        # the ray should not change direction
        self.assertTrue(np.allclose(rays, self.ray))

        # the index should now be 1.5 since it has entered the surface
        self.assertAlmostEqual(index, 1.5)

    def test_nonnormal_refraction(self):
        ray = cg.Ray(origin=cg.Point(1, 0, 0), direction=cg.Vector(1, 0, 1).normalize())
        rays, index = self.cube.shade(ray, 0.63, 1.0)

        # the ray should enter a higher index material and tilt towards the z-axis
        theta_2 = np.arcsin(np.sin(45 * np.pi / 180) / 1.5)
        expected_dir = cg.Vector(np.cos(theta_2), 0, np.sin(theta_2))
        actual_dir = rays[1]
        # the ray should not change direction
        self.assertTrue(np.allclose(actual_dir, expected_dir), f"expected {expected_dir} but got {actual_dir}")

        # the index should now be 1.5 since it has entered the surface
        self.assertAlmostEqual(index, 1.5)

    def test_arrayed_refraction(self):
        n_rays = 1000
        rays = cg.bundle_of_rays(n_rays)
        rays[0, 0] = 1
        rays[1, 0] = 1
        new_rays, index = self.cube.shade(rays, np.full(n_rays, 0.63), 1.0)

        # the returned index should be the length of the number of rays propagated
        self.assertTrue(index.shape, rays.shape[-1])

        # the ray should not change direction
        self.assertTrue(np.allclose(new_rays, rays))

        # the index should now be 1.5 since it has entered the surface
        self.assertTrue(np.allclose(index, 1.5))


if __name__ == '__main__':
    unittest.main()
