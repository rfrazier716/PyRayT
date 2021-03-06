import unittest
import numpy as np

import pyrayt.components
import tinygfx.g3d.primitives as primitives


class TestSource(unittest.TestCase):
    # using a line of rays as a proxy source for wrapper methods
    def setUp(self) -> None:
        self.source = pyrayt.components.LineOfRays(1)

    def test_transformed_ray_generation(self):
        # rotate and generate a single ray
        self.source.rotate_y(-90)
        rays = self.source.generate_rays(1)

        expected_dir = np.atleast_2d(primitives.Vector(0, 0, 1))
        self.assertTrue(np.allclose(rays.rays[1], expected_dir.T))

    def test_translated_ray_generation(self):
        # rotate and generate a single ray
        self.source.move_z(5)
        rays = self.source.generate_rays(1)

        expected_pos = np.atleast_2d(primitives.Point(0, 0, 5))
        self.assertTrue(np.allclose(rays.rays[0], expected_pos.T))

        expected_dir = np.atleast_2d(primitives.Vector(1, 0, 0))
        self.assertTrue(np.allclose(rays.rays[1], expected_dir.T))


class TestLineOfRays(unittest.TestCase):
    def setUp(self) -> None:
        self.source = pyrayt.components.LineOfRays(1)

    def test_single_ray_generation(self):
        # t_rays is shorthand for tracer rays
        t_rays = self.source.generate_rays(1)
        self.assertEqual(t_rays.rays.shape, (2, 4, 1))

        expected_pos = np.atleast_2d(primitives.Point())
        self.assertTrue(np.allclose(t_rays.rays[0], expected_pos.T))

        expected_dir = np.atleast_2d(primitives.Vector(1, 0, 0))
        self.assertTrue(np.allclose(t_rays.rays[1], expected_dir.T))

    def test_multi_ray_generation(self):
        n_rays = 3
        t_rays = self.source.generate_rays(n_rays)
        self.assertEqual(t_rays.rays.shape, (2, 4, n_rays))

        expected_pos = np.array((-0.5, 0, 0.5))
        self.assertTrue(np.allclose(t_rays.rays[0, 1], expected_pos))

        self.assertTrue(np.allclose(t_rays.rays[1, 0], 1))
        self.assertTrue(np.allclose(t_rays.rays[1, 1], 0))
        self.assertTrue(np.allclose(t_rays.rays[1, 2], 0))

    def test_wavelength_applied(self):
        # t_rays is shorthand for tracer rays
        self.source.wavelength = 1.5
        t_rays = self.source.generate_rays(1)
        self.assertEqual(t_rays.wavelength[0], 1.5)


if __name__ == "__main__":
    unittest.main()