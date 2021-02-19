import unittest
import numpy as np
import pyrayt.components.sources as sources
import pyrayt.simple_cg as cg


class TestLineOfRays(unittest.TestCase):
    def setUp(self) -> None:
        self.source = sources.LineOfRays(1)

    def test_single_ray_generation(self):
        rays = self.source.generate_rays(1)
        self.assertEqual(rays.shape, (2, 4, 1))

        expected_pos = np.atleast_2d(cg.Point())
        self.assertTrue(np.allclose(rays[0], expected_pos.T))

        expected_dir = np.atleast_2d(cg.Vector(1, 0, 0))
        self.assertTrue(np.allclose(rays[1], expected_dir.T))

    def test_multi_ray_generation(self):
        n_rays = 3
        rays = self.source.generate_rays(n_rays)
        self.assertEqual(rays.shape, (2, 4, n_rays))

        expected_pos = np.array((-0.5, 0, 0.5))
        self.assertTrue(np.allclose(rays[0, 1], expected_pos))

        self.assertTrue(np.allclose(rays[1, 0], 1))
        self.assertTrue(np.allclose(rays[1, 1], 0))
        self.assertTrue(np.allclose(rays[1, 2], 0))

    def test_transformed_ray_generation(self):
        # rotate and generate a single ray
        self.source.rotate_y(-90)
        rays = self.source.generate_rays(1)

        expected_dir = np.atleast_2d(cg.Vector(0, 0, 1))
        self.assertTrue(np.allclose(rays[1], expected_dir.T))

    def test_translated_ray_generation(self):
        # rotate and generate a single ray
        self.source.move_z(5)
        rays = self.source.generate_rays(1)

        expected_pos = np.atleast_2d(cg.Point(0, 0, 5))
        self.assertTrue(np.allclose(rays[0], expected_pos.T))

        expected_dir = np.atleast_2d(cg.Vector(1, 0, 0))
        self.assertTrue(np.allclose(rays[1], expected_dir.T))


if __name__ == '__main__':
    unittest.main()
