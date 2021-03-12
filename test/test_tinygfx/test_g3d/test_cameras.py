import unittest

import numpy as np

import tinygfx.g3d

if __name__ == '__main__':
    unittest.main()


class TestOrthographicCamera(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = tinygfx.g3d.world_objects.OrthoGraphicCamera(10, 1, 0.5)

    def test_number_of_rays_created(self):
        rays = self.camera.generate_rays()
        self.assertEqual(rays.n_rays, 50)

    def test_ray_direction(self):
        # by default the rays face the positive x-axis
        ray_set = self.camera.generate_rays()
        self.assertTrue(np.allclose(ray_set.rays[1].T, np.array((1, 0, 0, 0))))

        # but rotating the camera can change that direction
        ray_set = self.camera.rotate_y(90).generate_rays()
        self.assertTrue(np.allclose(ray_set.rays[1].T, np.array((0, 0, -1.0, 0))))

    def test_ray_position(self):
        ray_set = self.camera.generate_rays()
        self.assertTrue(np.allclose(ray_set.rays[0, 0], 0))

        x_spans = ray_set.rays[0, 1].reshape(5, 10)
        self.assertTrue(np.allclose(x_spans, np.linspace(-0.5, 0.5, 10)))

        y_spans = ray_set.rays[0,2].reshape(5,10).T
        self.assertTrue(np.allclose(y_spans, np.linspace(0.25, -0.25, 5)))
        if __name__ == '__main__':
            unittest.main()