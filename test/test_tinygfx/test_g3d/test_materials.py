import tinygfx.g3d.materials as matl
import tinygfx.g3d as cg
import numpy as np
import unittest

import tinygfx.g3d.materials.color


class TestGoochMaterial(unittest.TestCase):

    def setUp(self) -> None:
        self.material = matl.gooch.GoochMaterial(base_color=tinygfx.g3d.materials.color.WHITE,
                                                 warm_color=tinygfx.g3d.materials.color.YELLOW,
                                                 cool_color=tinygfx.g3d.materials.color.BLUE,
                                                 alpha=0,
                                                 beta=0)
        self.light_vector = cg.Point(0, 0, 10)
        self.normals = np.zeros((4, 10))
        self.normals[2] = 1
        self.rays = cg.bundle_of_rays(10)

    def test_single_light_source(self):
        # split the normal vector so half face away and half face towards
        self.normals[2, :5] = -1
        pixel_values = self.material.shade(self.rays, self.normals, self.light_vector)

        expected_color = tinygfx.g3d.materials.color.BLUE
        self.assertTrue(np.allclose(pixel_values[:, :5], np.atleast_2d(expected_color).T), pixel_values)

        expected_color = tinygfx.g3d.materials.color.YELLOW
        self.assertTrue(np.allclose(pixel_values[:, 5:], np.atleast_2d(expected_color).T))

    def test_single_ray_case(self):
        ray = cg.Ray(cg.Point(0, 0, 0), cg.Vector(1, 0, 0))
        normal = cg.Vector(0, 0, 1)
        pixel_values = self.material.shade(ray, normal, self.light_vector)

        self.assertEqual(pixel_values.shape, (4, 1))
        expected_color = tinygfx.g3d.materials.color.YELLOW
        self.assertTrue(np.allclose(pixel_values, np.atleast_2d(expected_color).T))

        # if we flip the normals the color shoudl be the opposite extreme
        normal = cg.Vector(0, 0, -1)
        pixel_values = self.material.shade(ray, normal, self.light_vector)

        self.assertEqual(pixel_values.shape, (4, 1))
        expected_color = tinygfx.g3d.materials.color.BLUE
        self.assertTrue(np.allclose(pixel_values, np.atleast_2d(expected_color).T))

        # if the normal is 90 degrees relative to the source we should get a mixture
        normal = cg.Vector(0, 1, 0)
        pixel_values = self.material.shade(ray, normal, self.light_vector)

        self.assertEqual(pixel_values.shape, (4, 1))
        expected_color = tinygfx.g3d.materials.color.RGBAColor(0.5, 0.5, 0.5)
        self.assertTrue(np.allclose(pixel_values, np.atleast_2d(expected_color).T), f"got {pixel_values}")


if __name__ == '__main__':
    unittest.main()
