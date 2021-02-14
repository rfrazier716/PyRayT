import unittest
import adpd.packaging.analytic_surface as asurf
import numpy as np
import adpd.packaging.simple_cg as cg


class TestSphere(unittest.TestCase):
    def setUp(self) -> None:
        self.radius = 3
        self.sphere = asurf.Sphere(self.radius)
        self.ray = cg.Ray(direction=cg.Vector(1, 0, 0))

    def test_intersection_scaled_sphere(self):
        # if the sphere is scaled, the intersection should grow with the scaling
        scale_factor = 10
        self.sphere.scale_all(scale_factor)
        hit = self.sphere.intersect(self.ray)
        self.assertAlmostEqual(hit[0], scale_factor*self.radius)

    def test_intersection_translated_sphere(self):
        movement = 10
        self.sphere.move_x(movement)
        hit = self.sphere.intersect(self.ray)
        self.assertAlmostEqual(hit[0], movement - self.radius)

    # def test_normals_scaled_sphere(self):
    #     # scaling a sphere should have no effect on the normals
    #     scaling = 5
    #     self.sphere.scale_all(scaling)
    #     scaled_intersection_points = ((0, 0, -5), (0, 0, 5), (0, 5, 0), (0, -5, 0), (5, 0, 0), (-5, 0, 0))
    #     self.intersections = [cg.Point(*intersection) for intersection in scaled_intersection_points]
    #     # for a nontransformed sphere the normals should be vectors of the coordinates
    #     normals = [self.sphere.normal(intersection) for intersection in self.intersections]
    #     for normal, intersection in zip(normals, self.intersection_points):
    #         expected = cg.Vector(*intersection)
    #         self.assertTrue(np.allclose(normal, expected))
    #         self.assertAlmostEqual(np.linalg.norm(normal), 1.0)
    #
    #     # assert that the operation did not overwrite the world transform matrix
    #     self.assertTrue(np.allclose(self.sphere.get_world_transform()[:-1, :-1], np.identity(3) * scaling))
    #
    # def test_normals_rotated_sphere(self):
    #     # rotation should give the same normals
    #     z_rotation = 45
    #     self.sphere.rotate_z(45)
    #
    #     normals = [self.sphere.normal(intersection) for intersection in self.intersections]
    #     for normal, intersection in zip(normals, self.intersection_points):
    #         expected = cg.Vector(*intersection)
    #         self.assertTrue(np.allclose(normal, expected), f"Expected {normal}, got {expected}")
    #         self.assertAlmostEqual(np.linalg.norm(normal), 1.0)
    #
    #     # assert that the operation did not overwrite the world transform matrix
    #
    # def test_normals_translated_sphere(self):
    #     translation = 10
    #     self.sphere.move_x(translation)
    #     translated_intersections = [intersection + np.array([translation, 0, 0, 0]) for intersection in
    #                                 self.intersections]
    #     normals = [self.sphere.normal(intersection) for intersection in translated_intersections]
    #     for normal, intersection in zip(normals, self.intersection_points):
    #         expected = cg.Vector(*intersection)
    #         self.assertTrue(np.allclose(normal, expected), f"Expected {expected}, got {normal}")
    #         self.assertAlmostEqual(np.linalg.norm(normal), 1.0)


if __name__ == '__main__':
    unittest.main()
