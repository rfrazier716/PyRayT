import unittest
import pyrayt.surfaces as surf
import numpy as np
import pyrayt.simple_cg as cg


class TestSphere(unittest.TestCase):
    def setUp(self) -> None:
        self.radius = 3
        self.sphere = surf.Sphere(self.radius)
        self.ray = cg.Ray(direction=cg.Vector(1, 0, 0))

        self.intersection_points = ((0, 0, -1), (0, 0, 1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0))
        self.intersections = [cg.Point(*intersection) for intersection in self.intersection_points]

    def test_intersection_scaled_sphere(self):
        # if the sphere is scaled, the intersection should grow with the scaling
        scale_factor = 10
        self.sphere.scale_all(scale_factor)
        hit = self.sphere.intersect(self.ray)
        self.assertAlmostEqual(hit[0], scale_factor * self.radius)

    def test_intersection_translated_sphere(self):
        movement = 10
        self.sphere.move_x(movement)
        hit = self.sphere.intersect(self.ray)
        self.assertAlmostEqual(hit[0], movement - self.radius)

    def test_normals_scaled_sphere(self):
        # scaling a sphere should have no effect on the normals
        scaling = 5
        self.sphere.scale_all(scaling)
        scaled_intersection_points = ((0, 0, -5), (0, 0, 5), (0, 5, 0), (0, -5, 0), (5, 0, 0), (-5, 0, 0))
        self.intersections = [cg.Point(*intersection) for intersection in scaled_intersection_points]
        # for a nontransformed sphere the normals should be vectors of the coordinates
        normals = [self.sphere.get_world_normals(intersection) for intersection in self.intersections]
        for normal, intersection in zip(normals, scaled_intersection_points):
            expected = cg.Vector(*intersection) / scaling
            self.assertTrue(np.allclose(normal, expected))
            self.assertAlmostEqual(np.linalg.norm(normal), 1.0)

        # assert that the operation did not overwrite the world transform matrix
        self.assertTrue(np.allclose(self.sphere.get_world_transform()[:-1, :-1], np.identity(3) * scaling))

    def test_normals_rotated_sphere(self):
        # rotation should give the same normals
        z_rotation = 45
        self.sphere.rotate_z(45)

        normals = [self.sphere.get_world_normals(intersection) for intersection in self.intersections]
        for normal, intersection in zip(normals, self.intersection_points):
            expected = cg.Vector(*intersection)
            self.assertTrue(np.allclose(normal, expected), f"Expected {normal}, got {expected}")
            self.assertAlmostEqual(np.linalg.norm(normal), 1.0)

        # assert that the operation did not overwrite the world transform matrix

    def test_normals_translated_sphere(self):
        translation = 10
        self.sphere.move_x(translation)
        translated_intersections = [intersection + np.array([translation, 0, 0, 0]) for intersection in
                                    self.intersections]
        normals = [self.sphere.get_world_normals(intersection) for intersection in translated_intersections]
        for normal, intersection in zip(normals, self.intersection_points):
            expected = cg.Vector(*intersection)
            self.assertTrue(np.allclose(normal, expected), f"Expected {expected}, got {normal}")
            self.assertAlmostEqual(np.linalg.norm(normal), 1.0)

    def test_normal_inversion(self):
        cube = surf.Cuboid()
        point = cg.Point(1,0,0)
        normals = cube.get_world_normals(point)
        self.assertTrue(np.allclose(normals, cg.Vector(*point[:-1])), f"{normals}")

        cube.invert_normals()
        normals = cube.get_world_normals(point)
        self.assertTrue(np.allclose(normals, -cg.Vector(*point[:-1])), f"{normals}")

class TestCuboid(unittest.TestCase):
    def setUp(self) -> None:
        self.cube = surf.Cuboid()
        self.rays = (
            cg.Ray(direction=cg.Vector(-1, 0, 0)),
            cg.Ray(direction=cg.Vector(1, 0, 0)),
            cg.Ray(direction=cg.Vector(0, -1, 0)),
            cg.Ray(direction=cg.Vector(0, 1, 0)),
            cg.Ray(direction=cg.Vector(0, 0, -1)),
            cg.Ray(direction=cg.Vector(0, 0, 1))
        )

    def test_default_constructor(self):
        hit = np.asarray([self.cube.intersect(ray) for ray in self.rays])
        # all hits should be a distance of 1
        self.assertTrue(np.allclose(hit, 1))

    def test_from_length_constructor(self):
        lengths = (2, 3, 4)
        cube = surf.Cuboid.from_lengths(*lengths)
        expected_hits = np.array((2, 2, 3, 3, 4, 4)) / 2
        hits = np.asarray([cube.intersect(ray)[0] for ray in self.rays])

        self.assertTrue(np.allclose(hits, expected_hits), f"{hits}")

    def test_from_corner_constructor_tuple(self):
        corner0 = (-1, -2, -3)
        corner1 = (4, 5, 6)

        cube = surf.Cuboid.from_corners(corner0, corner1)
        hits = np.asarray([cube.intersect(ray)[0] for ray in self.rays])

        # the hits in the negative direction should be the same as the first corner
        self.assertTrue(np.allclose(hits[::2], np.abs(corner0)), f"{hits}")

        # the hits in the positive direction should be the same as the second corner
        self.assertTrue(np.allclose(hits[1::2], np.abs(corner1)), f"{hits}")

    def test_from_corner_constructor_point(self):
        corner0 = cg.Point(*(-1, -2, -3))
        corner1 = cg.Point(*(4, 5, 6))

        cube = surf.Cuboid.from_corners(corner0, corner1)
        hits = np.asarray([cube.intersect(ray)[0] for ray in self.rays])

        # the hits in the negative direction should be the same as the first corner
        self.assertTrue(np.allclose(hits[::2], np.abs(corner0[:-1])), f"{hits}")

        # the hits in the positive direction should be the same as the second corner
        self.assertTrue(np.allclose(hits[1::2], np.abs(corner1[:-1])), f"{hits}")

    def test_from_corner_constructor_error(self):
        with self.assertRaises(ValueError):
            corner0 = cg.Point(*(-1, -2, -3))
            corner1 = cg.Point(*(4, 5, 6))
            cube = surf.Cuboid.from_corners(corner1, corner0)

    def test_scaling(self):
        scaling = 5
        self.cube.scale_all(scaling)
        hit = np.asarray([self.cube.intersect(ray) for ray in self.rays])
        # all hits should be a distance of 1
        self.assertTrue(np.allclose(hit, scaling))

    def test_translation(self):
        movement = 0.5
        self.cube.move(movement, movement, movement)
        hits = np.asarray([self.cube.intersect(ray) for ray in self.rays])

        # all hits in the negative directions should be 0.5
        self.assertTrue(np.allclose(hits[::2], movement), f"{hits}")

        # all hits in the negative directions should be 1.5
        self.assertTrue(np.allclose(hits[1::2], 1 + movement), f"{hits}")

    def test_normal_inversion(self):
        cube = surf.Cuboid()
        point = cg.Point(1,0,0)
        normals = cube.get_world_normals(point)
        self.assertTrue(np.allclose(normals, cg.Vector(*point[:-1])), f"{normals}")

        cube.invert_normals()
        normals = cube.get_world_normals(point)
        self.assertTrue(np.allclose(normals, -cg.Vector(*point[:-1])), f"{normals}")



if __name__ == '__main__':
    unittest.main()
