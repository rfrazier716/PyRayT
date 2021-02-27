import unittest
import pyrayt.surfaces as surf
import numpy as np
import pyrayt.simple_cg as cg
import abc


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
        point = cg.Point(1, 0, 0)
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
        point = cg.Point(1, 0, 0)
        normals = cube.get_world_normals(point)
        self.assertTrue(np.allclose(normals, cg.Vector(*point[:-1])), f"{normals}")

        cube.invert_normals()
        normals = cube.get_world_normals(point)
        self.assertTrue(np.allclose(normals, -cg.Vector(*point[:-1])), f"{normals}")


class TestAperturedSurface(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = surf.YZPlane()
        self.surface.aperture = surf.RectangularAperture(2, 2)

    def test_apertured_intersection(self):
        # any point in the aperture should still intersect
        ray = cg.Ray(cg.Point(), cg.Vector(1, 0, 0))
        self.assertTrue(self.surface.intersect(ray) != -1)

        # a point higher than 1 shouldn't register as a hit
        ray = cg.Ray(cg.Point(0, 10, 0), cg.Vector(1, 0, 0))
        self.assertTrue(self.surface.intersect(ray) == -1)

    def test_moving_surface(self):
        # the ray should no longer intersect the plane
        ray = cg.Ray(cg.Point(1, 10, 0), cg.Vector(1, 0, 0))
        self.assertTrue(self.surface.intersect(ray) == -1)

        # but if we move the surface the aperture will follow
        self.surface.move(2, 10)
        self.assertTrue(self.surface.intersect(ray) != -1)

        # and now a ray at the origin will no longer intersect
        ray = cg.Ray(cg.Point(), cg.Vector(1, 0, 0))
        self.assertTrue(self.surface.intersect(ray) == -1)

    def test_moving_aperture(self):
        # the aperture can also be moved independently of the surface
        ray = cg.Ray(cg.Point(0, 10, 0), cg.Vector(1, 0, 0))
        self.assertTrue(self.surface.intersect(ray) == -1)

        self.surface.aperture.move(0, 10, 0)
        self.assertTrue(self.surface.intersect(ray) != -1)

    def test_combination_movement(self):
        # you can move the aperture and then the surface again
        ray = cg.Ray(cg.Point(1, 10, 0), cg.Vector(1, 0, 0))
        self.assertTrue(self.surface.intersect(ray) == -1)

        # this time just moving the aperture won't work because the ray is still in front of the plane
        self.surface.aperture.move(0, 10, 0)
        self.assertTrue(self.surface.intersect(ray) == -1)

        # but if we move the whole surface it will
        self.surface.move(10)
        self.assertTrue(self.surface.intersect(ray) != -1)


class ApertureTest(unittest.TestCase):
    """
    Using a Circular Aperture to test the rotating, scaling, and translation properties
    """

    def setUp(self):
        # make a new instance of the aperture
        self.aperture = surf.CircularAperture(1)

    def aperture_intersection(self, point_in_aperture, point_out_of_aperture):
        self.assertTrue(self.aperture.points_in_aperture(point_in_aperture), f"{point_in_aperture} is not in aperture")
        self.assertFalse(self.aperture.points_in_aperture(point_out_of_aperture),
                         f"{point_out_of_aperture} is not in aperture")

    def test_default_intersections(self):
        self.aperture_intersection(
            cg.Point(100, 0.7, 0.7),  # in aperture
            cg.Point(0, 1, 1))  # out of aperture

    def test_scaled_intersections_smaller(self):
        self.aperture.scale_all(0.1)  # scale the aperture smaller
        self.aperture_intersection(
            cg.Point(100, 0, 0),  # in aperture
            cg.Point(0, 1, 1))  # out of aperture

    def test_scaled_intersections_larger(self):
        self.aperture.scale_all(1000)  # scale the aperture larger ( total scale is 1000)
        self.aperture_intersection(
            cg.Point(100, 500, 500),  # in aperture
            cg.Point(0, 1000.1, 0))  # out of aperture

    def test_moved_intersections(self):
        self.aperture.move(0, 10, 3)  # move the aperture in the yz plane
        self.aperture_intersection(
            cg.Point(0, 10, 3),
            cg.Point(0, 0, 0)
        )

    def test_moving_x_no_effect(self):
        self.aperture.move_x(100)
        self.aperture_intersection(
            cg.Point(100, 0.7, 0.7),  # in aperture
            cg.Point(0, 1, 1))  # out of aperture

    def test_rotated_intersections(self):
        self.aperture.move_y(10).rotate_x(90)
        self.aperture_intersection(
            cg.Point(0, 0, 10),
            cg.Point(0, 0, 0)
        )

    def test_arrayed_intersections(self):
        n_pts = 1000
        split_index = 500
        points = np.zeros((4, n_pts))
        points[-1] = 1
        points[1, :split_index] = 2
        intersections = self.aperture.points_in_aperture(points)
        self.assertFalse(np.all(intersections[:split_index]))
        self.assertTrue(np.all(intersections[split_index:]))


class TestCircularAperture(unittest.TestCase):
    def setUp(self):
        self.aperture = surf.CircularAperture(1)

    def test_getters(self):
        aperture = surf.CircularAperture(10)
        self.assertEqual(aperture.radius, 10)

    def test_points_in_aperture(self):
        coords = ((0, 0),
                  (1, 0),
                  (-1, 0),
                  (0, 1),
                  (0, -1),
                  (0.7, 0.7))

        for coord in coords:
            self.assertTrue(self.aperture.points_in_aperture(cg.Point(0, *coord)))

    def test_points_outside_aperture(self):
        coords = ((1, 1),
                  (-1, 1),
                  (-1, -1),
                  (1, -1),
                  (0, 1.001))

        for coord in coords:
            self.assertFalse(self.aperture.points_in_aperture(cg.Point(0, *coord)))


class TestEllipticalAperture(unittest.TestCase):
    def setUp(self):
        self.aperture = surf.EllipticalAperture(2, 1)

    def test_getters(self):
        aperture = surf.EllipticalAperture(4, 8)
        self.assertEqual(aperture.radii, (4, 8))

    def test_points_in_aperture(self):
        coords = ((0, 0),
                  (2, 0),
                  (-2, 0),
                  (0, 1),
                  (0, -1),
                  (0.7, 0.7))

        for coord in coords:
            self.assertTrue(self.aperture.points_in_aperture(cg.Point(0, *coord)))

    def test_points_outside_aperture(self):
        coords = ((2, 1),
                  (-2, 1),
                  (-2, -1),
                  (2, -1),
                  (0, 1.001),
                  (2.001, 0))

        for coord in coords:
            self.assertFalse(self.aperture.points_in_aperture(cg.Point(0, *coord)))


class TestRectangularAperture(unittest.TestCase):
    def setUp(self):
        self.aperture = surf.RectangularAperture(4, 2)

    def test_getters(self):
        aperture = surf.RectangularAperture(10, 12)
        self.assertEqual(aperture.side_lengths, (10, 12))

    def test_points_in_aperture(self):
        coords = ((2, 1),
                  (-2, 1),
                  (-2, -1),
                  (2, -1),
                  (0, 0),
                  (0, 1))

        for coord in coords:
            self.assertTrue(self.aperture.points_in_aperture(cg.Point(0, *coord)))

    def test_points_outside_aperture(self):
        coords = ((2.01, 1),
                  (3, 3),
                  (5, 5))

        for coord in coords:
            self.assertFalse(self.aperture.points_in_aperture(cg.Point(0, *coord)))


class TestSquare(unittest.TestCase):
    def setUp(self):
        self.aperture = surf.SquareAperture(2)

    def test_getters(self):
        aperture = surf.SquareAperture(20)
        self.assertEqual(aperture.side_lengths, (20, 20))

    def test_points_in_aperture(self):
        coords = ((1, 1),
                  (-1, 1),
                  (-1, -1),
                  (1, -1),
                  (0, 0),
                  (0, 1))

        for coord in coords:
            self.assertTrue(self.aperture.points_in_aperture(cg.Point(0, *coord)))

    def test_points_outside_aperture(self):
        coords = ((2, 1),
                  (-2, 1),
                  (-2, -1),
                  (2, -1))

        for coord in coords:
            self.assertFalse(self.aperture.points_in_aperture(cg.Point(0, *coord)))


if __name__ == '__main__':
    unittest.main()
