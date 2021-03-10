import unittest
import tinygfx.g3d as cg
import numpy as np
import tinygfx.g3d.primitives as primitives


class TestSphere(unittest.TestCase):
    def setUp(self) -> None:
        self.radius = 3
        self.sphere = cg.Sphere(self.radius)
        self.ray = primitives.Ray(direction=primitives.Vector(1, 0, 0))

        self.intersection_points = ((0, 0, -1), (0, 0, 1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0))
        self.intersections = [primitives.Point(*intersection) for intersection in self.intersection_points]

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
        self.intersections = [primitives.Point(*intersection) for intersection in scaled_intersection_points]
        # for a nontransformed sphere the normals should be vectors of the coordinates
        normals = [self.sphere.get_world_normals(intersection) for intersection in self.intersections]
        for normal, intersection in zip(normals, scaled_intersection_points):
            expected = primitives.Vector(*intersection) / scaling
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
            expected = primitives.Vector(*intersection)
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
            expected = primitives.Vector(*intersection)
            self.assertTrue(np.allclose(normal, expected), f"Expected {expected}, got {normal}")
            self.assertAlmostEqual(np.linalg.norm(normal), 1.0)

    def test_normal_inversion(self):
        cube = cg.Cuboid()
        point = primitives.Point(1, 0, 0)
        normals = cube.get_world_normals(point)
        self.assertTrue(np.allclose(normals, primitives.Vector(*point[:-1])), f"{normals}")

        cube.invert_normals()
        normals = cube.get_world_normals(point)
        self.assertTrue(np.allclose(normals, -primitives.Vector(*point[:-1])), f"{normals}")


class TestCuboid(unittest.TestCase):
    def setUp(self) -> None:
        self.cube = cg.Cuboid()
        self.rays = (
            primitives.Ray(direction=primitives.Vector(-1, 0, 0)),
            primitives.Ray(direction=primitives.Vector(1, 0, 0)),
            primitives.Ray(direction=primitives.Vector(0, -1, 0)),
            primitives.Ray(direction=primitives.Vector(0, 1, 0)),
            primitives.Ray(direction=primitives.Vector(0, 0, -1)),
            primitives.Ray(direction=primitives.Vector(0, 0, 1))
        )

    def test_default_constructor(self):
        hit = np.asarray([self.cube.intersect(ray) for ray in self.rays])
        # all hits should be a distance of 1
        self.assertTrue(np.allclose(hit, 1))

    def test_from_length_constructor(self):
        lengths = (2, 3, 4)
        cube = cg.Cuboid.from_lengths(*lengths)
        expected_hits = np.array((2, 2, 3, 3, 4, 4)) / 2
        hits = np.asarray([cube.intersect(ray)[0] for ray in self.rays])

        self.assertTrue(np.allclose(hits, expected_hits), f"{hits}")

    def test_from_corner_constructor_tuple(self):
        corner0 = (-1, -2, -3)
        corner1 = (4, 5, 6)

        cube = cg.Cuboid.from_corners(corner0, corner1)
        hits = np.asarray([cube.intersect(ray)[0] for ray in self.rays])

        # the hits in the negative direction should be the same as the first corner
        self.assertTrue(np.allclose(hits[::2], np.abs(corner0)), f"{hits}")

        # the hits in the positive direction should be the same as the second corner
        self.assertTrue(np.allclose(hits[1::2], np.abs(corner1)), f"{hits}")

    def test_from_corner_constructor_point(self):
        corner0 = primitives.Point(*(-1, -2, -3))
        corner1 = primitives.Point(*(4, 5, 6))

        cube = cg.Cuboid.from_corners(corner0, corner1)
        hits = np.asarray([cube.intersect(ray)[0] for ray in self.rays])

        # the hits in the negative direction should be the same as the first corner
        self.assertTrue(np.allclose(hits[::2], np.abs(corner0[:-1])), f"{hits}")

        # the hits in the positive direction should be the same as the second corner
        self.assertTrue(np.allclose(hits[1::2], np.abs(corner1[:-1])), f"{hits}")

    def test_from_corner_constructor_error(self):
        with self.assertRaises(ValueError):
            corner0 = primitives.Point(*(-1, -2, -3))
            corner1 = primitives.Point(*(4, 5, 6))
            cube = cg.Cuboid.from_corners(corner1, corner0)

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
        cube = cg.Cuboid()
        point = primitives.Point(1, 0, 0)
        normals = cube.get_world_normals(point)
        self.assertTrue(np.allclose(normals, primitives.Vector(*point[:-1])), f"{normals}")

        cube.invert_normals()
        normals = cube.get_world_normals(point)
        self.assertTrue(np.allclose(normals, -primitives.Vector(*point[:-1])), f"{normals}")


class TestAperturedSurface(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = cg.YZPlane()
        self.surface.aperture = cg.CircularAperture(1)

    def test_apertured_intersection(self):
        # any point in the aperture should still intersect
        ray = primitives.Ray(primitives.Point(), primitives.Vector(1, 0, 0))
        self.assertTrue(self.surface.intersect(ray) != -1)

        # a point higher than 1 shouldn't register as a hit
        ray = primitives.Ray(primitives.Point(0, 10, 0), primitives.Vector(1, 0, 0))
        self.assertTrue(self.surface.intersect(ray) == -1)

    def test_moving_surface(self):
        # the ray should no longer intersect the plane
        ray = primitives.Ray(primitives.Point(1, 10, 0), primitives.Vector(1, 0, 0))
        self.assertTrue(self.surface.intersect(ray) == -1)

        # but if we move the surface the aperture will follow
        self.surface.move(2, 10)
        self.assertTrue(self.surface.intersect(ray) != -1)

        # and now a ray at the origin will no longer intersect
        ray = primitives.Ray(primitives.Point(), primitives.Vector(1, 0, 0))
        self.assertTrue(self.surface.intersect(ray) == -1)

    def test_moving_aperture(self):
        # the aperture can also be moved independently of the surface
        ray = primitives.Ray(primitives.Point(0, 10, 0), primitives.Vector(1, 0, 0))
        self.assertTrue(self.surface.intersect(ray) == -1)

        self.surface.aperture.move(0, 10, 0)
        self.assertTrue(self.surface.intersect(ray) != -1)

    def test_combination_movement(self):
        # you can move the aperture and then the surface again
        ray = primitives.Ray(primitives.Point(1, 10, 0), primitives.Vector(1, 0, 0))
        self.assertTrue(self.surface.intersect(ray) == -1)

        # this time just moving the aperture won't work because the ray is still in front of the plane
        self.surface.aperture.move(0, 10, 0)
        self.assertTrue(self.surface.intersect(ray) == -1)

        # but if we move the whole surface it will
        self.surface.move(10)
        self.assertTrue(self.surface.intersect(ray) != -1)

    def test_aperture_masking_intersections(self):
        # if you intersect a surface with an aperture, even if there is a nearer intercept it will only count the
        # aperture intersection

        self.surface = cg.Sphere(1)
        ray = primitives.Ray(primitives.Point(-1, 0, -2), primitives.Vector(1, 0, 1))
        self.assertEqual(self.surface.intersect(ray)[0], 1)

        # if we add an aperture now the nearest intersection will be 2, because the first intersection is not within
        # the aperture

        self.surface.aperture = cg.CircularAperture(0.5)
        ray = primitives.Ray(primitives.Point(-1, 0, -2), primitives.Vector(1, 0, 1))
        self.assertEqual(self.surface.intersect(ray)[0], 2)


if __name__ == '__main__':
    unittest.main()
