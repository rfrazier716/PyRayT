import tinygfx.g3d.primitives as primitives
import numpy as np
import unittest


class TestHomogeneousCoordinate(unittest.TestCase):
    def setUp(self):
        self.coord = primitives.HomogeneousCoordinate(3, 4, 5, 6)

    def test_getting_object_values(self):
        getters = [getattr(self.coord, getter) for getter in "xyzw"]
        for n, getter in enumerate(getters):
            self.assertEqual(self.coord[n], getter)

    def test_setting_object_values(self):
        """
        we should be able to update the coordinate entries by both array indexing and value
        :return:
        """
        # update by calling the setter method
        value_to_set = 0
        [setattr(self.coord, setter, value_to_set) for setter in "xyzw"]
        for n, getter in enumerate("xyzw"):
            self.assertEqual(getattr(self.coord, getter), value_to_set)
            self.assertEqual(self.coord[n], value_to_set)

        # update by directly calling the array index
        value_to_set = 1000
        for n in range(4):
            self.coord[n] = value_to_set

        for n, getter in enumerate("xyzw"):
            self.assertEqual(getattr(self.coord, getter), value_to_set)
            self.assertEqual(self.coord[n], value_to_set)

    def test_normalizing(self):
        coord = primitives.HomogeneousCoordinate(1, 1, 1, 0)
        self.assertTrue(np.allclose(coord.normalize(), np.array((1, 1, 1, 0)) / np.sqrt(3)),
                        f" got {coord.normalize()}")

        coord = primitives.HomogeneousCoordinate(1, 1, 1, 1)
        self.assertTrue(np.allclose(coord.normalize(), np.array((1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), 1))),
                        f" got {coord.normalize()}")


class TestPoint(unittest.TestCase):
    def setUp(self):
        self.coord = primitives.Point(3, 4, 5)

    def test_fields_accessable(self):
        self.assertEqual(self.coord.w, 1)


class TestVector(unittest.TestCase):
    def setUp(self):
        self.coord = primitives.Vector(3, 4, 5)

    def test_fields_accessable(self):
        self.assertEqual(self.coord.w, 0)


class TestRay(unittest.TestCase):
    def test_ray_setters(self):
        ray = primitives.Ray()
        ray.direction = primitives.Vector(3, 2, 1)
        self.assertTrue(np.allclose(ray.direction, primitives.Vector(3, 2, 1)))

        ray.origin = primitives.Point(1, 2, 3)
        self.assertTrue(np.allclose(ray.origin, primitives.Point(1, 2, 3)))

    def test_ray_initialization(self):
        ray = primitives.Ray()
        self.assertTrue(np.allclose(ray.origin, primitives.Point()))
        self.assertTrue(np.allclose(ray.direction, primitives.Vector(1, 0, 0)))

        # the ray elements can be assigned
        ray.origin.x = 3
        self.assertEqual(ray.origin[0], 3)

    def test_ray_bundling(self):
        n_rays = 100
        all_rays = primitives.bundle_rays([primitives.Ray() for _ in range(n_rays)])
        self.assertTrue(all_rays.shape, (2, 4, n_rays))

        # rays lose their view when stacked
        with self.assertRaises(AttributeError):
            all_rays[0].origin
            all_rays[0].direction

        # but can be viewed as ray objects
        self.assertEqual(all_rays[0].view(primitives.Ray).origin.x, 0)


class TestSphere(unittest.TestCase):
    def setUp(self) -> None:
        self.sphere = primitives.Sphere()
        self.ray = primitives.Ray()

        self.intersection_points = ((0, 0, -1), (0, 0, 1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0))
        self.intersections = [primitives.Point(*intersection) for intersection in self.intersection_points]

    def test_getting_radius(self):
        # default constructor should assign a radius of 1
        self.assertEqual(self.sphere.get_radius(), 1)

        # a new sphere can have the radius assigned
        self.assertEqual(primitives.Sphere(3).get_radius(), 3)

    def test_boundary_points(self):
        # make sure you get the correct list of corners from the cube
        corners = primitives.Sphere(3).bounding_points
        expected_corners = {
            (-3, -3, -3),
            (-3, -3, 3),
            (-3, 3, -3),
            (-3, 3, 3),
            (3, -3, -3),
            (3, -3, 3),
            (3, 3, -3),
            (3, 3, 3),
        }
        self.assertEqual(expected_corners, set(map(tuple, corners[:3].T)))

    def test_ray_intersection_unit_sphere(self):
        hit = self.sphere.intersect(self.ray)
        self.assertEqual(hit.shape, (2, 1))

        # want hits to be -1 and 1 but dont' care about ordering
        self.assertTrue(1.0 in hit[:, 0])
        self.assertTrue(-1.0 in hit[:, 0], f"-1 not in {hit[:, 0]}")

        # if the ray is moved out of the radius of the sphere we get inf as the hit
        new_ray = primitives.Ray()
        new_ray.origin = primitives.Point(0, 0, 2)
        self.assertEqual(self.sphere.intersect(new_ray)[0, 0], np.inf)

    def test_intersection_sphere_behind_ray(self):
        ray_offset = 100
        ray = primitives.Ray(primitives.Point(ray_offset, 0, 0), primitives.Vector(1, 0, 0))
        hits = self.sphere.intersect(ray)
        expected_hits = [-ray_offset + j * self.sphere.get_radius() for j in (-1, 1)]
        for hit in expected_hits:
            self.assertTrue(hit in hits[:, 0], f"{hit} was not found in {hits[:, 0]}")

    def test_multi_ray_intersection(self):
        n_rays = 100
        rays = primitives.bundle_rays([primitives.Ray() for _ in range(n_rays)])
        all_hits = self.sphere.intersect(rays)
        self.assertEqual(all_hits.shape, (2, n_rays))
        self.assertTrue(np.allclose(all_hits[0], self.sphere.get_radius()))
        self.assertTrue(np.allclose(all_hits[1], -self.sphere.get_radius()))

    def test_intersection_skew_case(self):
        hit = self.sphere.intersect(
            primitives.Ray(primitives.Point(0, 0, 2 * self.sphere.get_radius()), primitives.Vector(1, 0, 0)))
        self.assertAlmostEqual(hit[0], np.inf)

    def test_double_intersection(self):
        hit = self.sphere.intersect(
            primitives.Ray(origin=primitives.Point(-1, 0, 1), direction=primitives.Vector(1, 0, 0)))
        self.assertTrue(np.allclose(hit[:, 0], 1.0))

    def test_normals_base_sphere(self):
        # for a nontransformed sphere the normals should be vectors of the coordinates
        normals = [self.sphere.normal(intersection) for intersection in self.intersections]
        for normal, intersection in zip(normals, self.intersection_points):
            expected = primitives.Vector(*intersection)
            self.assertTrue(np.allclose(normal, expected))
            self.assertAlmostEqual(np.linalg.norm(normal), 1.0)

    def test_arrayed_normals(self):
        # made a bunch of random points on the unit sphere
        normal_coordinates = 2 * (np.random.random_sample((4, 1000)) - 0.5)
        normal_coordinates[-1] = 1
        normal_coordinates[:-1] /= np.linalg.norm(normal_coordinates[:-1], axis=0)

        # make sure the sphere normal function accepts an array
        normals = self.sphere.normal(normal_coordinates)
        normals[-1] = 1
        self.assertTrue(np.allclose(normal_coordinates, normals))


class TestParaboloid(unittest.TestCase):
    def setUp(self) -> None:
        self.f = 5
        self.surface = primitives.Paraboloid(self.f)

    def test_object_getters(self):
        self.assertEqual(self.surface.get_focus(), self.f)

    def test_intersection_at_origin(self):
        hit = self.surface.intersect(primitives.Ray(primitives.Point(0, 0, 0), primitives.Vector(0, 1, 0)))
        self.assertTrue(np.allclose(hit, 0), f"got hits {hit}")

        hit = self.surface.intersect(primitives.Ray(primitives.Point(0, 0, 0), primitives.Vector(0, 0, 1)))
        self.assertTrue(np.allclose(hit, 0), f"got hits {hit}")

        hit = self.surface.intersect(primitives.Ray(primitives.Point(0, 0, 0), primitives.Vector(1, 0, 0)))
        self.assertTrue(np.allclose(hit, 0), f"got hits {hit}")

    def test_intersection_linear_case(self):
        hit = self.surface.intersect(primitives.Ray(primitives.Point(-1, 0, 0), primitives.Vector(1, 0, 0)))
        self.assertEqual(hit.shape, (2, 1))
        self.assertTrue(np.allclose(hit, 1))

        hit = self.surface.intersect(primitives.Ray(primitives.Point(0, 0, -2 * self.f), primitives.Vector(1, 0, 0)))
        self.assertTrue(np.allclose(hit, self.f))

    def test_intersection_trivial_case(self):
        hit = self.surface.intersect(primitives.Ray(primitives.Point(0, 0, 0)))
        self.assertTrue(np.allclose(hit, 0))

        hit = self.surface.intersect(primitives.Ray(primitives.Point(0, 0, 0), primitives.Vector(0, 1, 1) / np.sqrt(2)))
        self.assertTrue(np.allclose(hit, 0))

    def test_intersection_dbl_root_case(self):
        hit = self.surface.intersect(
            primitives.Ray(primitives.Point(self.f, -2 * self.f, 0), primitives.Vector(0, 1, 0)))
        self.assertTrue(np.allclose(np.sort(hit[:, 0]), np.array((0, 4 * self.f))), f"{hit}")

        hit = self.surface.intersect(primitives.Ray(primitives.Point(self.f, 0, 0), primitives.Vector(0, 1, 0)))
        self.assertTrue(np.allclose(np.sort(hit[:, 0]), np.array((-2, 2)) * self.f), f"{hit}")

    def test_intersection_skew_case(self):
        hit = self.surface.intersect(primitives.Ray(primitives.Point(-1, 0, 0), primitives.Vector(0, 1, 0)))
        self.assertTrue(np.allclose(hit, np.inf))

        hit = self.surface.intersect(primitives.Ray(primitives.Point(-1, 0, 0), primitives.Vector(0, 1, 1)))
        self.assertTrue(np.allclose(hit, np.inf))

        hit = self.surface.intersect(primitives.Ray(primitives.Point(-1, 0, 0), primitives.Vector(0, 1, -1)))
        self.assertTrue(np.allclose(hit, np.inf))

    def test_intersection_arrayed_case(self):
        # make a bunch of rays to intersect, move some s.t. they intersect teh surface at a different point
        n_rays = 1000
        split_index = int(n_rays / 2)
        rays = primitives.bundle_of_rays(n_rays)
        rays[0, 0, :split_index] = self.f  # move the ray's up to originate at the focus
        rays[1, 1, :split_index] = 1  # have the rays move towards the positive y_axis
        rays[1, 0, split_index:] = 1

        hits = self.surface.intersect(rays)

        self.assertEqual(hits.shape, (2, n_rays))

        self.assertTrue(np.allclose(np.sort(hits[:, :split_index], axis=0).T, np.array((-2 * self.f, 2 * self.f))))
        self.assertTrue(np.allclose(np.sort(hits[:, split_index:], axis=0).T, np.array((0, 0))))

    #  self.assertTrue(np.allclose(hits[split_index:], 0))

    def test_intersection_negative_focus(self):
        #  if the focus is negative the intersections should be in the -x region
        surface = primitives.Paraboloid(-self.f)
        hit = surface.intersect(primitives.Ray(primitives.Point(-self.f, 0, 0), primitives.Vector(0, 1, 0)))
        self.assertTrue(np.allclose(np.sort(hit[:, 0]), np.array((-2, 2)) * self.f), f"{hit}")

    def test_intersection_far(self):
        # check that an error is not raised when a grossly large value is passed to the intersection
        # TODO: decide a max distance to clip value at?
        hit = self.surface.intersect(primitives.Ray(primitives.Point(0, 0, 1000000000), primitives.Vector(1.3, 0, 0)))

    def test_normal(self):
        normal = self.surface.normal(primitives.Point(0, 0, 0))
        self.assertTrue(np.allclose(normal, primitives.Vector(-1, 0, 0)))

        normal = self.surface.normal(primitives.Point(self.f, 2 * self.f, 0))
        self.assertTrue(np.allclose(normal, primitives.Vector(-1, 1, 0) / np.sqrt(2)))

    def test_arrayed_normals(self):
        n_pts = 1000
        coords = np.zeros((4, n_pts))
        coords[-1] = 1  # make them points

        normals = self.surface.normal(coords)
        self.assertEqual(normals.shape, (4, n_pts))

        expected_normals = np.zeros((4, n_pts))
        expected_normals[0] = -1
        self.assertTrue(np.allclose(normals, expected_normals))


class TestPlane(unittest.TestCase):
    def setUp(self):
        self.surface = primitives.Plane()

    def test_positive_intersection(self):
        ray = primitives.Ray(primitives.Point(-1, 0, 0), primitives.Vector(1, 0, 0))
        hit = self.surface.intersect(ray)[0, 0]
        self.assertAlmostEqual(hit, 1)

        ray = primitives.Ray(primitives.Point(-1, 0, 0), primitives.Vector(1, 1, 0).normalize())
        hit = self.surface.intersect(ray)[0, 0]
        self.assertAlmostEqual(hit, np.sqrt(2))

    def test_negative_intersection(self):
        ray = primitives.Ray(primitives.Point(1, 0, 0), primitives.Vector(1, 0, 0))
        hit = self.surface.intersect(ray)[0, 0]
        self.assertAlmostEqual(hit, -1)

    def test_parallel_intersection(self):
        ray = primitives.Ray(primitives.Point(1, 0, 0), primitives.Vector(0, 1, 1).normalize())
        hit = self.surface.intersect(ray)[0, 0]
        self.assertAlmostEqual(hit, np.inf)

    def test_arrayed_intersection(self):
        n_rays = 1000
        split = int(n_rays / 2)

        # make a bunch of rays at x=-1, half will point towards the positive x-axis and the other will
        # point towards the positive y-axis
        rays = primitives.bundle_of_rays(1000)
        rays[0, 0] = -1
        rays[1, 0, :split] = 1
        rays[1, 1] = 1

        hit = self.surface.intersect(rays)
        self.assertEqual(hit.shape, (1, n_rays), f"Ray shape is {hit.shape}")
        self.assertTrue(np.allclose(hit[0, :split], 1))
        self.assertTrue(np.allclose(hit[0, split:], np.inf))


class TestCube(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = primitives.Cube()

    def test_getting_corner_points(self):
        # make sure you get the correct list of corners from the cube
        corners = primitives.Cube((-1, -1, -1), (1, 2, 3)).bounding_points
        expected_corners = {
            (-1, -1, -1),
            (-1, -1, 3),
            (-1, 2, -1),
            (-1, 2, 3),
            (1, -1, -1),
            (1, -1, 3),
            (1, 2, -1),
            (1, 2, 3),
        }
        self.assertEqual(expected_corners, set(map(tuple, corners[:3].T)))

    def test_intersection_within_cube(self):
        rays = (
            primitives.Ray(direction=primitives.Vector(1, 0, 0)),
            primitives.Ray(direction=primitives.Vector(0, 1, 0)),
            primitives.Ray(direction=primitives.Vector(0, 0, 1))
        )
        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[-1, 1]]).T), f"{hit}")

    def test_intersection_external_to_cube(self):
        rays = (
            primitives.Ray(origin=primitives.Point(-2, 0, 0), direction=primitives.Vector(1, 0, 0)),
            primitives.Ray(origin=primitives.Point(0, -2, 0), direction=primitives.Vector(0, 1, 0)),
            primitives.Ray(origin=primitives.Point(0, 0, -2), direction=primitives.Vector(0, 0, 1))
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[1, 3]]).T), f"{hit}")

    def test_intersection_at_angle(self):
        ray = primitives.Ray(origin=primitives.Point(-2, -1, 0), direction=primitives.Vector(1, 1, 0).normalize())

        hit = self.surface.intersect(ray)
        self.assertTrue(np.allclose(np.sort(hit, axis=0), np.sqrt(2) * np.array([[1, 2]]).T), f"{hit}")

    def test_skew_intersection(self):
        ray = primitives.Ray(origin=primitives.Point(-2, 0, 0), direction=primitives.Vector(0, 1, 0).normalize())
        hit = self.surface.intersect(ray)
        self.assertTrue(np.allclose(hit, np.inf))

    def test_nondefault_constructor(self):
        cube_extents = (1, 2, 5)
        surface = primitives.Cube((-1, -1, -1), cube_extents)
        rays = (
            primitives.Ray(direction=primitives.Vector(1, 0, 0)),
            primitives.Ray(direction=primitives.Vector(0, 1, 0)),
            primitives.Ray(direction=primitives.Vector(0, 0, 1))
        )

        for ray, extent in zip(rays, cube_extents):
            hit = surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[-1, extent]]).T), f"{hit}")

    def test_arrayed_intersection(self):
        # make a bunch of rays to intersect, move some s.t. they intersect teh surface at a different point
        n_rays = 1000
        split_index = int(n_rays / 2)
        rays = primitives.bundle_of_rays(n_rays)
        rays[0, 0, :split_index] = -0.5  # move half the rays back
        rays[1, 0, :split_index] = 1  # have the rays move towards the positive x-axis

        # make the back half of the rays skew
        rays[0, 0, split_index:] = -2
        rays[1, 1, split_index:] = -1

        hits = self.surface.intersect(rays)

        self.assertEqual(hits.shape, (2, n_rays))
        self.assertTrue(np.allclose(hits[0, :split_index], -0.5))
        self.assertTrue(np.allclose(hits[1, :split_index], 1.5))
        self.assertTrue(np.allclose(hits[0, split_index:], np.inf))
        self.assertTrue(np.allclose(hits[1, split_index:], np.inf))

    def test_normals(self):
        coords = (
            primitives.Point(-1, 0, 0),
            primitives.Point(1, 0, 0),
            primitives.Point(0, -1, 0),
            primitives.Point(0, 1, 0),
            primitives.Point(0, 0, -1),
            primitives.Point(0, 0, 1)
        )
        for coord in coords:
            expected = primitives.Vector(*coord[:-1])
            normal = self.surface.normal(coord)
            self.assertTrue(np.allclose(expected, normal), f"expected {expected}, got {normal}")

    def test_offcenter_normal(self):
        coord = primitives.Point(-1 + 1E-8, 0.3, 0.7)

        expected = primitives.Vector(-1, 0, 0)
        normal = self.surface.normal(coord)

        self.assertTrue(np.allclose(expected, normal), f"expected {expected}, got {normal}")

    def test_corner_normal(self):
        # for a corner case any of the three normals could be picked, but need to make sure the resulting normal is
        # in the right direction
        coord = primitives.Point(1, 1, 1)

        # can be any of these and still be valid,
        expected = primitives.Vector(1, 1, 1).normalize()
        normal = self.surface.normal(coord)

        self.assertTrue(np.allclose(expected, normal), f"expected {expected}, got {normal}")

    def test_arrayed_normals(self):
        # make a bunch of rays to intersect, move some s.t. they intersect thh surface at a different point
        n_rays = 1000
        split_index = int(n_rays / 2)
        points = np.zeros((4, n_rays))
        points[-1] = 1  # make points!
        points[0, :split_index] = -1
        points[1, split_index:] = 1

        normals = self.surface.normal(points)
        self.assertEqual(normals.shape, (4, n_rays))

        expected = np.zeros((4, split_index))
        expected[0] = -1
        self.assertTrue(np.allclose(normals[:, :split_index], expected))

        expected = np.zeros((4, split_index))
        expected[1] = 1
        self.assertTrue(np.allclose(normals[:, split_index:], expected))


class TestInfiniteCylinder(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = primitives.Cylinder(1, infinite=True)

    def test_intersection_to_sidewalls(self):
        rays = (
            primitives.Ray(origin=primitives.Point(-2, 0, 0), direction=primitives.Vector(1, 0, 0)),
            primitives.Ray(origin=primitives.Point(-2, 0, 1), direction=primitives.Vector(1, 0, 0)),
            primitives.Ray(origin=primitives.Point(-2, 0, 2), direction=primitives.Vector(1, 0, 0))
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[1, 3]]).T), f"{hit}")

    def test_no_intersection_inside(self):
        rays = (
            primitives.Ray(origin=primitives.Point(0, 0, 0), direction=primitives.Vector(0, 0, 1)),
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[-np.inf, np.inf]]).T), f"{hit}")

    def test_no_intersection_outside(self):
        """
        if the ray origin is outside of the cylinder and it does not intersect, the returned hits should both be np.inf
        this is so that when you eventually sort them with the cap hits you can tell there's no intersection
        :return:
        """
        rays = (
            primitives.Ray(origin=primitives.Point(2, 0, 0), direction=primitives.Vector(0, 0, 1)),
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[np.inf, np.inf]]).T), f"{hit}")

    def test_arrayed_intersection(self):
        n_rays = 1000
        split = int(n_rays / 2)

        # make a bunch of rays at x=0, half will point towards the positive x-axis and the other will
        # point towards the positive y-axis
        rays = primitives.bundle_of_rays(1000)
        rays[0, 0] = 0
        rays[1, 0, :split] = 1
        rays[1, 2, split:] = 1

        hit = self.surface.intersect(rays)
        self.assertEqual(hit.shape, (2, n_rays), f"Ray shape is {hit.shape}")
        self.assertTrue(np.allclose(hit[:, :split].T, np.array((-1, 1))))
        self.assertTrue(np.allclose(hit[:, split:].T, np.array((-np.inf, np.inf))))


class TestFiniteCylinder(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = primitives.Cylinder(1, infinite=False)

    def test_intersection_to_sidewalls(self):
        rays = (
            primitives.Ray(origin=primitives.Point(-2, 0, 0), direction=primitives.Vector(1, 0, 0)),
            primitives.Ray(origin=primitives.Point(-2, 0, 0.5), direction=primitives.Vector(1, 0, 0)),
            primitives.Ray(origin=primitives.Point(-2, 0, -0.5), direction=primitives.Vector(1, 0, 0))
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[1, 3]]).T), f"{hit}")

    def test_intersection_to_cap(self):
        rays = (
            primitives.Ray(origin=primitives.Point(0, 0, 0), direction=primitives.Vector(0, 0, 1)),
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[-1, 1]]).T), f"{hit}")

    def test_wall_cap_intersection(self):
        rays = (
            primitives.Ray(origin=primitives.Point(-2, 0, -1), direction=primitives.Vector(1, 0, 1)),
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[1, 2]]).T), f"{hit}")

    def test_no_intersection_outside(self):
        """
        if the ray origin is outside of the cylinder and it does not intersect, the returned hits should both be np.inf
        this is so that when you eventually sort them with the cap hits you can tell there's no intersection
        :return:
        """
        rays = (
            primitives.Ray(origin=primitives.Point(2, 0, 0), direction=primitives.Vector(0, 0, 1)),
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[np.inf, np.inf]]).T), f"{hit}")

    def test_arrayed_intersection(self):
        n_rays = 1000
        split = int(n_rays / 2)

        # make a bunch of rays at x=0, half will point towards the positive x-axis and the other will
        # point towards the positive y-axis
        rays = primitives.bundle_of_rays(1000)
        rays[0, 0] = 0
        rays[1, 0, :split] = 1
        rays[1, 2, split:] = 1

        hit = self.surface.intersect(rays)
        self.assertEqual(hit.shape, (2, n_rays), f"Ray shape is {hit.shape}")
        self.assertTrue(np.allclose(hit[:, :split].T, np.array((-1, 1))))
        self.assertTrue(np.allclose(hit[:, split:].T, np.array((-1, 1))))


if __name__ == '__main__':
    unittest.main()
