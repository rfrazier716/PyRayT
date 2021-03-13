import unittest
import numpy as np
import tinygfx.g3d as cg
from tinygfx.g3d import primitives


class TestCountedObject(unittest.TestCase):
    def setUp(self):
        self.obj = cg.CountedObject()

    def test_count_incrementing(self):
        obj_to_create = 20
        initial_count = cg.CountedObject.get_count()
        test_objects = [cg.CountedObject() for _ in range(obj_to_create)]

        # the count should be the same for each object and be equal to the number of objects created
        self.assertEqual(cg.CountedObject.get_count(), obj_to_create + initial_count)

        # none of the objects should have the same count_id
        obj_ids = set([test_object.get_id() for test_object in test_objects])
        self.assertEqual(len(obj_ids), obj_to_create)

        # the objects should be incremented sequentially
        self.assertTrue(np.all(np.diff(np.sort(np.array(list(obj_ids)))) == 1))

    def test_decrementing_object_count(self):
        obj_to_create = 20
        initial_count = cg.CountedObject.get_count()
        test_objects = [cg.CountedObject() for _ in range(obj_to_create)]

        # the count should be the same for each object and be equal to the number of objects created
        self.assertEqual(cg.CountedObject.get_count(), obj_to_create + initial_count)

        # deleting objects should reduce the count
        for n in range(len(test_objects)):
            test_objects.pop(-1)
            self.assertEqual(cg.CountedObject.get_count(), obj_to_create + initial_count - n - 1)

    def test_creating_obj_after_deletion(self):
        # create a list of objects then delete them all

        obj_to_create = 20
        initial_count = cg.CountedObject.get_count()
        test_objects = [cg.CountedObject() for _ in range(obj_to_create)]
        max_id = test_objects[-1].get_id()
        del test_objects

        # make sure that a new object indexes up and does not reuse any of the deleted IDs
        new_object = cg.CountedObject()
        self.assertEqual(new_object.get_id(), max_id + 1)


class WorldObjectTestCase(unittest.TestCase):
    """
    a base class whose setup method creates a new world object
    """

    def setUp(self):
        self._obj = cg.WorldObject()


class TestWorldObjectCreation(WorldObjectTestCase):
    def test_object_creation(self):
        # the object should be centered at the origin facing the positive z-axis
        self.assertTrue(np.array_equal(self._obj.get_position(), primitives.Point(0, 0, 0)))
        self.assertTrue(np.array_equal(self._obj.get_orientation(), primitives.Vector(0, 0, 1)))

    def test_modifying_transform_matrix(self):
        # transforming a returned value should not be copied
        tx_matrix = self._obj.get_world_transform()
        tx_matrix[3, 0] = 5
        tx_matrix = self._obj.get_world_transform()
        self.assertNotEqual(tx_matrix[3, 0], 5)

        tx_matrix = self._obj.get_object_transform()
        tx_matrix[3, 0] = 5
        tx_matrix = self._obj.get_object_transform()
        self.assertNotEqual(tx_matrix[3, 0], 5)

    def test_updating_world_transform(self):
        ### if you update the world transform matrix the object transform matrix will be updated when called
        self._obj.scale_all(10)
        to_world_mat = self._obj.get_object_transform()
        expected_matrix = np.identity(4, dtype=float)
        for x in range(3):
            expected_matrix[x, x] = 0.1
        self.assertTrue(np.allclose(expected_matrix, to_world_mat))

    def test_getting_world_coordinates(self):
        self._obj.scale_all(10)
        local_point = primitives.Point(1, 1, 1)
        world_point = self._obj.to_world_coordinates(local_point)
        self.assertTrue(np.allclose(world_point, primitives.Point(10, 10, 10)))


class TestWorldObjectScaling(WorldObjectTestCase):
    def setUp(self):
        super().setUp()  # call the parent setup function to make the object
        self._obj.move(1, 1, 1)  # move the object to 1,1,1 so scale operations work

    def test_orientation_does_not_scale(self):
        self._obj.scale(100, 100, 100)
        self.assertTrue(np.allclose(self._obj.get_orientation(), primitives.Vector(0, 0, 1.)))

    def test_single_axis_scale(self):
        # move the object to 1,1,1 so that the scale effects it
        scale_axes = "xyz"
        scale_fns = [getattr(self._obj, "scale_" + axis) for axis in scale_axes]
        scale_values = [3, 4, 5]
        for fn, scale in zip(scale_fns, scale_values):
            fn(scale)
        self.assertTrue(np.allclose(self._obj.get_position(), primitives.Point(*scale_values)),
                        f"{self._obj.get_position()}")

    def test_3axis_scale(self):
        scale_values = (3, 4, 5)
        self._obj.scale(*scale_values)
        self.assertTrue(np.allclose(self._obj.get_position(), primitives.Point(*scale_values)),
                        f"{self._obj.get_position()}")

    def test_chained_scale(self):
        scale_values = (3, 4, 5)
        self._obj.scale_x(scale_values[0]).scale_y(scale_values[1]).scale_z(scale_values[2])
        self.assertTrue(np.allclose(self._obj.get_position(), primitives.Point(*scale_values)),
                        f"{self._obj.get_position()}")

    def test_scale_all(self):
        scale_factor = 1000
        expected_pos = scale_factor * np.ones(3)
        self._obj.scale_all(scale_factor)
        self.assertTrue(np.allclose(self._obj.get_position(), primitives.Point(*expected_pos)),
                        f"{self._obj.get_position()}")

    def test_negative_scaling(self):
        # negative values should raise an exception
        scale_fns = (getattr(self._obj, x) for x in ['scale', 'scale_x', 'scale_y', 'scale_z', 'scale_all'])
        for fn in scale_fns:
            with self.assertRaises(ValueError):
                fn(-1)

    def test_invalid_norm(self):
        # force a scale of zero and assert an error is raised
        scale_factor = 0
        with self.assertRaises(ValueError):
            self._obj.scale_all(scale_factor)


class TestWorldObjectTranslation(WorldObjectTestCase):
    def test_3axis_movement(self):
        my_obj = cg.WorldObject()

        # We should be able to move the object multiple times and the position will move but not direction
        move_vector = np.array((1, 2, -5))
        my_obj.move(*move_vector)  # move the object in space
        self.assertTrue(np.array_equal(my_obj.get_position(), primitives.Point(*move_vector)))

        # reversing the move gets you back to the origin
        my_obj.move(*(-move_vector))  # move the object in space
        self.assertTrue(np.array_equal(my_obj.get_position(), primitives.Point()))

    def test_single_axis_movement(self):
        # individual move functions execute properly
        attr_names = ["move_" + direction for direction in "xyz"]
        move_attrs = [getattr(self._obj, attribute) for attribute in attr_names]
        movement = 3
        for n, fn_call in enumerate(move_attrs):
            fn_call(movement)
            self.assertEqual(self._obj.get_position()[n], movement)

    def test_chained_movement(self):
        movement = 3
        self._obj.move_x(movement).move_y(movement).move_z(movement)
        self.assertTrue(np.array_equal(self._obj.get_position(), primitives.Point(movement, movement, movement)))


class TestWorldObjectRotation(WorldObjectTestCase):
    def test_rotation(self):
        my_obj = self._obj

        # rotation about the y-axis by 90 degree should change the direction vector to x
        my_obj.rotate_y(90, units="deg")
        self.assertTrue(np.allclose(my_obj.get_orientation(), primitives.Vector(1., 0, 0)))

        # now rotation it about the z-axis 90 degree should have it point to positive y
        my_obj.rotate_z(90, units="deg")
        self.assertTrue(np.allclose(my_obj.get_orientation(), primitives.Vector(0, 1., 0)))

        # rotation 90 degree about the x-axis should reset it to positive z
        my_obj.rotate_x(90, units="deg")
        self.assertTrue(np.allclose(my_obj.get_orientation(), primitives.Vector(0, 0, 1.)))

    def test_rotation_chain(self):
        # rotations should be able to be cascaded
        self._obj.rotate_y(90, units="deg").rotate_z(90, units="deg").rotate_x(90, units="deg")
        self.assertTrue(np.allclose(self._obj.get_orientation(), primitives.Vector(0, 0, 1.)))

    def test_rotation_units(self):
        # rotations should work for radians and degrees
        rotation_angles = [90, np.pi / 2]
        rotation_units = ["deg", "rad"]
        for angle, unit in zip(rotation_angles, rotation_units):
            self._obj.rotate_y(angle, units=unit).rotate_z(angle, units=unit).rotate_x(angle, units=unit)
            self.assertTrue(np.allclose(self._obj.get_orientation(), primitives.Vector(0, 0, 1.)),
                            f"Test Failed for unit {unit}, has orientation {self._obj.get_orientation()}")

        # make sure that an invalid rotaiton option raises an error
        with self.assertRaises(ValueError):
            self._obj.rotate_x(90, "Chickens")


class TestWorldObjectQuaternion(WorldObjectTestCase):
    def test_default_quat(self):
        quat = self._obj.get_quaternion()
        self.assertTrue(np.allclose(quat[:3], np.zeros(3)))
        self.assertAlmostEqual(quat[-1], 1.)

    def test_single_axis_quat(self):
        rotation_angle = np.pi / 2
        self._obj.rotate_y(rotation_angle, "rad")  # rotate along the y-axis by 90 degrees
        quat = self._obj.get_quaternion()
        expected_vect = np.asarray((0, 1, 0)) * np.sqrt(2) / 2
        expected_scalar = np.sqrt(2) / 2
        self.assertTrue(np.allclose(quat[:3], expected_vect))
        self.assertAlmostEqual(quat[-1], expected_scalar)


class TestObjectGroup(unittest.TestCase):
    def setUp(self):
        self.group = cg.ObjectGroup()

        self.obj1 = cg.WorldObject()
        self.obj2 = cg.WorldObject()

        self.group.append(self.obj1)
        self.group.append(self.obj2)

    def testing_list_properties(self):
        # the group should have two elements in it
        self.assertEqual(len(self.group), 2)

        # the group should be iterable
        expected_refs = (self.obj1, self.obj2)
        for expected, actual in zip(expected_refs, self.group):
            self.assertEqual(expected, actual)

        self.assertTrue(hasattr(self.group, '__iter__'))

    def testing_operations_on_group(self):
        # objects can be moved outside of the group
        self.obj1.move(1, 0, 0)
        self.obj2.move(-1, 0, 0)

        scale = 2
        self.group.scale_all(scale)  # now scale the group by 2
        self.assertTrue(np.allclose(self.obj1.get_position(), primitives.Point(scale, 0, 0)))
        self.assertTrue(np.allclose(self.obj2.get_position(), primitives.Point(-scale, 0, 0)))

        # rotation also applies
        self.group.rotate_z(90)
        self.assertTrue(np.allclose(self.obj1.get_position(), primitives.Point(0, scale, 0)))
        self.assertTrue(np.allclose(self.obj2.get_position(), primitives.Point(0, -scale, 0)))

    def test_nesting_object_group(self):
        # make a subgroup and append it to the top level group
        subgroup = cg.ObjectGroup()

        sub_object = cg.WorldObject()
        sub_object.move(1, 0, 0)
        subgroup.append(sub_object)

        self.group.append(subgroup)

        x_movement = 3
        self.group.move_x(x_movement)  # move the top level group

        self.assertTrue(np.allclose(subgroup.get_position(), primitives.Point(x_movement, 0, 0)))
        self.assertTrue(np.allclose(sub_object.get_position(), primitives.Point(x_movement + 1, 0, 0)))


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


if __name__ == '__main__':
    unittest.main()
