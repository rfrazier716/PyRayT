import unittest
import pyrayt.simple_cg as cg
import numpy as np


class WorldObjectTestCase(unittest.TestCase):
    """
    a base class whose setup method creates a new world object
    """

    def setUp(self):
        self._obj = cg.WorldObject()


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


class TestHomogeneousCoordinate(unittest.TestCase):
    def setUp(self):
        self.coord = cg.HomogeneousCoordinate(3, 4, 5, 6)

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
        coord = cg.HomogeneousCoordinate(1, 1, 1, 0)
        self.assertTrue(np.allclose(coord.normalize(), np.array((1, 1, 1, 0)) / np.sqrt(3)),
                        f" got {coord.normalize()}")

        coord = cg.HomogeneousCoordinate(1, 1, 1, 1)
        self.assertTrue(np.allclose(coord.normalize(), np.array((1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), 1))),
                        f" got {coord.normalize()}")


class TestPoint(unittest.TestCase):
    def setUp(self):
        self.coord = cg.Point(3, 4, 5)

    def test_fields_accessable(self):
        self.assertEqual(self.coord.w, 1)


class TestVector(unittest.TestCase):
    def setUp(self):
        self.coord = cg.Vector(3, 4, 5)

    def test_fields_accessable(self):
        self.assertEqual(self.coord.w, 0)


class TestRay(unittest.TestCase):
    def test_ray_setters(self):
        ray = cg.Ray()
        ray.direction = cg.Vector(3, 2, 1)
        self.assertTrue(np.allclose(ray.direction, cg.Vector(3, 2, 1)))

        ray.origin = cg.Point(1, 2, 3)
        self.assertTrue(np.allclose(ray.origin, cg.Point(1, 2, 3)))

    def test_ray_initialization(self):
        ray = cg.Ray()
        self.assertTrue(np.allclose(ray.origin, cg.Point()))
        self.assertTrue(np.allclose(ray.direction, cg.Vector(1, 0, 0)))

        # the ray elements can be assigned
        ray.origin.x = 3
        self.assertEqual(ray.origin[0], 3)

    def test_ray_bundling(self):
        n_rays = 100
        all_rays = cg.bundle_rays([cg.Ray() for _ in range(n_rays)])
        self.assertTrue(all_rays.shape, (2, 4, n_rays))

        # rays lose their view when stacked
        with self.assertRaises(AttributeError):
            all_rays[0].origin
            all_rays[0].direction

        # but can be viewed as ray objects
        self.assertEqual(all_rays[0].view(cg.Ray).origin.x, 0)


class TestRaySet(unittest.TestCase):
    def setUp(self):
        self.n_rays = 1000
        self.set = cg.RaySet(self.n_rays)

    def test_field_initialization(self):
        self.assertEqual(self.set.rays.shape, (2, 4, self.n_rays))
        self.assertEqual(self.set.metadata.shape, (len(cg.RaySet.fields), self.n_rays))

    def test_field_accessing_after_modifying_metadata(self):
        # makes sure that if you update the actual metadata contents, the fields reflect it
        for j in range(self.set.metadata.shape[0]):
            self.set.metadata[j] = j
            field_value = getattr(self.set, cg.RaySet.fields[j])
            self.assertTrue(np.allclose(field_value, j), f"Failed at index {j} with attribute {cg.RaySet.fields[j]}")

    def test_metadata_accessing_after_modifying_fields(self):
        # makes sure that if you update the actual metadata contents, the fields reflect it
        for j in range(self.set.metadata.shape[0]):
            field = cg.RaySet.fields[j]
            setattr(self.set, field, j)
            self.assertTrue(np.allclose(self.set.metadata[j], j))

    def test_updating_slices_of_fields(self):
        self.set.generation[:10] = 7
        self.assertTrue(np.allclose(self.set.metadata[0, :10], 7))

    def test_creation_from_concatenation(self):
        set1 = cg.RaySet(10)
        set1.wavelength = -1
        set2 = cg.RaySet(20)
        set2.wavelength = 2

        joined_set = cg.RaySet.concat(set1, set2)

        self.assertEqual(joined_set.metadata.shape[-1], 30)
        self.assertEqual(joined_set.rays.shape, (2, 4, 30))

        self.assertTrue(np.allclose(joined_set.id, np.arange(30)), f"{joined_set.id}")
        self.assertTrue(np.allclose(joined_set.wavelength[:10], -1))
        self.assertTrue(np.allclose(joined_set.wavelength[10:], 2))


class TestWorldObjectCreation(WorldObjectTestCase):
    def test_object_creation(self):
        # the object should be centered at the origin facing the positive z-axis
        self.assertTrue(np.array_equal(self._obj.get_position(), cg.Point(0, 0, 0)))
        self.assertTrue(np.array_equal(self._obj.get_orientation(), cg.Vector(0, 0, 1)))

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
        local_point = cg.Point(1, 1, 1)
        world_point = self._obj.to_world_coordinates(local_point)
        self.assertTrue(np.allclose(world_point, cg.Point(10, 10, 10)))


class TestWorldObjectScaling(WorldObjectTestCase):
    def setUp(self):
        super().setUp()  # call the parent setup function to make the object
        self._obj.move(1, 1, 1)  # move the object to 1,1,1 so scale operations work

    def test_orientation_does_not_scale(self):
        self._obj.scale(100, 100, 100)
        self.assertTrue(np.allclose(self._obj.get_orientation(), cg.Vector(0, 0, 1.)))

    def test_single_axis_scale(self):
        # move the object to 1,1,1 so that the scale effects it
        scale_axes = "xyz"
        scale_fns = [getattr(self._obj, "scale_" + axis) for axis in scale_axes]
        scale_values = [3, 4, 5]
        for fn, scale in zip(scale_fns, scale_values):
            fn(scale)
        self.assertTrue(np.allclose(self._obj.get_position(), cg.Point(*scale_values)), f"{self._obj.get_position()}")

    def test_3axis_scale(self):
        scale_values = (3, 4, 5)
        self._obj.scale(*scale_values)
        self.assertTrue(np.allclose(self._obj.get_position(), cg.Point(*scale_values)), f"{self._obj.get_position()}")

    def test_chained_scale(self):
        scale_values = (3, 4, 5)
        self._obj.scale_x(scale_values[0]).scale_y(scale_values[1]).scale_z(scale_values[2])
        self.assertTrue(np.allclose(self._obj.get_position(), cg.Point(*scale_values)), f"{self._obj.get_position()}")

    def test_scale_all(self):
        scale_factor = 1000
        expected_pos = scale_factor * np.ones(3)
        self._obj.scale_all(scale_factor)
        self.assertTrue(np.allclose(self._obj.get_position(), cg.Point(*expected_pos)), f"{self._obj.get_position()}")

    def test_negative_scaling(self):
        # negative values should raise an exception
        scale_fns = (getattr(self._obj, x) for x in ['scale', 'scale_x', 'scale_y', 'scale_z', 'scale_all'])
        for fn in scale_fns:
            with self.assertRaises(ValueError):
                fn(-1)

    def test_invalid_norm(self):
        # force a scale of zero and assert an error is raised
        scale_factor = 0
        self._obj.scale_all(scale_factor)
        with self.assertRaises(ValueError):
            self._obj.get_orientation()


class TestWorldObjectTranslation(WorldObjectTestCase):
    def test_3axis_movement(self):
        my_obj = cg.WorldObject()

        # We should be able to move the object multiple times and the position will move but not direction
        move_vector = np.array((1, 2, -5))
        my_obj.move(*move_vector)  # move the object in space
        self.assertTrue(np.array_equal(my_obj.get_position(), cg.Point(*move_vector)))

        # reversing the move gets you back to the origin
        my_obj.move(*(-move_vector))  # move the object in space
        self.assertTrue(np.array_equal(my_obj.get_position(), cg.Point()))

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
        self.assertTrue(np.array_equal(self._obj.get_position(), cg.Point(movement, movement, movement)))


class TestWorldObjectRotation(WorldObjectTestCase):
    def test_rotation(self):
        my_obj = self._obj

        # rotation about the y-axis by 90 degree should change the direction vector to x
        my_obj.rotate_y(90, units="deg")
        self.assertTrue(np.allclose(my_obj.get_orientation(), cg.Vector(1., 0, 0)))

        # now rotation it about the z-axis 90 degree should have it point to positive y
        my_obj.rotate_z(90, units="deg")
        self.assertTrue(np.allclose(my_obj.get_orientation(), cg.Vector(0, 1., 0)))

        # rotation 90 degree about the x-axis should reset it to positive z
        my_obj.rotate_x(90, units="deg")
        self.assertTrue(np.allclose(my_obj.get_orientation(), cg.Vector(0, 0, 1.)))

    def test_rotation_chain(self):
        # rotations should be able to be cascaded
        self._obj.rotate_y(90, units="deg").rotate_z(90, units="deg").rotate_x(90, units="deg")
        self.assertTrue(np.allclose(self._obj.get_orientation(), cg.Vector(0, 0, 1.)))

    def test_rotation_units(self):
        # rotations should work for radians and degrees
        rotation_angles = [90, np.pi / 2]
        rotation_units = ["deg", "rad"]
        for angle, unit in zip(rotation_angles, rotation_units):
            self._obj.rotate_y(angle, units=unit).rotate_z(angle, units=unit).rotate_x(angle, units=unit)
            self.assertTrue(np.allclose(self._obj.get_orientation(), cg.Vector(0, 0, 1.)),
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
        self.assertTrue(np.allclose(self.obj1.get_position(), cg.Point(scale, 0, 0)))
        self.assertTrue(np.allclose(self.obj2.get_position(), cg.Point(-scale, 0, 0)))

        # rotation also applies
        self.group.rotate_z(90)
        self.assertTrue(np.allclose(self.obj1.get_position(), cg.Point(0, scale, 0)))
        self.assertTrue(np.allclose(self.obj2.get_position(), cg.Point(0, -scale, 0)))

    def test_nesting_object_group(self):
        # make a subgroup and append it to the top level group
        subgroup = cg.ObjectGroup()

        sub_object = cg.WorldObject()
        sub_object.move(1, 0, 0)
        subgroup.append(sub_object)

        self.group.append(subgroup)

        x_movement = 3
        self.group.move_x(x_movement)  # move the top level group

        self.assertTrue(np.allclose(subgroup.get_position(), cg.Point(x_movement, 0, 0)))
        self.assertTrue(np.allclose(sub_object.get_position(), cg.Point(x_movement + 1, 0, 0)))


class TestElementWiseDotProduct(unittest.TestCase):
    def setUp(self):
        self.n_elements = 40
        self.major_axis_length = 10
        self.test_matrix = np.arange(self.n_elements).reshape(self.major_axis_length, -1)

    def test_simple_dot_product(self):
        dot = cg.element_wise_dot(np.arange(4), np.arange(4))
        self.assertEqual(dot, 14)

    def test_axis_zero_dot(self):
        dot = cg.element_wise_dot(self.test_matrix, self.test_matrix, axis=0)
        self.assertEqual(dot.shape[0], (self.test_matrix.shape[-1]))
        for n, element in enumerate(dot):
            expected = np.sum(np.arange(n, self.n_elements, self.n_elements / self.major_axis_length) ** 2)
            self.assertEqual(expected, element)

    def test_axis_one_dot(self):
        dot = cg.element_wise_dot(self.test_matrix, self.test_matrix, axis=1)
        self.assertEqual(dot.shape[0], (self.test_matrix.shape[0]))
        minor_axis = self.test_matrix.shape[-1]
        for n, element in enumerate(dot):
            expected = np.sum(np.arange(n * minor_axis, (n + 1) * minor_axis) ** 2)
            self.assertEqual(expected, element)


class TestSmallestPositiveRoot(unittest.TestCase):
    def test_root_dbl(self):
        # define roots
        a = 1
        b = -2
        c = 1
        root = cg.smallest_positive_root(a, b, c)
        self.assertEqual(root, 1)

        n_elements = 1000
        a = np.full(n_elements, 1)
        b = np.full(n_elements, -2)
        c = np.full(n_elements, 1)
        root = cg.smallest_positive_root(a, b, c)
        self.assertTrue(root.shape[0], n_elements)
        self.assertTrue(np.allclose(root, 1))

    def test_root_neg(self):
        # define roots
        a = 1
        b = 2
        c = 1
        root = cg.smallest_positive_root(a, b, c)
        self.assertEqual(root, np.inf)

        # arrayed case
        n_elements = 1000
        a = np.full(n_elements, 1)
        b = np.full(n_elements, 2)
        c = np.full(n_elements, 1)
        root = cg.smallest_positive_root(a, b, c)
        self.assertTrue(root.shape[0], n_elements)
        self.assertTrue(np.allclose(root, np.inf))

    def test_root_imag(self):
        # define roots
        a = 1
        b = 1
        c = 1
        root = cg.smallest_positive_root(a, b, c)
        self.assertEqual(root, np.inf)

        # arrayed case
        n_elements = 1000
        a = np.full(n_elements, 1)
        b = np.full(n_elements, 1)
        c = np.full(n_elements, 1)
        root = cg.smallest_positive_root(a, b, c)
        self.assertTrue(root.shape[0], n_elements)
        self.assertTrue(np.allclose(root, np.inf))

    def test__root_pos_and_neg(self):
        # define roots
        a = 1
        b = 0
        c = -4
        root = cg.smallest_positive_root(a, b, c)
        self.assertEqual(root, 2)

        # arrayed case
        n_elements = 1000
        a = np.full(n_elements, 1)
        b = np.full(n_elements, 0)
        c = np.full(n_elements, -4)
        root = cg.smallest_positive_root(a, b, c)
        self.assertTrue(root.shape[0], n_elements)
        self.assertTrue(np.allclose(root, 2))


class TestBinomialRoot(unittest.TestCase):
    def test_root_dbl(self):
        # define roots
        a = 1
        b = -2
        c = 1
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(roots, 1.0))

        n_elements = 1000
        a = np.full(n_elements, 1)
        b = np.full(n_elements, -2)
        c = np.full(n_elements, 1)
        roots = cg.binomial_root(a, b, c)
        self.assertEqual(roots.shape, (2, n_elements))
        self.assertTrue(np.allclose(roots, 1))

    def test_root_neg(self):
        # define roots
        a = 1
        b = 2
        c = 1
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(roots, -1.0))

        # arrayed case
        n_elements = 1000
        a = np.full(n_elements, 1)
        b = np.full(n_elements, 2)
        c = np.full(n_elements, 1)
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(roots, -1))

    def test_root_imag(self):
        # define roots
        a = 1
        b = 1
        c = 1
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(roots, np.inf))

    def test_root_pos_and_neg(self):
        # define roots
        a = 1
        b = 0
        c = -4
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(np.sort(roots, axis=0), np.atleast_2d(np.array((-2, 2))).T), f"got {roots}")

    def test_linear_roots(self):
        # define roots
        a = 0
        b = 2
        c = -4
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(roots, 2))

    def test_infinite_roots(self):
        # define roots
        a = 0
        b = 0
        c = -4
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(roots, np.inf))

    def test_mixed_roots(self):
        n_elements = 1000
        split = 500
        coeffs = np.zeros((3, n_elements))
        coeffs[:, :split] = np.tile((1, 0, -4), (split, 1)).T
        coeffs[:, split:] = np.tile((0, 0, 1), (split, 1)).T

        roots = np.sort(cg.binomial_root(*coeffs), axis=0)  # sort the roots so we can compare
        self.assertTrue(np.allclose(roots[0, :split], -2))
        self.assertTrue(np.allclose(roots[1, :split], 2))
        self.assertTrue(np.allclose(roots[0, split:], np.inf))
        self.assertTrue(np.allclose(roots[1, split:], np.inf))


class TestReflections(unittest.TestCase):
    def test_single_vector_reflection(self):
        vect_in = cg.Vector(1, -1, 0)
        normal = cg.Vector(0, 1, 0)
        reflection = cg.reflect(vect_in, normal)
        self.assertTrue(np.allclose(reflection, cg.Vector(1, 1, 0)), f"expected (1,1,0), got {reflection}")

        vect_in = cg.Vector(0, -1, 0)
        normal = cg.Vector(1, 1, 0) / np.sqrt(2)
        reflection = cg.reflect(vect_in, normal)
        self.assertTrue(np.allclose(reflection, cg.Vector(1, 0, 0), atol=1E-5), f"expected (1,0,0), got {reflection}")

    def test_single_normal_reflection(self):
        # make an array of a bunch of identical elements
        n_vects = 1000
        vect_in = np.zeros((4, n_vects))
        vect_in[0] = 1
        vect_in[1] = -1

        normal = cg.Vector(0, 1, 0)
        reflection = cg.reflect(vect_in, normal)
        self.assertTrue(np.allclose(reflection, np.tile(cg.Vector(1, 1, 0), (n_vects, 1)).T),
                        f"expected (1,1,0), got {reflection}")

    def test_multi_normal_reflection(self):
        # make an array of a bunch of identical elements
        n_vects = 1000
        vect_in = np.zeros((4, n_vects))
        vect_in[0] = 1
        vect_in[1] = -1

        normals = np.tile(cg.Vector(0, 1, 0), (n_vects, 1)).T
        reflection = cg.reflect(vect_in, normals)
        self.assertTrue(np.allclose(reflection, np.tile(cg.Vector(1, 1, 0), (n_vects, 1)).T),
                        f"expected (1,1,0), got {reflection}")


class TestRefraction(unittest.TestCase):
    def setUp(self) -> None:
        self.vector = cg.Vector(1, 1, 0).normalize()
        self.normal = cg.Vector(-1, 0, 0).normalize()

    def test_refraction_into_higher(self):
        n1 = 1
        n2 = 1.5
        refracted, index = cg.refract(self.vector, self.normal, n1, n2)

        # the ray should have refracted into the higher index
        self.assertEqual(index, n2)

        # the refracted vector should be closer to the normal and defined by snells law
        theta_2 = np.arcsin(n1 * np.sqrt(2) / (2 * n2))
        expected_vector = cg.Vector(np.cos(theta_2), np.sin(theta_2))
        self.assertTrue(np.allclose(refracted, expected_vector), f"Expected {expected_vector}, got {refracted}")

    def test_refraction_into_lower(self):
        n1 = 1.1
        n2 = 1.0
        refracted, index = cg.refract(self.vector, self.normal, n1, n2)

        # the ray should have refracted into the higher index
        self.assertEqual(index, n2)

        # the refracted vector should be closer to the normal and defined by snells law
        theta_2 = np.arcsin(n1 * np.sqrt(2) / (2 * n2))
        expected_vector = cg.Vector(np.cos(theta_2), np.sin(theta_2))
        self.assertTrue(np.allclose(refracted, expected_vector), f"Expected {expected_vector}, got {refracted}")

    def test_refraction_into_world(self):
        n1 = 1.5
        n2 = 1.5
        n_world = 1.4

        refracted, index = cg.refract(self.vector, -self.normal, n1, n2, n_world)

        # the ray should have refracted into the world index
        self.assertEqual(index, n_world)

        # the refracted vector should be closer to the normal and defined by snells law
        theta_2 = np.arcsin(n1 * np.sqrt(2) / (2 * n_world))
        expected_vector = cg.Vector(np.cos(theta_2), np.sin(theta_2))
        self.assertTrue(np.allclose(refracted, expected_vector), f"Expected {expected_vector}, got {refracted}")

    def test_total_internal_reflection(self):
        # testing case where ray exits surface but is TIR'd back into surface
        n1 = 1.5
        n2 = 1.5
        n_world = 1.0

        refracted, index = cg.refract(self.vector, -self.normal, n1, n2, n_world)
        # the ray should have refracted into the world index
        self.assertEqual(index, n1)

        expected_vector = cg.Vector(-1, 1).normalize()
        self.assertTrue(np.allclose(refracted, expected_vector), f"Expected {expected_vector}, got {refracted}")

        # testing case where ray exits surface but is TIR'd back into surface
        n1 = 1.5
        n2 = 1.0

        refracted, index = cg.refract(self.vector, self.normal, n1, n2, n_world)
        # the ray should have refracted into the world index
        self.assertEqual(index, n1)

        expected_vector = cg.Vector(-1, 1).normalize()
        self.assertTrue(np.allclose(refracted, expected_vector), f"Expected {expected_vector}, got {refracted}")

    def test_arrayed_refraction(self):
        n_elements = 1000
        split = int(n_elements / 2)
        n1_element = 1.5
        n2_element = 1.6
        n1 = np.full(n_elements, n1_element)
        n2 = np.full(n_elements, n2_element)
        n2[:split] = 1.0

        # make vectors traveling <1,1>
        vectors = np.zeros((4, n_elements))
        vectors[:2, :] = 1 / np.sqrt(2)

        normals = np.zeros((4, n_elements))
        normals[0] = -1

        refracted, index = cg.refract(vectors, normals, n1, n2)
        self.assertTrue(np.allclose(index[:split], n1_element))  # first bundle reflect
        self.assertTrue(np.allclose(index[split:], n2_element))  # second bundle refract

        expected_refracted = np.zeros((4, split))
        expected_refracted[0] = -1
        expected_refracted[1] = 1
        expected_refracted /= np.sqrt(2)
        self.assertTrue(np.allclose(refracted[:, :split], expected_refracted))

        expected_refracted = np.zeros((4, split))
        theta_2 = np.arcsin(n1_element * np.sqrt(2) / (2 * n2_element))

        expected_refracted[0] = np.cos(theta_2)
        expected_refracted[1] = np.sin(theta_2)
        self.assertTrue(np.allclose(refracted[:, split:], expected_refracted))


class TestSphere(unittest.TestCase):
    def setUp(self) -> None:
        self.sphere = cg.Sphere()
        self.ray = cg.Ray()

        self.intersection_points = ((0, 0, -1), (0, 0, 1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0))
        self.intersections = [cg.Point(*intersection) for intersection in self.intersection_points]

    def test_getting_radius(self):
        # default constructor should assign a radius of 1
        self.assertEqual(self.sphere.get_radius(), 1)

        # a new sphere can have the radius assigned
        self.assertEqual(cg.Sphere(3).get_radius(), 3)

    def test_ray_intersection_unit_sphere(self):
        hit = self.sphere.intersect(self.ray)
        self.assertEqual(hit.shape, (2, 1))

        # want hits to be -1 and 1 but dont' care about ordering
        self.assertTrue(1.0 in hit[:, 0])
        self.assertTrue(-1.0 in hit[:, 0], f"-1 not in {hit[:, 0]}")

        # if the ray is moved out of the radius of the sphere we get inf as the hit
        new_ray = cg.Ray()
        new_ray.origin = cg.Point(0, 0, 2)
        self.assertEqual(self.sphere.intersect(new_ray)[0, 0], np.inf)

    def test_intersection_sphere_behind_ray(self):
        ray_offset = 100
        ray = cg.Ray(cg.Point(ray_offset, 0, 0), cg.Vector(1, 0, 0))
        hits = self.sphere.intersect(ray)
        expected_hits = [-ray_offset + j * self.sphere.get_radius() for j in (-1, 1)]
        for hit in expected_hits:
            self.assertTrue(hit in hits[:, 0], f"{hit} was not found in {hits[:, 0]}")

    def test_multi_ray_intersection(self):
        n_rays = 100
        rays = cg.bundle_rays([cg.Ray() for _ in range(n_rays)])
        all_hits = self.sphere.intersect(rays)
        self.assertEqual(all_hits.shape, (2, n_rays))
        self.assertTrue(np.allclose(all_hits[0], self.sphere.get_radius()))
        self.assertTrue(np.allclose(all_hits[1], -self.sphere.get_radius()))

    def test_intersection_skew_case(self):
        hit = self.sphere.intersect(cg.Ray(cg.Point(0, 0, 2 * self.sphere.get_radius()), cg.Vector(1, 0, 0)))
        self.assertAlmostEqual(hit[0], np.inf)

    def test_double_intersection(self):
        hit = self.sphere.intersect(cg.Ray(origin=cg.Point(-1, 0, 1), direction=cg.Vector(1, 0, 0)))
        self.assertTrue(np.allclose(hit[:, 0], 1.0))

    def test_normals_base_sphere(self):
        # for a nontransformed sphere the normals should be vectors of the coordinates
        normals = [self.sphere.normal(intersection) for intersection in self.intersections]
        for normal, intersection in zip(normals, self.intersection_points):
            expected = cg.Vector(*intersection)
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
        self.surface = cg.Paraboloid(self.f)

    def test_object_getters(self):
        self.assertEqual(self.surface.get_focus(), self.f)

    def test_intersection_at_origin(self):
        hit = self.surface.intersect(cg.Ray(cg.Point(0, 0, 0), cg.Vector(0, 1, 0)))
        self.assertTrue(np.allclose(hit, 0), f"got hits {hit}")

        hit = self.surface.intersect(cg.Ray(cg.Point(0, 0, 0), cg.Vector(0, 0, 1)))
        self.assertTrue(np.allclose(hit, 0), f"got hits {hit}")

        hit = self.surface.intersect(cg.Ray(cg.Point(0, 0, 0), cg.Vector(1, 0, 0)))
        self.assertTrue(np.allclose(hit, 0), f"got hits {hit}")

    def test_intersection_linear_case(self):
        hit = self.surface.intersect(cg.Ray(cg.Point(-1, 0, 0), cg.Vector(1, 0, 0)))
        self.assertEqual(hit.shape, (2, 1))
        self.assertTrue(np.allclose(hit, 1))

        hit = self.surface.intersect(cg.Ray(cg.Point(0, 0, -2 * self.f), cg.Vector(1, 0, 0)))
        self.assertTrue(np.allclose(hit, self.f))

    def test_intersection_trivial_case(self):
        hit = self.surface.intersect(cg.Ray(cg.Point(0, 0, 0)))
        self.assertTrue(np.allclose(hit, 0))

        hit = self.surface.intersect(cg.Ray(cg.Point(0, 0, 0), cg.Vector(0, 1, 1) / np.sqrt(2)))
        self.assertTrue(np.allclose(hit, 0))

    def test_intersection_dbl_root_case(self):
        hit = self.surface.intersect(cg.Ray(cg.Point(self.f, -2 * self.f, 0), cg.Vector(0, 1, 0)))
        self.assertTrue(np.allclose(np.sort(hit[:, 0]), np.array((0, 4 * self.f))), f"{hit}")

        hit = self.surface.intersect(cg.Ray(cg.Point(self.f, 0, 0), cg.Vector(0, 1, 0)))
        self.assertTrue(np.allclose(np.sort(hit[:, 0]), np.array((-2, 2)) * self.f), f"{hit}")

    def test_intersection_skew_case(self):
        hit = self.surface.intersect(cg.Ray(cg.Point(-1, 0, 0), cg.Vector(0, 1, 0)))
        self.assertTrue(np.allclose(hit, np.inf))

        hit = self.surface.intersect(cg.Ray(cg.Point(-1, 0, 0), cg.Vector(0, 1, 1)))
        self.assertTrue(np.allclose(hit, np.inf))

        hit = self.surface.intersect(cg.Ray(cg.Point(-1, 0, 0), cg.Vector(0, 1, -1)))
        self.assertTrue(np.allclose(hit, np.inf))

    def test_intersection_arrayed_case(self):
        # make a bunch of rays to intersect, move some s.t. they intersect teh surface at a different point
        n_rays = 1000
        split_index = int(n_rays / 2)
        rays = cg.bundle_of_rays(n_rays)
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
        surface = cg.Paraboloid(-self.f)
        hit = surface.intersect(cg.Ray(cg.Point(-self.f, 0, 0), cg.Vector(0, 1, 0)))
        self.assertTrue(np.allclose(np.sort(hit[:, 0]), np.array((-2, 2)) * self.f), f"{hit}")

    def test_intersection_far(self):
        # check that an error is not raised when a grossly large value is passed to the intersection
        # TODO: decide a max distance to clip value at?
        hit = self.surface.intersect(cg.Ray(cg.Point(0, 0, 1000000000), cg.Vector(1.3, 0, 0)))

    def test_normal(self):
        normal = self.surface.normal(cg.Point(0, 0, 0))
        self.assertTrue(np.allclose(normal, cg.Vector(-1, 0, 0)))

        normal = self.surface.normal(cg.Point(self.f, 2 * self.f, 0))
        self.assertTrue(np.allclose(normal, cg.Vector(-1, 1, 0) / np.sqrt(2)))

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
        self.surface = cg.Plane()

    def test_positive_intersection(self):
        ray = cg.Ray(cg.Point(-1, 0, 0), cg.Vector(1, 0, 0))
        hit = self.surface.intersect(ray)[0, 0]
        self.assertAlmostEqual(hit, 1)

        ray = cg.Ray(cg.Point(-1, 0, 0), cg.Vector(1, 1, 0).normalize())
        hit = self.surface.intersect(ray)[0, 0]
        self.assertAlmostEqual(hit, np.sqrt(2))

    def test_negative_intersection(self):
        ray = cg.Ray(cg.Point(1, 0, 0), cg.Vector(1, 0, 0))
        hit = self.surface.intersect(ray)[0, 0]
        self.assertAlmostEqual(hit, -1)

    def test_parallel_intersection(self):
        ray = cg.Ray(cg.Point(1, 0, 0), cg.Vector(0, 1, 1).normalize())
        hit = self.surface.intersect(ray)[0, 0]
        self.assertAlmostEqual(hit, np.inf)

    def test_arrayed_intersection(self):
        n_rays = 1000
        split = int(n_rays / 2)

        # make a bunch of rays at x=-1, half will point towards the positive x-axis and the other will
        # point towards the positive y-axis
        rays = cg.bundle_of_rays(1000)
        rays[0, 0] = -1
        rays[1, 0, :split] = 1
        rays[1, 1] = 1

        hit = self.surface.intersect(rays)
        self.assertEqual(hit.shape, (1, n_rays), f"Ray shape is {hit.shape}")
        self.assertTrue(np.allclose(hit[0, :split], 1))
        self.assertTrue(np.allclose(hit[0, split:], np.inf))


class TestCube(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = cg.Cube()

    def test_intersection_within_cube(self):
        rays = (
            cg.Ray(direction=cg.Vector(1, 0, 0)),
            cg.Ray(direction=cg.Vector(0, 1, 0)),
            cg.Ray(direction=cg.Vector(0, 0, 1))
        )
        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[-1, 1]]).T), f"{hit}")

    def test_intersection_external_to_cube(self):
        rays = (
            cg.Ray(origin=cg.Point(-2, 0, 0), direction=cg.Vector(1, 0, 0)),
            cg.Ray(origin=cg.Point(0, -2, 0), direction=cg.Vector(0, 1, 0)),
            cg.Ray(origin=cg.Point(0, 0, -2), direction=cg.Vector(0, 0, 1))
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[1, 3]]).T), f"{hit}")

    def test_intersection_at_angle(self):
        ray = cg.Ray(origin=cg.Point(-2, -1, 0), direction=cg.Vector(1, 1, 0).normalize())

        hit = self.surface.intersect(ray)
        self.assertTrue(np.allclose(np.sort(hit, axis=0), np.sqrt(2) * np.array([[1, 2]]).T), f"{hit}")

    def test_skew_intersection(self):
        ray = cg.Ray(origin=cg.Point(-2, 0, 0), direction=cg.Vector(0, 1, 0).normalize())
        hit = self.surface.intersect(ray)
        self.assertTrue(np.allclose(hit, np.inf))

    def test_arrayed_intersection(self):
        # make a bunch of rays to intersect, move some s.t. they intersect teh surface at a different point
        n_rays = 1000
        split_index = int(n_rays / 2)
        rays = cg.bundle_of_rays(n_rays)
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
            cg.Point(-1, 0, 0),
            cg.Point(1, 0, 0),
            cg.Point(0, -1, 0),
            cg.Point(0, 1, 0),
            cg.Point(0, 0, -1),
            cg.Point(0, 0, 1)
        )
        for coord in coords:
            expected = cg.Vector(*coord[:-1])
            normal = self.surface.normal(coord)
            self.assertTrue(np.allclose(expected, normal), f"expected {expected}, got {normal}")

    def test_offcenter_normal(self):
        coord = cg.Point(-1 + 1E-8, 0.3, 0.7)

        expected = cg.Vector(-1, 0, 0)
        normal = self.surface.normal(coord)

        self.assertTrue(np.allclose(expected, normal), f"expected {expected}, got {normal}")

    def test_corner_normal(self):
        # for a corner case any of the three normals could be picked, but need to make sure the resulting normal is
        # in the right direction
        coord = cg.Point(1, 1, 1)

        # can be any of these and still be valid,
        expected = cg.Vector(1, 1, 1).normalize()
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
        self.surface = cg.Cylinder(1, infinite=True)

    def test_intersection_to_sidewalls(self):
        rays = (
            cg.Ray(origin=cg.Point(-2, 0, 0), direction=cg.Vector(1, 0, 0)),
            cg.Ray(origin=cg.Point(-2, 0, 1), direction=cg.Vector(1, 0, 0)),
            cg.Ray(origin=cg.Point(-2, 0, 2), direction=cg.Vector(1, 0, 0))
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[1, 3]]).T), f"{hit}")

    def test_no_intersection_inside(self):
        rays = (
            cg.Ray(origin=cg.Point(0, 0, 0), direction=cg.Vector(0, 0, 1)),
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
            cg.Ray(origin=cg.Point(2, 0, 0), direction=cg.Vector(0, 0, 1)),
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[np.inf, np.inf]]).T), f"{hit}")

    def test_arrayed_intersection(self):
        n_rays = 1000
        split = int(n_rays / 2)

        # make a bunch of rays at x=0, half will point towards the positive x-axis and the other will
        # point towards the positive y-axis
        rays = cg.bundle_of_rays(1000)
        rays[0, 0] = 0
        rays[1, 0, :split] = 1
        rays[1, 2, split:] = 1

        hit = self.surface.intersect(rays)
        self.assertEqual(hit.shape, (2, n_rays), f"Ray shape is {hit.shape}")
        self.assertTrue(np.allclose(hit[:, :split].T, np.array((-1, 1))))
        self.assertTrue(np.allclose(hit[:, split:].T, np.array((-np.inf, np.inf))))


class TestFiniteCylinder(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = cg.Cylinder(1, infinite=False)

    def test_intersection_to_sidewalls(self):
        rays = (
            cg.Ray(origin=cg.Point(-2, 0, 0), direction=cg.Vector(1, 0, 0)),
            cg.Ray(origin=cg.Point(-2, 0, 0.5), direction=cg.Vector(1, 0, 0)),
            cg.Ray(origin=cg.Point(-2, 0, -0.5), direction=cg.Vector(1, 0, 0))
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[1, 3]]).T), f"{hit}")

    def test_intersection_to_cap(self):
        rays = (
            cg.Ray(origin=cg.Point(0, 0, 0), direction=cg.Vector(0, 0, 1)),
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[-1, 1]]).T), f"{hit}")

    def test_wall_cap_intersection(self):
        rays = (
            cg.Ray(origin=cg.Point(-2, 0, -1), direction=cg.Vector(1, 0, 1)),
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
            cg.Ray(origin=cg.Point(2, 0, 0), direction=cg.Vector(0, 0, 1)),
        )

        for ray in rays:
            hit = self.surface.intersect(ray)
            self.assertTrue(np.allclose(np.sort(hit, axis=0), np.array([[np.inf, np.inf]]).T), f"{hit}")

    def test_arrayed_intersection(self):
        n_rays = 1000
        split = int(n_rays / 2)

        # make a bunch of rays at x=0, half will point towards the positive x-axis and the other will
        # point towards the positive y-axis
        rays = cg.bundle_of_rays(1000)
        rays[0, 0] = 0
        rays[1, 0, :split] = 1
        rays[1, 2, split:] = 1

        hit = self.surface.intersect(rays)
        self.assertEqual(hit.shape, (2, n_rays), f"Ray shape is {hit.shape}")
        self.assertTrue(np.allclose(hit[:, :split].T, np.array((-1, 1))))
        self.assertTrue(np.allclose(hit[:, split:].T, np.array((-1, 1))))


if __name__ == '__main__':
    unittest.main()
