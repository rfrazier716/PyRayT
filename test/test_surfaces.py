import unittest
import abc
import adpd.packaging.surfaces as surf
import adpd.packaging.simple_cg as cg
import re
import numpy as np


class ImplementedTracerSurface(surf.TracerSurface):
    """
    Simply overwrites the tracer surface abstract method
    """

    def _create_optica_function_arguments(self):
        return "Surface", "{1.0, 1.0}"


class TestMathematicaConverters(unittest.TestCase):
    def test_dict_conversion(self):
        test_dict = dict(zip('abc', [1, 2, 3]))


class TestNamedObject(unittest.TestCase):
    def setUp(self):
        self.object = surf.NamedObject('MyObject')

    def test_getters(self):
        self.assertEqual(self.object.get_name(), 'MyObject')

    def test_string_repr(self):
        self.assertEqual(str(self.object), self.object.get_name())


class CommonSurfaceTests(object):

    @classmethod
    def create_new_surface(cls):
        return cls.surface_to_test(**cls.default_surface_args)

    @staticmethod
    def match_options_string(optica_string):
        patterns_re_string = r"[a-zA-Z]+[0-9]{4}OPTS = {((([a-zA-Z]+ -> .+?),[ ]+)*([a-zA-Z]+ -> .+?)?)};"
        match = re.match(patterns_re_string, optica_string)
        return match  # return the match containing the ABC -> aaa list

    @staticmethod
    def match_definitions_string(optica_string):
        definition_re_string = r"([a-zA-Z]+[0-9]{4}) = ([a-zA-Z]+)\[(.*),[ ]?([a-zA-Z]+[0-9]{4})OPTS];"
        match = re.match(definition_re_string, optica_string)
        return match  # return the match containing the ABC -> aaa list

    @staticmethod
    def match_movement_string(optica_string):
        movement_re_string = r"([a-zA-Z0-9]+) = Move\[([a-zA-Z]+)[0-9]{4},[ ]?({[0-9.]+,[0-9.]+,[0-9.]+}),[ ]?" \
                             r"({(?:{[0-9.]+,[0-9.]+,[0-9.]+}, ){2}{[0-9.]+,[0-9.]+,[0-9.]+}})];"
        match = re.match(movement_re_string, optica_string)
        return match

    def assertMovementFormatted(self, movement_string):
        # make sure that the movement string matches the required format
        match = self.match_movement_string(movement_string)
        self.assertTrue(match)

    def assertOptionsFormatted(self, options_string):
        match = self.match_options_string(options_string)
        self.assertTrue(match, f"Regex Did not find match with string \"{options_string}\"")

    def assertFunctionDefinitionFormatted(self, definition_string):
        match = self.match_definitions_string(definition_string)
        self.assertTrue(match,
                        f"Regex Did not find match with string \"{definition_string}\"")  # assert there was a match
        # assert the first and third match are the same (definition and options reference eachother
        self.assertEqual(match.group(1), match.group(4))
        self.assertEqual(match.group(2), type(self).optica_function_name)

    def test_naming_convention(self):
        """
        Require that the naming convention has a label#### format and only includes letters
        :return:
        """
        surfaces = [type(self).create_new_surface() for _ in range(20)]
        naming_re_pattern = re.compile(r"([a-zA-z]+)([0-9]{4})")
        for n, surface in enumerate(surfaces):
            match = naming_re_pattern.match(surface.get_label())
            self.assertTrue(bool(match))  # require that a match was made

            # check that the numbers are incrementing
            surface_number = int(match.group(2))
            if n == 0:
                prev_surface_number = surface_number - 1

            self.assertEqual(surface_number, prev_surface_number + 1,
                             f"expected surface number {prev_surface_number + 1} but got {surface_number}")

            prev_surface_number += 1

    def test_optica_string_format(self):
        surface = type(self).create_new_surface()

        optica_fn = surface.create_optica_function()
        str_options, str_function, str_movement = optica_fn.split('\n')
        self.assertOptionsFormatted(str_options)  # make sure the options string is formatted
        self.assertFunctionDefinitionFormatted(str_function)  # make sure the function definition complies
        self.assertMovementFormatted(str_movement)
        # There should be three lines in the expression, the options, the surface, and the movement

    def test_json_export(self):
        # make sure a default object can be successfully exported to json
        surface = type(self).create_new_surface()
        surface.to_json_string()


class ThickSurfaceTests(CommonSurfaceTests):
    def test_setting_and_exporting_thickess(self):
        # any thick surface object should have setter and getter methods for thickness
        # the thickness should be exported with the object parameters
        surface = type(self).create_new_surface()
        new_thickness = 3
        surface.set_thickness(new_thickness)
        self.assertEqual(surface.get_thickness(), new_thickness)
        obj_export = surface.collect_parameters()
        self.assertEqual(obj_export["thickness"], new_thickness)


class TestTracerSurface(unittest.TestCase, CommonSurfaceTests):
    default_surface_args = {"name": "mySurface"}
    surface_to_test = ImplementedTracerSurface
    optica_function_name = "Surface"

    def test_surface_parameter_export(self):
        print(surf.TracerSurface().__mro__())
        surface = type(self).create_new_surface()  # make a new surface object
        obj_params = surface.collect_parameters()  # get the parameters as an object
        # the name should be mySurface
        expected_keys = ('name', 'type', 'position', 'rotation', 'aperture')
        expected_values = ('mySurface', 'GenericSurface', (0, 0, 0), (0, 0, 0, 1.0), 1)
        for key, expected in zip(expected_keys, expected_values):
            self.assertEqual(obj_params[key], expected)

        # now move the surface and make sure key values are still updated
        surface.rotate_x(90)  # rotate by 90 degrees
        surface.move(1, 2, 4)  # move to 1,2,4
        surface.set_aperture_shape((3, 3))

        obj_params = surface.collect_parameters()  # get the parameters as an object
        expected_keys = ('position', 'rotation', 'aperture')
        expected_values = ((1, 2, 4), (np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2), (3, 3))
        for key, expected in zip(expected_keys, expected_values):
            # the tuples need to be compared as arrays
            if isinstance(expected, tuple):
                self.assertTrue(np.allclose(np.array(obj_params[key]), np.array(expected)))
            else:
                self.assertEqual(obj_params[key], expected)

    def test_adding_custom_parameters(self):
        # custom parameters should be dumped into the json string
        surface = type(self).create_new_surface()  # make a new surface object
        custom_key = "Chickens"
        custom_value = "Cute"
        surface.add_custom_parameter(custom_key, custom_value)
        obj_params = surface.collect_parameters()
        self.assertEqual(obj_params[custom_key], custom_value)

    def test_surface_parameter_export(self):
        surface = type(self).create_new_surface()  # make a new surface object
        surface.set_aperture_offset(3, 5)
        obj_params = surface.collect_parameters()  # get the parameters as an object
        self.assertEqual(obj_params["OffAxis"], (3, 5))


class TestBaffle(unittest.TestCase, ThickSurfaceTests):
    default_surface_args = {"thickness": 10, "name": "MyBaffle"}
    surface_to_test = surf.Baffle
    optica_function_name = "Baffle"


class TestWindow(unittest.TestCase, ThickSurfaceTests):
    default_surface_args = {"thickness": 10, "name": "MyWindow"}
    surface_to_test = surf.Window
    optica_function_name = "Window"


class TestSphere(unittest.TestCase):
    def setUp(self) -> None:
        self.sphere = surf.Sphere()
        self.ray = cg.Ray()

        self.intersection_points = ((0, 0, -1), (0, 0, 1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0))
        self.intersections = [cg.Point(*intersection) for intersection in self.intersection_points]

    def test_getting_radius(self):
        # default constructor should assign a radius of 1
        self.assertEqual(self.sphere.get_radius(), 1)

        # a new sphere can have the radius assigned
        self.assertEqual(surf.Sphere(3).get_radius(), 3)

    def test_ray_intersection_unit_sphere(self):
        hit = self.sphere.intersect(self.ray)
        self.assertEqual(hit.shape, (1,))
        self.assertAlmostEqual(hit[0], 1.)

        # if the ray is moved out of the radius of the sphere we get inf as the hit
        new_ray = cg.Ray()
        new_ray.origin = cg.Point(3, 0, 0)
        self.assertEqual(self.sphere.intersect(new_ray)[0], np.inf)

    def test_intersection_scaled_sphere(self):
        # if the sphere is scaled, the intersection should grow with the scaling
        scale_factor = 10
        self.sphere.scale_all(scale_factor)
        hit = self.sphere.intersect(self.ray)
        self.assertAlmostEqual(hit[0], scale_factor)

    def test_intersection_translated_sphere(self):
        movement = 10
        self.sphere.move_x(movement)
        hit = self.sphere.intersect(self.ray)
        self.assertAlmostEqual(hit[0], movement - self.sphere.get_radius())

    def test_intersection_sphere_behind_ray(self):
        self.sphere.move_x(-100)
        self.assertEqual(self.sphere.intersect(self.ray)[0], np.inf)

    def test_multi_ray_intersection(self):
        rays = cg.bundle_rays([cg.Ray() for _ in range(100)])
        all_hits = self.sphere.intersect(rays)
        self.assertTrue(np.allclose(all_hits[:], 1.))

    def test_normals_base_sphere(self):
        # for a nontransformed sphere the normals should be vectors of the coordinates
        normals = [self.sphere.normal(intersection) for intersection in self.intersections]
        for normal, intersection in zip(normals, self.intersection_points):
            expected = cg.Vector(*intersection)
            self.assertTrue(np.allclose(normal, expected))
            self.assertAlmostEqual(np.linalg.norm(normal), 1.0)

    def test_normals_scaled_sphere(self):
        # scaling a sphere should have no effect on the normals
        scaling = 5
        self.sphere.scale_all(scaling)
        scaled_intersection_points = ((0, 0, -5), (0, 0, 5), (0, 5, 0), (0, -5, 0), (5, 0, 0), (-5, 0, 0))
        self.intersections = [cg.Point(*intersection) for intersection in scaled_intersection_points]
        # for a nontransformed sphere the normals should be vectors of the coordinates
        normals = [self.sphere.normal(intersection) for intersection in self.intersections]
        for normal, intersection in zip(normals, self.intersection_points):
            expected = cg.Vector(*intersection)
            self.assertTrue(np.allclose(normal, expected))
            self.assertAlmostEqual(np.linalg.norm(normal), 1.0)

        # assert that the operation did not overwrite the world transform matrix
        self.assertTrue(np.allclose(self.sphere.get_world_transform()[:-1, :-1], np.identity(3) * scaling))

    def test_normals_rotated_sphere(self):
        # rotation should give the same normals
        z_rotation = 45
        self.sphere.rotate_z(45)

        normals = [self.sphere.normal(intersection) for intersection in self.intersections]
        for normal, intersection in zip(normals, self.intersection_points):
            expected = cg.Vector(*intersection)
            print(normal)
            self.assertTrue(np.allclose(normal, expected), f"Expected {normal}, got {expected}")
            self.assertAlmostEqual(np.linalg.norm(normal), 1.0)

        # assert that the operation did not overwrite the world transform matrix

    def test_normals_translated_sphere(self):
        translation = 10
        self.sphere.move_x(translation)
        translated_intersections = [intersection + np.array([translation,0,0,0]) for intersection in self.intersections]
        normals = [self.sphere.normal(intersection) for intersection in self.intersections]
        for normal, intersection in zip(normals, self.intersection_points):
            expected = cg.Vector(*intersection)
            print(normal)
            self.assertTrue(np.allclose(normal, expected), f"Expected {expected}, got {normal}")
            self.assertAlmostEqual(np.linalg.norm(normal), 1.0)

if __name__ == '__main__':
    unittest.main()
