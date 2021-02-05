import unittest
import abc
import adpd.packaging.surfaces as surf
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


class CommonSurfaceTests(object):

    @classmethod
    @abc.abstractmethod
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
            match = naming_re_pattern.match(surface.get_surface_id())
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


if __name__ == '__main__':
    unittest.main()
