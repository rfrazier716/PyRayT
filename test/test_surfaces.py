import unittest
import abc
import adi_optics.surfaces as surf
import re


class TestMathematicaConverters(unittest.TestCase):
    def test_dict_conversion(self):
        test_dict = dict(zip('abc', [1, 2, 3]))
        print(surf.to_mathematica_rules(test_dict))


class ImplementedTracerSurface(surf.TracerSurface):
    """
    Simply overwrites the tracer surface abstract method
    """

    def _create_optica_function_arguments(self):
        return "Surface", "{1.0, 1.0}"


class TracerSurfaceTestCase(unittest.TestCase):
    default_surface_args = {"name": "mySurface"}
    surface_to_test = ImplementedTracerSurface

    @classmethod
    @abc.abstractmethod
    def create_new_surface(cls):
        return cls.surface_to_test(**cls.default_surface_args)

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



if __name__ == '__main__':
    unittest.main()
