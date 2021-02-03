import unittest
import adi_optics.surfaces as surf


class TestMathematicaConverters(unittest.TestCase):
    def test_dict_conversion(self):
        test_dict = dict(zip('abc',[1,2,3]))
        print(surf.to_mathematica_rules(test_dict))

class TracerSurfaceTestCase(unittest.TestCase):
    pass

class TestSurfaces(unittest.TestCase):
    def test_something(self):
        my_window = surf.Window(1, name = "MyWindow")
        my_window.set_material("Silicon")
        my_window.set_aperture_shape((3,3))
        my_window.set_aperture_offset(3,0)
        my_window.add_custom_parameter("RefractiveIndex",1.5)
        print(my_window.create_optica_function())


if __name__ == '__main__':
    unittest.main()
