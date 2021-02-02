import unittest
import adi_optics.surfaces as surf


class TestSurfaces(unittest.TestCase):
    def test_something(self):
        my_window = surf.Window(1, name = "window")
        my_window.set_aperture_shape((3,3))
        print(my_window.create_optica_function())


if __name__ == '__main__':
    unittest.main()
