import unittest
import pyrayt.materials as materials


class TestAbsorbingMaterial(unittest.TestCase):
    def setUp(self) -> None:
        self.material = materials.absorber

    def test_something(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
