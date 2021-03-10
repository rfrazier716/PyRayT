import unittest
import pyrayt.designer as designer
from tinygfx.g3d.primitives import WorldObject, ObjectGroup
import tinygfx.g3d as cg


class TestFlattenFn(unittest.TestCase):
    def test_flatten_system(self):
        system = []
        system.append(0)
        system.append((1, (2, 3)))
        system.append((4, 5, 6))
        system.append(7)
        flattened = designer.flatten(system)
        for n, field in enumerate(flattened):
            self.assertEqual(n, field)


class TestAnalyticSystem(unittest.TestCase):
    def setUp(self):
        self.system = designer.OpticalSystem()

    def test_calling_fields(self):
        group_fields = ("sources", "components", "detectors")
        for field in group_fields:
            attr = getattr(self.system, field)
            self.assertTrue(isinstance(attr, ObjectGroup))

        # have a bounding box which should be inherit from tracersurface
        self.assertTrue(isinstance(self.system.boundary, cg.Cuboid))

    def test_adding_to_system_fields(self):
        # we should be able to add to the dictionary field
        my_object = WorldObject()
        my_object.move(3, 0, 0)

        self.system.components.append(my_object)
        self.assertEqual(self.system.components[0], my_object)

    def test_default_values_unique(self):
        system1 = designer.OpticalSystem()
        system2 = designer.OpticalSystem()

        system1.boundary.move_x(3)
        self.assertEqual(system2.boundary.get_position()[0], 0)





if __name__ == '__main__':
    unittest.main()
