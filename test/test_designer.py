import unittest
import pyrayt.designer as designer
import pyrayt.simple_cg as cg


class TestFlattenFn(unittest.TestCase):
    def test_flatten_system(self):
        system = []
        system.append(0)
        system.append((1, (2, 3)))
        system.append((4, 5, 6))
        system.append(7)
        flattened = designer.flatten(system)
        for n,field in enumerate(flattened):
            self.assertEqual(n,field)

class TestAnalyticSystem(unittest.TestCase):
    def setUp(self):
        self.system = designer.AnalyticSystem()

    def test_calling_fields(self):
        fields = ("sources", "components", "detectors")
        for field in fields:
            attr = getattr(self.system, field)
            self.assertTrue(isinstance(attr, cg.ObjectGroup))

        print(self.system)

    def test_calling_dict_fields(self):
        # the three default fields should be created with the object
        default_keys = ("sources", "components", "detectors")
        system_keys = self.system.keys()
        for key in default_keys:
            self.assertTrue(key in system_keys)
            self.assertTrue(isinstance(self.system[key], cg.ObjectGroup))

    def test_adding_to_system_fields(self):
        # we should be able to add to the dictionary field
        my_object = cg.WorldObject()
        my_object.move(3, 0, 0)

        self.system["components"].append(my_object)
        self.assertEqual(self.system["components"][0], my_object)


if __name__ == '__main__':
    unittest.main()
