import unittest
import io
import adpd.packaging.designer as designer
import adpd.packaging.simple_cg as cg
import adpd.packaging.surfaces as surfaces

class TestSurface(surfaces.TracerSurface):

    def _create_optica_function_arguments(self):
        pass

    def create_optica_function(self):
        return self.get_label()


class TestOpticalSystem(unittest.TestCase):
    def setUp(self):
        self.system = designer.OpticalSystem()

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

    def test_optica_export(self):

        self.system["sources"].append(TestSurface('a'))
        sub_group = cg.ObjectGroup()
        sub_group.append(TestSurface())
        self.system["sources"].append(sub_group)
        self.system["components"].append(TestSurface())
        self.system["detectors"].append(TestSurface())
        string_stream = io.StringIO()
        self.system.to_optica(string_stream)

if __name__ == '__main__':
    unittest.main()
