import unittest
import pyrayt

import numpy as np


class TestRaySet(unittest.TestCase):
    def setUp(self):
        self.n_rays = 1000
        self.set = pyrayt.RaySet(self.n_rays)

    def test_field_initialization(self):
        self.assertEqual(self.set.rays.shape, (2, 4, self.n_rays))
        self.assertEqual(self.set.metadata.shape, (len(pyrayt.RaySet.fields), self.n_rays))

    def test_field_accessing_after_modifying_metadata(self):
        # makes sure that if you update the actual metadata contents, the fields reflect it
        for j in range(self.set.metadata.shape[0]):
            self.set.metadata[j] = j
            field_value = getattr(self.set, pyrayt.RaySet.fields[j])
            self.assertTrue(np.allclose(field_value, j),
                            f"Failed at index {j} with attribute {pyrayt.RaySet.fields[j]}")

    def test_metadata_accessing_after_modifying_fields(self):
        # makes sure that if you update the actual metadata contents, the fields reflect it
        for j in range(self.set.metadata.shape[0]):
            field = pyrayt.RaySet.fields[j]
            setattr(self.set, field, j)
            self.assertTrue(np.allclose(self.set.metadata[j], j))

    def test_updating_slices_of_fields(self):
        self.set.generation[:10] = 7
        self.assertTrue(np.allclose(self.set.metadata[0, :10], 7))

    def test_creation_from_concatenation(self):
        set1 = pyrayt.RaySet(10)
        set1.wavelength = -1
        set2 = pyrayt.RaySet(20)
        set2.wavelength = 2

        joined_set = pyrayt.RaySet.concat(set1, set2)

        self.assertEqual(joined_set.metadata.shape[-1], 30)
        self.assertEqual(joined_set.rays.shape, (2, 4, 30))

        self.assertTrue(np.allclose(joined_set.id, np.arange(30)), f"{joined_set.id}")
        self.assertTrue(np.allclose(joined_set.wavelength[:10], -1))
        self.assertTrue(np.allclose(joined_set.wavelength[10:], 2))


if __name__ == '__main__':
    unittest.main()