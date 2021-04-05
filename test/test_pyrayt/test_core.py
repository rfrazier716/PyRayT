import unittest
import pyrayt
import tinygfx.g3d as cg
from pyrayt.components import LineOfRays

import numpy as np


class TestRaySet(unittest.TestCase):
    def setUp(self):
        self.n_rays = 1000
        self.set = pyrayt.RaySet(self.n_rays)

    def test_field_initialization(self):
        # rays and metadata should be separately indexable
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


class TestRayTrace(unittest.TestCase):
    def setUp(self) -> None:
        self.source = LineOfRays()
        self.surface = cg.XYPlane(material=pyrayt.materials.mirror).rotate_y(-90).move_x(3)
        self.tracer = pyrayt.RayTracer([self.source], [self.surface])

    def test_result_length(self):
        # with just one mirror the results should be 10 elements long
        self.tracer.set_rays_per_source(10)
        results = self.tracer.trace()
        self.assertEqual(results.shape[0], 10)  # rays should only intersect once then done

        # all rays should end at 3.0
        self.assertTrue(np.allclose(results['x1'], 3.0))

    def test_result_infinite_reflections(self):
        second_plane = cg.XYPlane(material=pyrayt.materials.mirror).rotate_y(90).move_x(-3)
        generation_limit = 10
        n_rays = 10

        self.tracer = pyrayt.RayTracer([self.source], [self.surface, second_plane], generation_limit=generation_limit)
        self.tracer.set_rays_per_source(n_rays)

        results = self.tracer.trace()
        self.assertEqual(results.shape[0], generation_limit * n_rays)  # rays should only intersect once then done

        # make sure the generation number went to 9
        self.assertEqual(set(results['generation']), set(range(10)))

    def test_multiple_sources(self):
        # if we pass in two sources we should generate number_of_rays for both sources
        self.tracer = pyrayt.RayTracer([LineOfRays(), LineOfRays()], [self.surface])
        rays_per_source = 10
        self.tracer.set_rays_per_source(rays_per_source)
        results = self.tracer.trace()
        self.assertEqual(results.shape[0], 2 * rays_per_source)  # rays should only intersect once then done

        # all rays should still end at 3.0
        self.assertTrue(np.allclose(results['x1'], 3.0))

    def test_passing_unarrayed_inputs(self):
        # if a single surface is applied the ray trace should pad it and treat it as normal
        self.tracer = pyrayt.RayTracer([self.source], self.surface)
        results = self.tracer.trace()
        self.tracer.set_rays_per_source(10)
        results = self.tracer.trace()
        self.assertEqual(results.shape[0], 10)  # rays should only intersect once then done

        # all rays should end at 3.0
        self.assertTrue(np.allclose(results['x1'], 3.0))

        # this should also hold if we pass an unarrayed source
        self.tracer = pyrayt.RayTracer(self.source, [self.surface])
        results = self.tracer.trace()
        self.tracer.set_rays_per_source(10)
        results = self.tracer.trace()
        self.assertEqual(results.shape[0], 10)  # rays should only intersect once then done

        # all rays should end at 3.0
        self.assertTrue(np.allclose(results['x1'], 3.0))



if __name__ == '__main__':
    unittest.main()
