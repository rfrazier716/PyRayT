import unittest

import pyrayt
from pyrayt.components.sources import LineOfRays
import tinygfx.g3d.world_objects as cg
import pyrayt.shaders.analytic as mat
import numpy as np


# TODO: Validate the renderer when the there' an intersection with an apertured surface before a second surface

class TestRenderReflection(unittest.TestCase):
    def setUp(self) -> None:
        self.system = pyrayt.OpticalSystem()
        self.sources = (LineOfRays().rotate_y(-45), LineOfRays().rotate_y(-135))
        self.surfaces = (cg.YZPlane(material=mat.mirror).move_x(-0.5),
                         cg.YZPlane(material=mat.mirror).move_x(0.5),
                         cg.YZPlane(material=mat.mirror).move_x(1))

        self.system.sources += self.sources
        self.system.components += self.surfaces
        self.renderer = pyrayt.RayTracer(self.system, generation_limit=20)

    def test_getters(self):
        self.renderer.set_generation_limit(1000)
        self.assertEqual(self.renderer.get_generation_limit(), 1000)

        new_system = pyrayt.OpticalSystem()
        new_system.sources.append("Hello World")
        self.renderer.load_system(new_system)
        self.assertEqual(self.renderer.get_system().sources[0], "Hello World")

    def test_render_results(self):
        rays_per_source = 5
        generation_limit = 10
        self.renderer.set_rays_per_source(rays_per_source)
        self.renderer.set_generation_limit(generation_limit)
        self.renderer.trace()
        results = self.renderer.get_results()

        # the resulting dataframe should have 2*5*10 (100) elements in it
        self.assertEqual(results.shape[0], 100)

        # all the x-values should be either -0.5 or 0.5 after the first 10
        self.assertTrue(np.allclose(np.abs(results['x0'][10:]), 0.5))
        self.assertTrue(np.allclose(np.abs(results['x1'][10:]), 0.5))

        # the y_values should increment by 1 each time except the first
        self.assertTrue(np.allclose((results['z1'] - results['z0'])[10:], 1.0))

        # the surface intersection should alternate between zero and 1
        surf0 = results['surface'] == 0
        self.assertTrue(np.allclose(results['x1'][surf0], -0.5))

        surf1 = results['surface'] == 1
        self.assertTrue(np.allclose(results['x1'][surf1], 0.5))

    def test_trimming_rays(self):
        # putting a plane at a height of 1.5 with absorbing should terminate rays early
        self.renderer.set_rays_per_source(1)  # only want one ray per source for this test
        top_baffle = cg.YZPlane(material=mat.absorber).rotate_y(90).move_z(1)
        self.system.components.append(top_baffle)
        results = self.renderer.trace()

        # verify that only four intersections were recorded and then the rays were killed
        self.assertEqual(results.shape[0], 4)

    def test_trimming_certain_rays(self):
        # make a ray that points right at the baffle and another that reflects infinitely
        # putting a plane at a height of 1.5 with absorbing should terminat rays early

        self.sources[0].rotate_y(45)  # rotate a source to point towards the positive x-axis
        self.sources[1].rotate_y(45)  # rotate the second to point straight up
        self.renderer.set_rays_per_source(1)  # only want one ray per source for this test
        self.renderer.set_generation_limit(10)

        top_baffle = cg.YZPlane(material=mat.absorber).rotate_y(90).move_z(1)
        self.system.components.append(top_baffle)
        results = self.renderer.trace()

        # verify that only four intersections were recorded and then the rays were killed
        self.assertEqual(results.shape[0], 11)

    def test_recording_no_intersection(self):
        # rotate a ray straight vertical so it does not have an intersection
        self.sources[1].rotate_y(45)  # rotate the second to point straight up
        self.renderer.set_generation_limit(10)
        self.renderer.set_rays_per_source(1)

        results = self.renderer.trace()
        self.assertEqual(results.shape[0], 11)

        # the first result should have a valid tilt but no distance
        dead_ray = results.iloc[1]
        coords = 'xyz'
        for coord in coords:
            self.assertEqual(dead_ray[coord + '0'], dead_ray[coord + '1'], f"failed for coordinate {coord}")

        self.assertAlmostEqual(dead_ray['x_tilt'], 0)
        self.assertAlmostEqual(dead_ray['y_tilt'], 0)
        self.assertAlmostEqual(dead_ray['z_tilt'], 1.)

    def test_saving_refractive_index_info(self):
        # when the rays propagate into a medium the refractive index should be updated
        system = pyrayt.OpticalSystem()
        sources = (LineOfRays().rotate_y(-90),)
        surfaces = [
            cg.Cuboid(material=mat.NKShader(material=mat.Material.REFRACTIVE, n=1.5)),
        ]
        surfaces[0].invert_normals()

        system.sources += sources
        system.components += surfaces
        renderer = pyrayt.RayTracer(system, rays_per_source=10, generation_limit=20)

        results = renderer.trace()
        self.assertEqual(results.shape[0], 20)
        self.assertTrue(np.allclose(results['index'][10:], 1.5))


class TestRenderRepeatedIntersection(unittest.TestCase):
    def setUp(self) -> None:
        self.system = pyrayt.OpticalSystem()
        self.sources = (LineOfRays().rotate_x(90).rotate_y(-180),)
        self.surfaces = (cg.Cuboid(material=mat.mirror),)

        self.system.sources += self.sources
        self.system.components += self.surfaces
        self.renderer = pyrayt.RayTracer(self.system, generation_limit=20)

    def test_render_results(self):
        """
        This test is here to check the floating point offset for repeated intersections to make sure it does not
        cascade and cause missed hits

        :return:
        """
        rays_per_source = 5
        generation_limit = 100
        self.renderer.set_rays_per_source(rays_per_source)
        self.renderer.set_generation_limit(generation_limit)
        self.renderer.trace()
        results = self.renderer.get_results()

        # rays should reflect back and forth until generation limit reached
        self.assertEqual(results.shape[0], rays_per_source * generation_limit)


if __name__ == '__main__':
    unittest.main()
