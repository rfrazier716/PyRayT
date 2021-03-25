import unittest
import pyrayt.materials as materials
import pyrayt
import numpy as np
import tinygfx.g3d as cg
from copy import copy


class TestAbsorbingMaterial(unittest.TestCase):
    def setUp(self) -> None:
        self.material = materials.absorber
        self.surface = cg.XYPlane()

    def test_absorber_destroys_direction(self):
        rays = pyrayt.RaySet(10)
        new_rays = self.material.trace(self.surface, rays)

        # the original and new ray sets should be the same
        self.assertTrue(np.allclose(new_rays.rays[1], 0))
        self.assertTrue(np.allclose(rays.rays[1], 0))


class TestMirrorMaterial(unittest.TestCase):
    def setUp(self) -> None:
        self.material = materials.mirror
        self.surface = cg.XYPlane()

    def test_mirror_reflection_perpendicular(self):
        rays = pyrayt.RaySet(10)
        rays.rays[1,2] = 1.

        new_rays = self.material.trace(self.surface, rays)

        # the original and new ray sets should be the same
        self.assertTrue(np.allclose(new_rays.rays[1,2], -1))

    def test_mirror_reflection_at_angle(self):
        rays = pyrayt.RaySet(10)
        rays.rays[1,1:3] = 1.

        new_rays = self.material.trace(self.surface, rays)

        # the original and new ray sets should be the same
        self.assertTrue(np.allclose(new_rays.rays[1,2], -1))
        self.assertTrue(np.allclose(new_rays.rays[1,1], 1))


if __name__ == '__main__':
    unittest.main()
