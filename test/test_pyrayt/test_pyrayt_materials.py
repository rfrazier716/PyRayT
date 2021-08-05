import unittest
from unittest.mock import MagicMock, patch
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


class TestRefractiveMaterial(unittest.TestCase):
    def setUp(self) -> None:
        self.index = 1.6
        self.material = materials.BasicRefractor(1.6)
        self.surface = cg.XYPlane()


    def test_index_is_updated_entering(self):
        rays = pyrayt.RaySet(10)
        rays.rays[1,2] = -1.

        # if we're entering the material the index should be updated
        new_rays = self.material.trace(self.surface, rays)
        self.assertTrue(np.allclose(new_rays.index, self.index))

    def test_index_updated_exiting(self):
        rays = pyrayt.RaySet(10)
        rays.rays[1, 2] = 1.
        rays.index = 20

        # if we're exiting the material the index should be set to 1
        new_rays = self.material.trace(self.surface, rays)
        self.assertTrue(np.allclose(new_rays.index, 1.0), new_rays.index[0])

    def test_refraction_occurs_incoming(self):
        rays = pyrayt.RaySet(10)
        # put the rays into the medium at 45 degrees
        rays.rays[1, 1]= 1
        rays.rays[1,2] = -1
        expected_angle = np.arcsin(np.sin(np.pi/4)/self.index)

        new_rays = self.material.trace(self.surface, rays)
        new_angle = np.arctan(np.abs(new_rays.rays[1,1]/new_rays.rays[1,2]))
        self.assertTrue(np.allclose(new_angle, expected_angle))

    def test_refraction_occurs_outgoing(self):
        rays = pyrayt.RaySet(10)
        # put the ray leaving the medium as almost normal incidence
        rays.rays[1, 1]= np.sin(0.1)
        rays.rays[1,2] = np.cos(0.1)
        rays.index = self.index

        expected_angle = np.arcsin(np.sin(0.1)*self.index)

        new_rays = self.material.trace(self.surface, rays)
        new_angle = np.arctan(np.abs(new_rays.rays[1,1]/new_rays.rays[1,2]))
        self.assertTrue(np.allclose(new_angle, expected_angle), f"expected {expected_angle}, got {new_angle[0]}")

    def test_total_internal_reflection(self):
        rays = pyrayt.RaySet(10)

        # put the ray leaving the medium at 45 degrees
        rays.rays[1, 1]= 1
        rays.rays[1,2] = 1
        rays.index = self.index

        expected_angle = np.pi/4

        new_rays = self.material.trace(self.surface, rays)
        new_angle = np.arctan(np.abs(new_rays.rays[1,1]/new_rays.rays[1,2]))
        self.assertTrue(np.allclose(new_angle, expected_angle), f"expected {expected_angle}, got {new_angle[0]}")
        self.assertTrue(np.allclose(rays.index, self.index))

class TestSellmeierRefractor(unittest.TestCase):

    def test_dispersive_index_single(self):
        coeffs = [
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1]
        ]
        for coeff in coeffs:
            material = materials.SellmeierRefractor(*coeff)
            self.assertAlmostEqual(material.index_at(2.0), np.sqrt(7/3))
    
    def test_dispersive_index_array(self):
        coeffs = [
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1]
        ]
        for coeff in coeffs:
            material = materials.SellmeierRefractor(*coeff)
            indices = material.index_at(np.full(100,2.0))
            expected = np.sqrt(7/3)
            self.assertTrue(np.allclose(indices, expected), f"expected {expected} but got {indices[:10]}")
    
    def test_refraction_at_normal_incidence(self):
        # make a ray set propogating along the -z axis (so that it enters the material)
        rays = pyrayt.RaySet(2)
        rays.wavelength[:] = 2.0
        rays.rays[1,2] = -1.0

        # Create a material and surface to interact with
        material = materials.SellmeierRefractor(b1=1, c1 = 1)
        surface = cg.XYPlane()

        # trace the rays through the material
        new_rays = material.trace(surface, rays)

        # assert that the index is updated
        self.assertTrue(np.allclose(new_rays.index, np.sqrt(7/3)), f"new rays have index {new_rays.index}, expected {np.sqrt(7/3)}")
    
    def test_refraction_at_angled_incidence(self):
        # make a ray set propagating along at a 45 degree angle
        rays = pyrayt.RaySet(10)
        rays.wavelength[:] = 2.0
        rays.rays[1,2] = -1.0
        rays.rays[1,1] = 1.0

        # Create a material and surface to interact with
        material = materials.SellmeierRefractor(b1=1, c1 = 1)
        surface = cg.XYPlane()

        # trace the rays through the material
        new_rays = material.trace(surface, rays)

        # assert that the index is updated
        new_angle = np.arctan(np.abs(new_rays.rays[1,1]/new_rays.rays[1,2]))
        expected_angle = np.arcsin(np.sqrt(3/7)*np.sqrt(2)/2)
        self.assertTrue(np.allclose(new_angle, expected_angle), f"new rays have angle {new_angle}, expected {expected_angle}")

if __name__ == '__main__':
    unittest.main()
