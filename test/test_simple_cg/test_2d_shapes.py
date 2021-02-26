import unittest
import pyrayt.simple_cg as cg
import numpy as np
import abc


class BaseShape(object):
    def test_in_surface_single(self):
        for point in self.points_in_surface:
            self.assertTrue(self.surface.point_in_shape(point), f"Failed with point {point}")

        for point in self.points_not_in_surface:
            self.assertFalse(self.surface.point_in_shape(point), f"Failed with point {point}")

    def test_in_surface_arrayed(self):
        self.assertTrue(np.all(
            self.surface.point_in_shape(self.points_in_surface.T))
            , f"Pointset failed {self.points_in_surface}")

        self.assertFalse(np.all(
            self.surface.point_in_shape(self.points_not_in_surface.T))
            , f"Pointset failed {self.points_not_in_surface}")


class TestDisk(BaseShape, unittest.TestCase):
    def setUp(self):
        self.surface = cg.Disk()
        self.points_in_surface = np.array([[0, 0], [0.7, 0.7], [0, -1], [-0.5, 0.7]])
        self.points_not_in_surface = np.array([[10, 10], [-20, 20], [0, 1.01], [2, 2]])
        

class TestRectangle(BaseShape, unittest.TestCase):
    def setUp(self):
        self.surface = cg.Rectangle()
        self.points_in_surface = np.array([[0, 0], [0.7, 0.7], [1, 1], [-0.5, 0.7]])
        self.points_not_in_surface = np.array([[1.01, 1], [-20, 20], [0, 1.01], [2, 2]])


if __name__ == '__main__':
    unittest.main()
