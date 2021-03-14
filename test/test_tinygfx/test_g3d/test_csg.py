import tinygfx.g3d as cg
import tinygfx.g3d.csg as csg
import numpy as np

import unittest


class TestCSGAdd(unittest.TestCase):
    def setUp(self) -> None:
        self.l_shape = cg.Sphere(1)
        self.r_shape = cg.Sphere(1)
        self.csg = csg.CSGSurface(self.l_shape, self.r_shape, csg.Operation.UNION)

    def test_bounding_box_updating(self):
        spans = self.csg.bounding_box.axis_spans
        expected_spans = np.array(((-1, -1, -1), (1, 1, 1))).T
        self.assertTrue(np.allclose(spans, expected_spans))

        # if we move a child surface the bounding box should update
        self.r_shape.move_x(1)
        spans = self.csg.bounding_box.axis_spans
        expected_spans = np.array(((-1, -1, -1), (2, 1, 1))).T
        self.assertTrue(np.allclose(spans, expected_spans))

    def test_intersection(self):
        n_rays = 10
        self.r_shape.move_y(-1)
        rays = cg.bundle_of_rays(n_rays)
        rays[1, 0] = 1  # point towards the right axis
        rays[0, 0] = -5
        rays[0, 1] = np.linspace(-2, 2, n_rays)

        hits = self.csg.intersect(rays)


    def test_moving_to_nonintersection(self):
        with self.assertRaises(ValueError):
            self.r_shape.move(4)


class TestCSGIntersect(unittest.TestCase):
    def setUp(self) -> None:
        self.l_shape = cg.Sphere(1)
        self.r_shape = cg.Sphere(1)
        self.csg = csg.CSGSurface(self.l_shape, self.r_shape, csg.Operation.INTERSECT)

    def test_bounding_box_updating(self):
        spans = self.csg.bounding_box.axis_spans
        expected_spans = np.array(((-1, -1, -1), (1, 1, 1))).T
        self.assertTrue(np.allclose(spans, expected_spans))

        # if we move a child surface the bounding box should update
        self.r_shape.move_x(3)
        spans = self.csg.bounding_box.axis_spans
        expected_spans = np.array(((-0, -1, -1), (1, 1, 1))).T
        self.assertTrue(np.allclose(spans, expected_spans), f"expected {expected_spans}, got {spans}")

    def test_moving_to_nonintersection(self):
        with self.assertRaises(ValueError):
            self.r_shape.move(4)


class TestArrayCSGOperation(unittest.TestCase):
    def setUp(self) -> None:
        self.array1 = np.array((0, 2, 4, 9, 11, 13))
        self.array2 = np.array((5, 7, 8, 9, 12, 14))

    def test_add_operation(self):
        unioned = csg.array_csg(self.array1, self.array2, csg.Operation.UNION)
        print(unioned)

    def test_diff_operation(self):
        diff = csg.array_csg(self.array1, self.array2, csg.Operation.DIFFERENCE)
        print(diff)

    def test_intersection_operation(self):
        intersect = csg.array_csg(self.array1, self.array2, csg.Operation.INTERSECT)
        print(intersect)


if __name__ == '__main__':
    unittest.main()
