import tinygfx.g3d as cg
import tinygfx.g3d.csg as csg
import numpy as np

import unittest


class TestCSGAdd(unittest.TestCase):
    def setUp(self) -> None:
        self.l_shape = cg.Sphere(1)
        self.r_shape = cg.Sphere(1)
        self.csg = cg.CSGSurface(self.l_shape, self.r_shape, cg.CSGSurface.operations.ADD)

    def test_bounding_box_updating(self):
        spans = self.csg.bounding_box.axis_spans
        expected_spans = np.array(((-1, -1, -1), (1, 1, 1))).T
        self.assertTrue(np.allclose(spans, expected_spans))

        # if we move a child surface the bounding box should update
        self.r_shape.move_x(3)
        spans = self.csg.bounding_box.axis_spans
        expected_spans = np.array(((-1, -1, -1), (4, 1, 1))).T
        self.assertTrue(np.allclose(spans, expected_spans))


class TestCSGIntersect(unittest.TestCase):
    def setUp(self) -> None:
        self.l_shape = cg.Sphere(1)
        self.r_shape = cg.Sphere(1)
        self.csg = cg.CSGSurface(self.l_shape, self.r_shape, cg.CSGSurface.operations.INT)

    def test_bounding_box_updating(self):
        spans = self.csg.bounding_box.axis_spans
        expected_spans = np.array(((-1, -1, -1), (1, 1, 1))).T
        self.assertTrue(np.allclose(spans, expected_spans))

        # if we move a child surface the bounding box should update
        self.r_shape.move_x(3)
        spans = self.csg.bounding_box.axis_spans
        expected_spans = np.array(((-0, -1, -1), (1, 1, 1))).T
        self.assertTrue(np.allclose(spans, expected_spans), f"expected {expected_spans}, got {spans}")


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
