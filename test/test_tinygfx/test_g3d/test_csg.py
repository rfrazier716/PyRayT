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
        self.r_shape.move_y(-1)
        spans = self.csg.bounding_box.axis_spans
        expected_spans = np.array(((-1, -2, -1), (1, 1, 1))).T
        self.assertTrue(np.allclose(spans, expected_spans), f"expected:\n{expected_spans}\ngot\n:{spans}")

    def test_intersection(self):
        self.r_shape.move_y(-1)  # move one shape to the left by 1

        n_rays = 11
        y_vals = np.linspace(-2, 2, n_rays)
        rays = cg.bundle_of_rays(n_rays)
        rays[1, 0] = 1  # point towards the right axis
        rays[0, 0] = -5
        rays[0, 1] = y_vals

        hits = self.csg.intersect(rays)

        # should only have two hits per ray, since it's the csg of quadratics
        self.assertTrue(np.all(np.isinf(hits[2:])))

        hit_missed = np.all(np.isinf(hits), axis=0)

        # the hits should miss at values <2 and >1
        self.assertFalse(np.any(hit_missed[np.logical_and(y_vals > -2, y_vals < 1)]))

        # hits should be sorted
        self.assertTrue(np.allclose(hits, np.sort(hits, axis=0)))

        # any hits <-0.5 should be identical to the right sphere hits
        r_sphere_hits = self.r_shape.intersect(rays)
        self.assertTrue(np.allclose(hits[:2, y_vals < -0.5], r_sphere_hits[:, y_vals < -0.5]))

        # any hits >-0.5 should be identical to the left sphere hits
        l_sphere_hits = self.l_shape.intersect(rays)
        self.assertTrue(np.allclose(hits[:2, y_vals > -0.5], l_sphere_hits[:, y_vals > -0.5]))

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

    def test_intersection(self):
        self.r_shape.move_y(-1)  # move one shape to the left by 1

        n_rays = 11
        y_vals = np.linspace(-2, 2, n_rays)
        rays = cg.bundle_of_rays(n_rays)
        rays[1, 0] = 1  # point towards the right axis
        rays[0, 0] = -5
        rays[0, 1] = y_vals

        hits = self.csg.intersect(rays)

        # should only have two hits per ray, since it's the csg of quadratics
        self.assertTrue(np.all(np.isinf(hits[2:])))

        hit_missed = np.all(np.isinf(hits), axis=0)

        # the hits should miss at values <-1 and >0
        self.assertFalse(np.any(hit_missed[np.logical_and(y_vals > -1, y_vals < 0)]))

        # hits should be sorted
        self.assertTrue(np.allclose(hits, np.sort(hits, axis=0)))

        r_sphere_hits = self.r_shape.intersect(rays)
        l_sphere_hits = self.l_shape.intersect(rays)

        # any hits <-0.5 should be identical to the left sphere hits
        self.assertTrue(np.allclose(hits[:2, y_vals < -0.5], l_sphere_hits[:, y_vals < -0.5]))

        # any hits >-0.5 should be identical to the right sphere hits
        self.assertTrue(np.allclose(hits[:2, y_vals > -0.5], r_sphere_hits[:, y_vals > -0.5]))

    def test_moving_to_nonintersection(self):
        with self.assertRaises(ValueError):
            self.r_shape.move(4)


class TestCSGDifference(unittest.TestCase):
    def setUp(self) -> None:
        self.l_shape = cg.Sphere(1)
        self.r_shape = cg.Sphere(1).move_y(-1)
        self.csg = csg.CSGSurface(self.l_shape, self.r_shape, csg.Operation.DIFFERENCE)

    def test_bounding_box_updating(self):
        spans = self.csg.bounding_box.axis_spans
        expected_spans = np.array(((-1, -1, -1), (1, 1, 1))).T
        self.assertTrue(np.allclose(spans, expected_spans), spans)

        # if we move the r shape the bounding box should not update
        self.r_shape.move_y(-1)
        spans = self.csg.bounding_box.axis_spans
        self.assertTrue(np.allclose(spans, expected_spans), f"expected:\n{expected_spans}\ngot\n:{spans}")

        # if we move the l shape the bounding box should follow
        self.l_shape.move_y(-1)
        expected_spans = np.array(((-1, -2, -1), (1, 0, 1))).T
        spans = self.csg.bounding_box.axis_spans
        self.assertTrue(np.allclose(spans, expected_spans), f"expected:\n{expected_spans}\ngot\n:{spans}")

    def test_intersection(self):
        n_rays = 101
        y_vals = np.linspace(-2, 2, n_rays)
        rays = cg.bundle_of_rays(n_rays)
        rays[1, 0] = 1  # point towards the right axis
        rays[0, 0] = -5
        rays[0, 1] = y_vals

        hits = self.csg.intersect(rays)

        # have any point <0 and >-0.5 should have 4 hits since the sphere has cut away from the surface
        self.assertTrue(np.all(np.isinf(hits[2:, y_vals > 0])))
        self.assertFalse(np.any(np.isinf(hits[2:, np.logical_and(y_vals < 0, y_vals > -0.5)])))

        hit_missed = np.all(np.isinf(hits), axis=0)

        # the hits should miss at values <-0.5 and >1
        self.assertTrue(np.all(hit_missed[np.logical_or(y_vals < -0.5, y_vals > 1)]))

        # hits should be sorted
        self.assertTrue(np.allclose(hits, np.sort(hits, axis=0)))

        # any hits >0 should be identical to the left sphere hits
        l_sphere_hits = self.l_shape.intersect(rays)
        r_sphere_hits = self.r_shape.intersect(rays)
        self.assertTrue(np.allclose(hits[:2, y_vals > 0], l_sphere_hits[:, y_vals > 0]))

        # for hits between -0.5 and 0 the outer hits should belong to the left sphere, and inner belong to the right
        # sphere
        subset = np.logical_and(y_vals > -0.5, y_vals < 0)
        self.assertTrue(np.allclose(hits[[[0],[3]], subset], l_sphere_hits[:, subset]), f"{l_sphere_hits[:,subset]}")
        self.assertTrue(np.allclose(hits[1:3, subset], r_sphere_hits[:, subset]), f"{r_sphere_hits[:,subset]}")


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
