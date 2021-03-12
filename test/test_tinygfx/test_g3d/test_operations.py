import unittest
import numpy as np
import tinygfx.g3d as cg
import tinygfx.g3d.primitives as primitives

class TestElementWiseDotProduct(unittest.TestCase):
    def setUp(self):
        self.n_elements = 40
        self.major_axis_length = 10
        self.test_matrix = np.arange(self.n_elements).reshape(self.major_axis_length, -1)

    def test_simple_dot_product(self):
        dot = cg.element_wise_dot(np.arange(4), np.arange(4))
        self.assertEqual(dot, 14)

    def test_axis_zero_dot(self):
        dot = cg.element_wise_dot(self.test_matrix, self.test_matrix, axis=0)
        self.assertEqual(dot.shape[0], (self.test_matrix.shape[-1]))
        for n, element in enumerate(dot):
            expected = np.sum(np.arange(n, self.n_elements, self.n_elements / self.major_axis_length) ** 2)
            self.assertEqual(expected, element)

    def test_axis_one_dot(self):
        dot = cg.element_wise_dot(self.test_matrix, self.test_matrix, axis=1)
        self.assertEqual(dot.shape[0], (self.test_matrix.shape[0]))
        minor_axis = self.test_matrix.shape[-1]
        for n, element in enumerate(dot):
            expected = np.sum(np.arange(n * minor_axis, (n + 1) * minor_axis) ** 2)
            self.assertEqual(expected, element)


class TestSmallestPositiveRoot(unittest.TestCase):
    def test_root_dbl(self):
        # define roots
        a = 1
        b = -2
        c = 1
        root = cg.smallest_positive_root(a, b, c)
        self.assertEqual(root, 1)

        n_elements = 1000
        a = np.full(n_elements, 1)
        b = np.full(n_elements, -2)
        c = np.full(n_elements, 1)
        root = cg.smallest_positive_root(a, b, c)
        self.assertTrue(root.shape[0], n_elements)
        self.assertTrue(np.allclose(root, 1))

    def test_root_neg(self):
        # define roots
        a = 1
        b = 2
        c = 1
        root = cg.smallest_positive_root(a, b, c)
        self.assertEqual(root, np.inf)

        # arrayed case
        n_elements = 1000
        a = np.full(n_elements, 1)
        b = np.full(n_elements, 2)
        c = np.full(n_elements, 1)
        root = cg.smallest_positive_root(a, b, c)
        self.assertTrue(root.shape[0], n_elements)
        self.assertTrue(np.allclose(root, np.inf))

    def test_root_imag(self):
        # define roots
        a = 1
        b = 1
        c = 1
        root = cg.smallest_positive_root(a, b, c)
        self.assertEqual(root, np.inf)

        # arrayed case
        n_elements = 1000
        a = np.full(n_elements, 1)
        b = np.full(n_elements, 1)
        c = np.full(n_elements, 1)
        root = cg.smallest_positive_root(a, b, c)
        self.assertTrue(root.shape[0], n_elements)
        self.assertTrue(np.allclose(root, np.inf))

    def test__root_pos_and_neg(self):
        # define roots
        a = 1
        b = 0
        c = -4
        root = cg.smallest_positive_root(a, b, c)
        self.assertEqual(root, 2)

        # arrayed case
        n_elements = 1000
        a = np.full(n_elements, 1)
        b = np.full(n_elements, 0)
        c = np.full(n_elements, -4)
        root = cg.smallest_positive_root(a, b, c)
        self.assertTrue(root.shape[0], n_elements)
        self.assertTrue(np.allclose(root, 2))


class TestBinomialRoot(unittest.TestCase):
    def test_root_dbl(self):
        # define roots
        a = 1
        b = -2
        c = 1
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(roots, 1.0))

        n_elements = 1000
        a = np.full(n_elements, 1)
        b = np.full(n_elements, -2)
        c = np.full(n_elements, 1)
        roots = cg.binomial_root(a, b, c)
        self.assertEqual(roots.shape, (2, n_elements))
        self.assertTrue(np.allclose(roots, 1))

    def test_root_neg(self):
        # define roots
        a = 1
        b = 2
        c = 1
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(roots, -1.0))

        # arrayed case
        n_elements = 1000
        a = np.full(n_elements, 1)
        b = np.full(n_elements, 2)
        c = np.full(n_elements, 1)
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(roots, -1))

    def test_root_imag(self):
        # define roots
        a = 1
        b = 1
        c = 1
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(roots, np.inf))

    def test_root_pos_and_neg(self):
        # define roots
        a = 1
        b = 0
        c = -4
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(np.sort(roots, axis=0), np.atleast_2d(np.array((-2, 2))).T), f"got {roots}")

    def test_linear_roots(self):
        # define roots
        a = 0
        b = 2
        c = -4
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(roots, 2))

    def test_infinite_roots(self):
        # define roots
        a = 0
        b = 0
        c = -4
        roots = cg.binomial_root(a, b, c)
        self.assertTrue(np.allclose(np.sort(roots.T), np.array((-np.inf, np.inf))))

    def test_mixed_roots(self):
        n_elements = 1000
        split = 500
        coeffs = np.zeros((3, n_elements))
        coeffs[:, :split] = np.tile((1, 0, -4), (split, 1)).T
        coeffs[:, split:] = np.tile((0, 0, 1), (split, 1)).T

        roots = np.sort(cg.binomial_root(*coeffs), axis=0)  # sort the roots so we can compare
        self.assertTrue(np.allclose(roots[0, :split], -2))
        self.assertTrue(np.allclose(roots[1, :split], 2))
        self.assertTrue(np.allclose(roots[0, split:], np.inf))
        self.assertTrue(np.allclose(roots[1, split:], np.inf))


class TestReflections(unittest.TestCase):
    def test_single_vector_reflection(self):
        vect_in = primitives.Vector(1, -1, 0)
        normal = primitives.Vector(0, 1, 0)
        reflection = cg.reflect(vect_in, normal)
        self.assertTrue(np.allclose(reflection, primitives.Vector(1, 1, 0)), f"expected (1,1,0), got {reflection}")

        vect_in = primitives.Vector(0, -1, 0)
        normal = primitives.Vector(1, 1, 0) / np.sqrt(2)
        reflection = cg.reflect(vect_in, normal)
        self.assertTrue(np.allclose(reflection, primitives.Vector(1, 0, 0), atol=1E-5), f"expected (1,0,0), got {reflection}")

    def test_single_normal_reflection(self):
        # make an array of a bunch of identical elements
        n_vects = 1000
        vect_in = np.zeros((4, n_vects))
        vect_in[0] = 1
        vect_in[1] = -1

        normal = primitives.Vector(0, 1, 0)
        reflection = cg.reflect(vect_in, normal)
        self.assertTrue(np.allclose(reflection, np.tile(primitives.Vector(1, 1, 0), (n_vects, 1)).T),
                        f"expected (1,1,0), got {reflection}")

    def test_multi_normal_reflection(self):
        # make an array of a bunch of identical elements
        n_vects = 1000
        vect_in = np.zeros((4, n_vects))
        vect_in[0] = 1
        vect_in[1] = -1

        normals = np.tile(primitives.Vector(0, 1, 0), (n_vects, 1)).T
        reflection = cg.reflect(vect_in, normals)
        self.assertTrue(np.allclose(reflection, np.tile(primitives.Vector(1, 1, 0), (n_vects, 1)).T),
                        f"expected (1,1,0), got {reflection}")


class TestRefraction(unittest.TestCase):
    def setUp(self) -> None:
        self.vector = primitives.Vector(1, 1, 0).normalize()
        self.normal = primitives.Vector(-1, 0, 0).normalize()

    def test_refraction_into_higher(self):
        n1 = 1
        n2 = 1.5
        refracted, index = cg.refract(self.vector, self.normal, n1, n2)

        # the ray should have refracted into the higher index
        self.assertEqual(index, n2)

        # the refracted vector should be closer to the normal and defined by snells law
        theta_2 = np.arcsin(n1 * np.sqrt(2) / (2 * n2))
        expected_vector = primitives.Vector(np.cos(theta_2), np.sin(theta_2))
        self.assertTrue(np.allclose(refracted, expected_vector), f"Expected {expected_vector}, got {refracted}")

    def test_refraction_into_lower(self):
        n1 = 1.1
        n2 = 1.0
        refracted, index = cg.refract(self.vector, self.normal, n1, n2)

        # the ray should have refracted into the higher index
        self.assertEqual(index, n2)

        # the refracted vector should be closer to the normal and defined by snells law
        theta_2 = np.arcsin(n1 * np.sqrt(2) / (2 * n2))
        expected_vector = primitives.Vector(np.cos(theta_2), np.sin(theta_2))
        self.assertTrue(np.allclose(refracted, expected_vector), f"Expected {expected_vector}, got {refracted}")

    def test_refraction_into_world(self):
        n1 = 1.5
        n2 = 1.5
        n_world = 1.4

        refracted, index = cg.refract(self.vector, -self.normal, n1, n2, n_world)

        # the ray should have refracted into the world index
        self.assertEqual(index, n_world)

        # the refracted vector should be closer to the normal and defined by snells law
        theta_2 = np.arcsin(n1 * np.sqrt(2) / (2 * n_world))
        expected_vector = primitives.Vector(np.cos(theta_2), np.sin(theta_2))
        self.assertTrue(np.allclose(refracted, expected_vector), f"Expected {expected_vector}, got {refracted}")

    def test_total_internal_reflection(self):
        # testing case where ray exits surface but is TIR'd back into surface
        n1 = 1.5
        n2 = 1.5
        n_world = 1.0

        refracted, index = cg.refract(self.vector, -self.normal, n1, n2, n_world)
        # the ray should have refracted into the world index
        self.assertEqual(index, n1)

        expected_vector = primitives.Vector(-1, 1).normalize()
        self.assertTrue(np.allclose(refracted, expected_vector), f"Expected {expected_vector}, got {refracted}")

        # testing case where ray exits surface but is TIR'd back into surface
        n1 = 1.5
        n2 = 1.0

        refracted, index = cg.refract(self.vector, self.normal, n1, n2, n_world)
        # the ray should have refracted into the world index
        self.assertEqual(index, n1)

        expected_vector = primitives.Vector(-1, 1).normalize()
        self.assertTrue(np.allclose(refracted, expected_vector), f"Expected {expected_vector}, got {refracted}")

    def test_arrayed_refraction(self):
        n_elements = 1000
        split = int(n_elements / 2)
        n1_element = 1.5
        n2_element = 1.6
        n1 = np.full(n_elements, n1_element)
        n2 = np.full(n_elements, n2_element)
        n2[:split] = 1.0

        # make vectors traveling <1,1>
        vectors = np.zeros((4, n_elements))
        vectors[:2, :] = 1 / np.sqrt(2)

        normals = np.zeros((4, n_elements))
        normals[0] = -1

        refracted, index = cg.refract(vectors, normals, n1, n2)
        self.assertTrue(np.allclose(index[:split], n1_element))  # first bundle reflect
        self.assertTrue(np.allclose(index[split:], n2_element))  # second bundle refract

        expected_refracted = np.zeros((4, split))
        expected_refracted[0] = -1
        expected_refracted[1] = 1
        expected_refracted /= np.sqrt(2)
        self.assertTrue(np.allclose(refracted[:, :split], expected_refracted))

        expected_refracted = np.zeros((4, split))
        theta_2 = np.arcsin(n1_element * np.sqrt(2) / (2 * n2_element))

        expected_refracted[0] = np.cos(theta_2)
        expected_refracted[1] = np.sin(theta_2)
        self.assertTrue(np.allclose(refracted[:, split:], expected_refracted))


if __name__ == '__main__':
    unittest.main()
