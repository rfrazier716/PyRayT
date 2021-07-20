import numpy as np


def smallest_positive_root(a, b, c):
    """
    finds the 2nd order polynomial roots given the equations a*x^2 + b*x + c = 0, returns the smallest root > 0, and
    np.inf if a root does not satisfy that condition. Note that if a has a value of zero, it will be cast as 1 to avoid
    a divide by zero error. it's up the user to filter out these results before/after execution (using np.where etc.).

    :param a: coefficients for the second order of the polynomial
    :param b: coefficients for the first order of the polynomial
    :param c: coefficients for the zeroth order of the polynomial
    :return: an array of length n with the smallest positive root for the array
    """
    disc = b ** 2 - 4 * a * c  # find the discriminant
    root = np.sqrt(
        np.maximum(0, disc)
    )  # the square root of the discriminant protected from being nan
    polyroots = np.array(((-b + root), (-b - root))) / (
        2 * a + np.isclose(a, 0)
    )  # the positive element of the polynomial root

    # want to keep the smallest hit that is positive, so if hits[1]<0, just keep the positive hit
    nearest_hit = np.where(polyroots[1] >= 0, np.amin(polyroots, axis=0), polyroots[0])
    return np.where(np.logical_and(disc >= 0, nearest_hit >= 0), nearest_hit, np.inf)


def binomial_root(a, b, c, disc=None):
    """
    finds the 2nd order polynomial roots given the equations a*x^2 + b*x + c = 0

    :param a: coefficients for the second order of the polynomial
    :param b: coefficients for the first order of the polynomial
    :param c: coefficients for the zeroth order of the polynomial
    :param disc: the precalculated Discriminant array, if provided the discriminant won't be recalculated

    :return: an array of length n with the smallest positive root for the array
    """
    disc = (
        b ** 2 - 4 * a * c if disc is None else disc
    )  # find the discriminant, or use the provided one
    # linear cases are where there's no expontential term, have to be handled separately
    linear_cases = np.isclose(a, 0)
    root = np.sqrt(
        np.maximum(0, disc)
    )  # the square root of the discriminant protected from being nan

    # solve for the polynomial roots
    polyroots = np.vstack(((-b + root), (-b - root))) / (2 * a + np.isclose(a, 0))
    # Now correct for the cases that should be infinite
    polyroots = np.where(disc >= 0, polyroots, np.inf)

    # update for the linear roots case
    polyroots = np.where(linear_cases, np.tile(-c / (b + (b == 0)), (2, 1)), polyroots)

    # correct for cases that have neither a or b terms
    # if there's not a or b and the discriminant is positive, return +/-inf, otherwise just + inf
    # this is for cylinder intersections cases
    c_terms_only = np.logical_and(linear_cases, np.isclose(b, 0))
    polyroots = np.where(c_terms_only, np.inf, polyroots)
    polyroots[0] = np.where(np.logical_and(c_terms_only, c <= 0), -np.inf, polyroots[0])

    return polyroots


def element_wise_dot(mat_1, mat_2, axis=0):
    """
    calculates the row-wise/column-wise dot product two nxm matrices

    :param mat_1: the first matrix for the dot product
    :param mat_2: the second matrix for the dot product
    :param axis: axis to perform the dot product along, 0 or 1
    :return: a 1D array of length m for axis 0 and n for axis 1
    """
    # if a regular array was passed just do the dot product
    if mat_1.ndim == 1:
        return mat_1.dot(mat_2)

    einsum_string = "ij,ij->j"
    if axis == 1:
        einsum_string = "ij,ij->i"

    return np.einsum(einsum_string, mat_1, mat_2)


def reflect(vectors, normals):
    """
    reflects an array of vectors by a normal vector.

    :param vectors: a mxn array of vectors, where each column corresponds to one vector
    :param normals: a mxn array of unit-normal vectors or a 4x0 single normal vector. If only one normal is provided,
        every vector is reflected across that normal
    :return: an mxn array of reflected vector
    """
    # if we got 2x 1x4 arrays it's a basic case
    if vectors.ndim == 1 and normals.ndim == 1:
        return vectors - normals * 2 * vectors.dot(normals)

    # if only one normal was provided reflect every vector off of it
    elif normals.ndim == 1:
        dots = np.einsum("ij,i->j", vectors, normals)
        return vectors - 2 * np.tile(normals, (vectors.shape[1], 1)).T * dots

    # otherwise it's full blown matrix math
    else:
        dots = element_wise_dot(vectors, normals, axis=0)
        return vectors - 2 * normals * dots


def refract(vectors, normals, n1, n2, n_global=1):
    """
    calculates the refracted vector at an interface with index mismatch of n1 and n2. If the angle between the normal
        and the vector is  <90 degrees, the vector is exiting the medium, and the global refractive index value is used
        instead. this is analygous to a ray exiting a glass interface and entering air

    :param vectors: the ray transmission vectors in the current medium
    :param normals: the normals to the surface at the point of intersection with the ray
    :param n1: refractive index of the medium the ray is currently transmitting through
    :param n2: refractive index of the medium the ray is entering
    :param n_global: the world refractive index to use if the ray is exiting the medium

    :return: a 4xn array of homogeneous vectors representing the new transmission direction of the vectors after the
        medium
    """
    vectors /= np.linalg.norm(vectors, axis=0)
    cos_theta1_p = element_wise_dot(
        vectors, normals, axis=0
    )  # the angle dotted with the normals
    cos_theta1_n = element_wise_dot(
        vectors, -normals, axis=0
    )  # the vector dotted with the negative normals

    # anywhere that cos_theta1_p>0, we're exiting the medium and the n2 value should be updated with the global index
    n2_local = np.where(cos_theta1_p > 0, n_global, n2)
    normals = np.where(
        cos_theta1_p > 0, -normals, normals
    )  # update normals so they always are along ray direction
    r = n1 / n2_local  # the ratio of refractive indices

    # we want to keep the positive values only, which is the angle between the norm and the vector
    cos_theta1 = np.where(cos_theta1_p > 0, cos_theta1_p, cos_theta1_n)

    # see https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form for more details on using this
    radicand = 1 - (r ** 2) * (
        1 - cos_theta1 ** 2
    )  # radicand of the square root function
    cos_theta2 = np.sqrt(
        np.maximum(0, radicand)
    )  # pipe in a zero there so that we don't take the sqrt of a negative number

    # return the refracted or reflected ray depending on the radicand
    refracted = np.where(
        radicand > 0,
        r * vectors + (r * cos_theta1 - cos_theta2) * normals,
        vectors + 2 * cos_theta1 * normals,
    )
    refracted /= np.linalg.norm(refracted, axis=0)  # normalize the vectors

    n_refracted = np.where(
        radicand > 0, n2_local, n1
    )  # the refracted index is the original material if TIR
    return refracted, n_refracted
