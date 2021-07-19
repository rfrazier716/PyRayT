import abc
from functools import lru_cache
from functools import wraps
from typing import Union, Tuple

import numpy as np

import pyrayt
import pyrayt.materials as matl
from tinygfx import g3d as cg


def _lens(func):
    """
    Decorator to add kwargs common to all lenses.

    :param func:
    :return:
    """

    @wraps(func)
    def wrapper_function(*args, **kwargs):
        lens_arguments = {"aperture": 1, "material": matl.glass}
        # update default values with any supplied by the user
        lens_arguments.update(kwargs)
        return func(*args, **lens_arguments).rotate_y(90).rotate_x(90)

    return wrapper_function


def _create_aperture(aperture: Union[float, tuple], thickness):
    if not hasattr(aperture, "__len__"):
        # if a single value was passed, it's a circular aperture so a cylinder should be returned
        return cg.Cylinder(
            radius=aperture / 2, min_height=-thickness / 2, max_height=thickness / 2
        )

    elif aperture[0] > 0 and aperture[1] > 0:
        min_corner = (-aperture[0] / 2, -aperture[1] / 2, -thickness / 2)
        max_corner = (aperture[0] / 2, aperture[1] / 2, thickness / 2)
        return cg.Cuboid(min_corner, max_corner)

    elif aperture[0] < 0 and aperture[1] < 0:
        # if two values were passed it's elliptical
        aperture = cg.Cylinder(
            radius=np.abs(aperture[0]) / 2,
            min_height=-thickness / 2,
            max_height=thickness / 2,
        )
        aperture.scale_y(aperture[1] / aperture[0])

    else:
        raise TypeError(f"Could not deduce an aperture from {aperture}")


def _sphere_sample(n_pts: int, max_angle: float) -> np.ndarray:
    """
    Creates N random samples on the unit sphere where theta ranges from [0,max_angle] and phi ranges [0,2*pi]

    :param n_pts:
    :param max_angle: the max angle (in radians) of the theta value
    :return: a 2xn array of sampled angle values, the first row is theta, and the second is phi
    """
    # going to use inverse cdf sampling to generate theta and phi points
    uv_samples = np.random.random_sample((2, n_pts))

    # convert row 0 to theta samples
    uv_samples[0] = np.arccos(1 - uv_samples[0] * (1 - np.cos(max_angle)))
    uv_samples[1] *= 2 * np.pi  # convert row 1 to phi samples
    return uv_samples


@_lens
def biconvex_lens(r1: float, r2: float, thickness: float, **kwargs) -> cg.Intersectable:
    """
    Creates a thick lens with two convex surfaces.

    Any radii values less than zero will be treated as the absolute value of the input. passing :code:`math.inf` for
    either radius will make that surface planar.

    :param r1: Radius of curvature for the first lens surface.
    :param r2: Radius of curvature for the second lens surface.
    :param thickness: full thickness of the lens. If the thickness is set such that the two surface curvatures 'clip'
        into eachother inside of the aperture bounds, the resulting aperture will be set by the clip point.
    :param kwargs: Additional keyword arguments
    :return: A traceable lens centered at the origin. The returned lens is oriented such that the
        first surfaces faces the -X axis, and the second surface faces +X. The aperture is defined in the YZ Plane.
    """

    # create an aperture from the aperture arguments
    aperture_shape = _create_aperture(kwargs.get("aperture"), thickness)
    left_side = cg.Sphere(r2).move_z(r1 - thickness / 2)
    right_side = cg.Sphere(r1).move_z(-(r1 - thickness / 2))

    # assign materials from the kwargs
    material = kwargs.get("material")
    aperture_shape.material = material
    left_side.material = material
    right_side.material = material

    # perform csg to get the final lens
    lens = cg.csg.intersect(cg.csg.intersect(left_side, right_side), aperture_shape)

    # now we need to rotate the lens so it's orientated in the YZ plane
    return lens


@_lens
def plano_convex_lens(r: float, thickness: float, **kwargs) -> cg.csg.Intersectable:
    """
    Creates a thick lens with one convex surface and one planar surface.

    :param r: Radius of curvature for the spherical surface. A radius value less than zero will be treated as the
        absolute value of the input. passing :code:`math.inf` for the radius will make the second surface planar,
        resulting in a planar window.
    :param thickness: full thickness of the lens. If the thickness is set such that the two surface curvatures 'clip'
        into each other inside of the aperture bounds, the resulting aperture will be set by the clip point.
    :param kwargs: Additional keyword arguments
    :return: A traceable lens centered at the origin, oriented such that the planar surface faces the -X axis,
        and the spherical surface faces +X. The aperture is defined in the YZ Plane.
    """

    # create an aperture from the aperture arguments
    aperture = _create_aperture(kwargs.get("aperture"), thickness)
    right_side = cg.Sphere(r).move_z(-(r - thickness / 2))

    # assign materials from the kwargs
    material = kwargs.get("material")
    aperture.material = material
    right_side.material = material

    # perform csg to get the final lens
    lens = cg.csg.intersect(right_side, aperture)

    # now we need to rotate the lens so it's orientated in the YZ plane
    return lens


def _mirror(func):
    """
    Decorator to add kwargs common to all mirrors.

    :param func:
    :return:
    """

    @wraps(func)
    def wrapper_function(*args, **kwargs):
        lens_arguments = {"aperture": 1, "material": matl.mirror, "off_axis": (0, 0)}
        # update default values with any supplied by the user
        lens_arguments.update(kwargs)
        return func(*args, **lens_arguments).rotate_y(90).rotate_x(90)

    return wrapper_function


@_mirror
def plane_mirror(thickness: float, **kwargs) -> cg.Intersectable:
    """
    Creates a plane mirror where every side is reflective. If o

    :param thickness: The mirror thickness. A thickness of zero will result in a 2D planar mirror.
    :param kwargs: additional keyword arguments.
    :return: A mirror composed of two reflective surfaces, each of which is parallel to the YZ plane.
        The first surface is at x=-thickness, and the second surface is at x=0. The mirrors aperture is centered
        at the origin unless modified by off_axis.
    """
    off_axis = kwargs.get("off_axis")
    mirror = _create_aperture(kwargs.get("aperture"), thickness).move(
        *off_axis, 0
    )  # move the mirror to it's off axis pt
    mirror.material = kwargs.get("material")
    return mirror


@_mirror
def spherical_mirror(radius: float, thickness: float, **kwargs) -> cg.Intersectable:
    """
    Creates a spherical mirror defined by the surface function.

    .. math::
        x(y,z) = r-\\sqrt{r^2-(x^2+y^2)}

    Only the spherical surface is reflective, all sidewalls are absorbing.

    :param radius: radius of the spherical surface. must be > 0. A radius of :code:`math.inf` will result in a
        plane mirror.
    :param thickness: back thickness of the mirror. The mirror's aperture will be extended with an absorbing surface
        up to the plane defined by x=-thickness. If the resulting back-plane clips into the spherical surface, the
        surface will be cut off at the back-plane
    :param kwargs:
    :return: A spherical mirror with an absorbing backplane facing towards the -X axis, and a spherical surface facing
        towards +X. The focal point of the mirror is at (r/2, 0, 0).
    """

    off_axis = kwargs.get("off_axis")
    material = kwargs.get("material")
    aperture = kwargs.get("aperture")

    # need to calculate aperture thickness based on the off-axis value
    l = np.sqrt(off_axis[0] ** 2 + off_axis[1] ** 2)
    if hasattr(aperture, "__len__"):
        # if it's a rectangular aperture the dl is the norm of he rectangle
        dl = np.linalg.norm(aperture) / 2
    else:
        # otherwise if it's circular it's just the radius
        dl = aperture / 2

    aperture_front_thickness = abs(radius) - np.sqrt(radius ** 2 - (l + dl) ** 2)
    total_thickness = (
        aperture_front_thickness + thickness
    )  # the total aperture thickness

    aperture = _create_aperture(
        kwargs.get("aperture"), thickness + aperture_front_thickness
    )
    aperture.material = matl.absorber
    aperture.move(*off_axis, 0)

    # a radius >0 puts the curvature on the right side of the lens
    if radius > 0:
        mirror_surface = cg.Sphere(radius, material=material).move_z(radius)
        aperture.move_z(total_thickness / 2 - thickness)
    elif radius < 0:
        mirror_surface = cg.Sphere(abs(radius), material=material).move_z(radius)
        aperture.move_z(thickness - total_thickness / 2)
    mirror = cg.csg.difference(aperture, mirror_surface)
    return mirror


# @_mirror
# def elliptical_mirror(major_radius: float, minor_radius: float, thickness: float, **kwargs) -> cg.csg.CSGSurface:
#     """
#     :param major_radius:
#     :param minor_radius:
#     :param thickness:
#     :param kwargs:
#     :return:
#     """
#     off_axis = kwargs.get('off_axis')
#     material = kwargs.get('material')
#     aperture = kwargs.get('aperture')
#     aperture_thickness = thickness + minor_radius
#
#     aperture = _create_aperture(aperture, aperture_thickness)
#     aperture.material = matl.absorber
#     aperture.move(*off_axis, 0)
#     aperture.move_z(minor_radius / 2 - thickness)
#
#     mirror_surface = cg.Sphere(minor_radius, material=material)
#     mirror_surface.scale_y(major_radius / minor_radius)
#     mirror_surface.move_z(minor_radius)
#     mirror = cg.csg.difference(aperture, mirror_surface)
#     return mirror


@_mirror
def parabolic_mirror(focus: float, thickness: float, **kwargs) -> cg.csg.Intersectable:
    """
    Creates a parabolic mirror defined by the below surface.

    .. math::
        x(y,z)=\\frac{1}{4f} (y^2+z^2) - f

    Only the parabolic surface is reflective, and all sidewalls are absorbing.

    :param focus: focal length of the parabola
    :param thickness: Back thickness of the mirror. The parabola will be extended with an absorbing surface that extends
        until x=-(focus + thickness)
    :param kwargs: additional keyword arguments
    :return: A Parabolic mirror whose focus is on the origin.
    """

    off_axis = kwargs.get("off_axis")
    material = kwargs.get("material")
    aperture = kwargs.get("aperture")

    # need to calculate aperture thickness based on the off-axis value
    if hasattr(aperture, "__len__"):
        # if it's a rectangular aperture the dl is the norm of he rectangle
        furthest_point = np.linalg.norm(
            np.abs(np.asarray(off_axis)) + np.asarray(aperture) / 2
        )
    else:
        # otherwise if it's circular it's just the radius
        furthest_point = np.linalg.norm(np.asarray(off_axis)) + aperture

    front_thickness = (
        1 / (4 * focus) * furthest_point ** 2
    )  # the front thickness is the parabola value at the furthest point
    total_thickness = thickness + front_thickness

    aperture_shape = _create_aperture(aperture, total_thickness).move(*off_axis, 0)
    aperture_shape.material = matl.absorber  # assign the material
    aperture_shape.move_z(total_thickness / 2 - thickness)

    # now make the parabolic mirror, pad the height a bit, since it's slicing away from the aperture
    mirror_surface = cg.Paraboloid(
        focus, height=1.5 * front_thickness, material=material
    )
    mirror = cg.csg.difference(aperture_shape, mirror_surface)

    # center the focus off the mirror at the origin
    mirror.move_z(-focus)
    return mirror


def baffle(aperture: Union[float, Tuple[float, float]]) -> cg.Intersectable:
    """
    Creates a planar baffle that absorbs all intersecting rays.

    :param aperture: Aperture specification for the baffle. see :ref:`Specifying Apertures <Apertures>` for additional
        details.
    :return: a planar baffle centered at the origin, coplanar to the YZ Plane.
    """

    return cg.XYPlane(aperture[0], aperture[1], material=matl.absorber).rotate_y(90)


class Source(cg.WorldObject, abc.ABC):
    def __init__(self, wavelength=0.633, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wavelength = wavelength

    def generate_rays(self, n_rays: int) -> pyrayt.RaySet:
        ray_set = self._local_ray_generation(n_rays)
        ray_set.rays = np.matmul(
            self._world_coordinate_transform, ray_set.rays
        )  # transform rays to world space
        ray_set.rays[1] /= np.linalg.norm(
            ray_set.rays[1], axis=0
        )  # normalize the direction vector
        return ray_set

    @abc.abstractmethod
    def _local_ray_generation(self, n_rays: int) -> pyrayt.RaySet:
        pass

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value


class LineOfRays(Source):
    def __init__(self, spacing=1, wavelength=0.633, *args, **kwargs):
        super().__init__(wavelength, *args, **kwargs)
        self._spacing = spacing

    def _local_ray_generation(self, n_rays: int) -> pyrayt.RaySet:
        """
        creates a line of rays directed towards the positive x-axis along the y-axis

        :param n_rays:
        :return:
        """
        rayset = pyrayt.RaySet(n_rays)
        # if we want more than one ray, linearly space them, otherwise default position is fine
        if n_rays > 1:
            ray_position = np.linspace(-self._spacing / 2, self._spacing / 2, n_rays)
            rayset.rays[0, 1] = ray_position  # space rays along the y-axis
        rayset.rays[1, 0] = 1  # direct rays along positive x
        rayset.wavelength = self._wavelength
        return rayset


class ConeOfRays(Source):
    def __init__(self, cone_angle: float, wavelength=0.633, *args, **kwargs):
        super().__init__(wavelength, *args, **kwargs)
        self._angle = cone_angle * np.pi / 180.0

    def _local_ray_generation(self, n_rays: int) -> pyrayt.RaySet:
        """
        creates a line of rays directed towards the positive x-axis along the y-axis

        :param n_rays:
        :return:
        """
        rayset = pyrayt.RaySet(n_rays)
        # if we want more than one ray, change them to have the desired cone angle
        if n_rays > 1:
            angles = 2 * np.pi * np.arange(0, n_rays) / n_rays
            rayset.rays[1, 1] = np.sin(self._angle) * np.sin(angles)
            rayset.rays[1, 2] = np.sin(self._angle) * np.cos(angles)
        # the position in the x-direction is the cosine of the ray angle
        rayset.rays[1, 0] = np.cos(self._angle)
        rayset.wavelength = self._wavelength
        return rayset


class Lamp(Source):
    """
    a lamp
    """

    def __init__(
        self, width: float, length: float, max_angle: float = 90, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._max_angle = (
            max_angle * np.pi / 180
        )  # convert the max angle to radians internally
        self._width = width  # width is how far the lamp extends in the y-direction
        self._length = length  # length is how far the lamp extends in the z-direction

    def _local_ray_generation(self, n_rays: int) -> pyrayt.RaySet:
        rayset = pyrayt.RaySet(n_rays)
        rayset.wavelength = self._wavelength  # set the wavelength
        theta, phi = _sphere_sample(n_rays, self._max_angle)

        # randomly distribute the rays about the pd surface
        rayset.rays[0, 1] = self._width * (np.random.random_sample(n_rays) - 0.5)
        rayset.rays[0, 2] = self._length * (np.random.random_sample(n_rays) - 0.5)

        # orient the ray angles based on the theta/phi values
        rayset.rays[1, 0] = np.cos(theta)
        rayset.rays[1, 1] = np.sin(theta) * np.cos(phi)
        rayset.rays[1, 2] = np.sin(theta) * np.sin(phi)

        # set the intensity based on the theta value
        rayset.intensity = 100.0 * np.cos(theta)

        return rayset  # return the rotated rayset


class StaticLamp(Lamp):
    @lru_cache(10)
    def generate_rays(self, n_rays: int) -> pyrayt.RaySet:
        return super().generate_rays(n_rays)
