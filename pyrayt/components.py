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
        lens_arguments = {"aperture": 1, "material": matl.glass["ideal"]}
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
def thick_lens(r1: float, r2: float, thickness: float, **kwargs) -> cg.Intersectable:
    """
    Creates a Thick lens with arbitrary surface curvature.

    See https://en.wikipedia.org/wiki/Radius_of_curvature_(optics) for convention on positive vs. negative radius of curvature.


    :param r1: Radius of curvature for the first lens surface. Set to :code:`np.inf` for a planar surface.
    :type r1: float
    :param r2: Radius of curvature for the second lens surface. Set to :code:`np.inf` for a planar surface.
    :type r2: float
    :param thickness: Thickness of the lens along the optical (x) axis. For a biconvex lens this will be the thickest point, for a concave lens
        it will be the thinnest.
    :type thickness: float
    :return: A traceable lens centered at the origin. The returned lens is oriented such that the
        first surfaces faces the -X axis, and the second surface faces +X. The aperture is defined in the YZ Plane
    :rtype: cg.Intersectable
    """

    aperture_thickness, aperture_offset = _lens_full_thickness(
        r1, r2, thickness, kwargs.get("aperture")
    )

    # create the original lens
    lens = _create_aperture(kwargs.get("aperture"), aperture_thickness).move_z(
        aperture_offset / 2
    )
    lens.material = kwargs.get("material")

    # build the left side, if it's infinite leave it blank
    if np.isfinite(r1):
        left_side = cg.Sphere(r1, material=kwargs.get("material")).move_z(
            r1 - thickness / 2
        )
        # if it's concave, we cut out, convex, we intersect
        lens = (
            cg.csg.intersect(lens, left_side)
            if r1 > 0
            else cg.csg.difference(lens, left_side)
        )

    # build the right side, if it's infinite leave it blank
    if np.isfinite(r2):
        right_side = cg.Sphere(r2, material=kwargs.get("material")).move_z(
            r2 + thickness / 2
        )
        # if it's concave, we cut out, convex, we intersect
        lens = (
            cg.csg.intersect(lens, right_side)
            if r2 < 0
            else cg.csg.difference(lens, right_side)
        )

    return lens


def _lens_full_thickness(r1, r2, thickness, aperture) -> Tuple[float, float]:
    """Helper function to Create the aperture for a thick lens

    :param r1: [description]
    :type r1: [type]
    :param r2: [description]
    :type r2: [type]
    :param thickness: [description]
    :type thickness: [type]
    :param aperture: [description]
    :type aperture: [type]
    """
    if not hasattr(aperture, "__len__"):
        # if the aperture is a single value (circular) the max height is the radius
        max_height = aperture / 2

    else:
        # otherwise it's rectangular in which case the max height is the norm of the aperture
        max_height = np.linalg.norm(aperture) / 2

    # The left thickness is based on the radius of curvature, positive is concave, negative is convex
    left_thickness = thickness / 2
    if np.isfinite(r1) and r1 < 0:
        left_thickness += np.abs(r1) - np.sqrt(np.abs(r1) ** 2 - max_height ** 2)

    right_thickness = thickness / 2
    if np.isfinite(r2) and r2 > 0:
        right_thickness += np.abs(r2) - np.sqrt(np.abs(r2) ** 2 - max_height ** 2)

    center_shift = right_thickness - left_thickness
    total_thickness = right_thickness + left_thickness

    # return the total thickness of the aperture plus any shift along the x-axis
    return total_thickness, center_shift


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


def equilateral_prism(
    side_length: float,
    width: float,
    material: matl.TracableMaterial = matl.glass["BK7"],
) -> cg.csg.Intersectable:
    """Creates an equilateral prism.

    :param side_length: Side length of the triangular edges.
    :type side_length: float
    :param width: Width of the prism.
    :type width: float
    :param material: Prism material, defaults to matl.glass["BK7"]
    :type material: pyrayt.materials.TracableMaterial, optional
    :return: An equilateral prism with the body center located at the origin. The triangular faces are parallel to the YZ plane and the base of the prism is parallel to the XY plane.
    :rtype: cg.csg.Intersectable
    """
    # make the first cuboid which will remain after the corners are subtracted
    cut_length = (
        1.1 * side_length / np.sin(60 * np.pi / 180)
    )  # how long the cuts will need to be to remove the material from the cube\

    # create a prism by cutting away corners from the cube
    prism = cg.csg.difference(
        cg.csg.difference(
            cg.Cuboid.from_sides(side_length, width, side_length, material=material),
            cg.Cuboid.from_sides(cut_length, 1.1 * width, cut_length, material=material)
            .move(-cut_length / 2, 0, cut_length / 2)
            .rotate_y(30)
            .move(-side_length / 2, 0, -side_length / 2),
        ),
        cg.Cuboid.from_sides(cut_length, 1.1 * width, cut_length, material=material)
        .move(cut_length / 2, 0, cut_length / 2)
        .rotate_y(-30)
        .move(side_length / 2, 0, -side_length / 2),
    ).move_z(side_length / 2 * (1 - np.sin(60 * np.pi / 180)))
    return prism


def baffle(aperture: Union[float, Tuple[float, float]]) -> cg.Intersectable:
    """
    Creates a planar baffle that absorbs all intersecting rays.

    :param aperture: Aperture specification for the baffle. see :ref:`Specifying Apertures <Apertures>` for additional
        details.
    :return: a planar baffle centered at the origin, coplanar to the YZ Plane.
    """

    return cg.XYPlane(aperture[0], aperture[1], material=matl.absorber).rotate_y(90)


def aperture(
    size: Union[float, Tuple[float, float]],
    aperture_size: Union[float, Tuple[float, float]],
) -> cg.Intersectable:
    """Creates a planar baffle with a central opening specified by the :code:`aperture_size` argument. Rays that intersect the baffle are absorbed but will transmit through the apertured region.

    :param size: The size of the absorbing region of the aperture. See :ref:`Specifying Apertures <Apertures>` for additional details.
    :type size: float | Tuple[float, float]
    :param aperture_size: The size of the aperture opeining. See :ref:`Specifying Apertures <Apertures>` for additional details.
    :type aperture_size: Union[float, Tuple[float, float]]
    :return: A planar aperture with both baffle and opening centered at the origin, coplanar to the YZ plane.
    :rtype: cg.Intersectable
    """

    aperture_stop = baffle(size).rotate_y(-90)
    aperture = _create_aperture(aperture_size, thickness=0.1)

    return cg.csg.difference(aperture_stop, aperture).rotate_y(90).rotate_x(-90)


class Source(cg.WorldObject, abc.ABC):
    def __init__(self, wavelength=0.633, *args, **kwargs):
        """Base Class for all Sources

        :param wavelength: Wavelength of the source, defaults to 0.633 (units in um)
        :type wavelength: float, optional
        """
        super().__init__(*args, **kwargs)
        self._wavelength = wavelength

    def generate_rays(self, n_rays: int) -> pyrayt.RaySet:
        """Generates a :class:`~pyrayt.RaySet` based on the source parameters.

        :param n_rays: The number of rays to put in the resulting rayset
        :type n_rays: int
        :return: A set of rays whose position, direction, and wavelength are set based on the source type.
        :rtype: pyrayt.RaySet
        """
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


class CircleOfRays(Source):
    def __init__(self, diameter=1, wavelength=0.633, *args, **kwargs):
        """A Source that uniformly generates parallel rays about a circular arc.

        :param diameter: Diameter of the circle rays are generated about, defaults to 1
        :type diameter: int, optional
        :param wavelength: wavelength of the rays, defaults to 0.633
        :type wavelength: float, optional
        """
        super().__init__(wavelength, *args, **kwargs)
        self._diameter = diameter

    def _local_ray_generation(self, n_rays: int) -> pyrayt.RaySet:

        rayset = pyrayt.RaySet(n_rays)
        # if we want more than one ray, linearly space them, otherwise default position is fine
        theta = np.linspace(0, 2 * np.pi, n_rays)
        rayset.rays[0, 1] = (
            self._diameter / 2 * np.sin(theta)
        )  # space rays along the y-axis
        rayset.rays[0, 2] = (
            self._diameter / 2 * np.cos(theta)
        )  # space rays along the y-axis
        rayset.rays[1, 0] = 1  # direct rays along positive x
        rayset.wavelength = self._wavelength
        return rayset


class ConeOfRays(Source):
    def __init__(self, cone_angle: float, wavelength=0.633, *args, **kwargs):
        """Source that generates a set of rays originating from the same point, uniformly distributed about the x-axis with the specified cone angle.

        e.g: An un-rotated. source with a cone angle of 45 degrees will expand to a uniformly distributed circle with a radius of 10mm after traveling 10mm along the optical axis.

        :param cone_angle: Angle between every ray and the optical axis, in degrees.
        :type cone_angle: float
        :param wavelength: [description], defaults to 0.633
        :type wavelength: float, optional
        """
        super().__init__(wavelength, *args, **kwargs)
        self._angle = cone_angle * np.pi / 180.0

    def _local_ray_generation(self, n_rays: int) -> pyrayt.RaySet:
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


class WedgeOfRays(Source):
    def __init__(self, angle: float, wavelength=0.633, *args, **kwargs):
        """Source that generates a wedge of rays originating from a point, directed towards the positive x-axis along the y-axis. The angle of the rays is uniformly distributed between [-angle/2, angle/2].

        :param angle: The full angle of the wedge source.
        :type angle: float
        :param wavelength: Wavelength of the source, defaults to 0.633
        :type wavelength: float, optional
        """
        super().__init__(wavelength, *args, **kwargs)
        self._angle = angle * np.pi / 180.0

    def _local_ray_generation(self, n_rays: int) -> pyrayt.RaySet:

        rayset = pyrayt.RaySet(n_rays)

        # generate the wedge angles
        angles = np.linspace(-self._angle / 2, self._angle / 2, n_rays)

        # tilt rays along the wedge
        rayset.rays[1, 0] = np.cos(angles)  # x-dim
        rayset.rays[1, 1] = np.sin(angles)  # y-dim

        # the position in the x-direction is the cosine of the ray angle
        rayset.wavelength = self._wavelength
        return rayset


class Lamp(Source):
    def __init__(
        self, width: float, length: float, max_angle: float = 90, *args, **kwargs
    ) -> None:
        """Source that generates a lambertian distribution of rays. Every ray originates with a random position and direction about the surface of the lamp. The intensity of the distribution follows `Lambert's Cosine Law <https://en.wikipedia.org/wiki/Lambert%27s_cosine_law>`_.


        :param width: Width of the rectangular region that rays can generate from
        :type width: float
        :param length: Length of the rectangular region that rays can generate from.
        :type length: float
        :param max_angle: The maximum angle a ray can generate projected along the x-axis, defaults to 90 (degrees)
        :type max_angle: float, optional
        """
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
    """Identical to a :class:`Lamp`, except the ray generation function is cached so the same set of rays is generated across multiple simulations. Useful for running Monte-Carlo models where the random noise of the source is larger than the resolution you're trying to capture at."""

    @lru_cache(10)
    def generate_rays(self, n_rays: int) -> pyrayt.RaySet:
        return super().generate_rays(n_rays)
