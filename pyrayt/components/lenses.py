import tinygfx.g3d as cg
import pyrayt.materials as matl
from typing import Union, Tuple
import numpy as np


def _lens(func):
    """
    Decorator to add kwargs common to all lenses.

    :param func:
    :return:
    """

    def wrapper_function(*args, **kwargs):
        lens_arguments = {
            'aperture': 1,
            'material': matl.glass
        }
        # update default values with any supplied by the user
        lens_arguments.update(kwargs)
        return func(*args, **lens_arguments).rotate_y(90).rotate_x(90)

    return wrapper_function


def _create_aperture(aperture: Union[float, tuple], thickness):
    if not hasattr(aperture, '__len__'):
        # if a single value was passed, it's a circular aperture so a cylinder should be returned
        return cg.Cylinder(radius=aperture / 2, min_height=-thickness / 2, max_height=thickness / 2)

    elif aperture[0] > 0 and aperture[1] > 0:
        min_corner = (-aperture[0] / 2, -aperture[1] / 2, -thickness / 2)
        max_corner = (aperture[0] / 2, aperture[1] / 2, thickness / 2)
        return cg.Cuboid(min_corner, max_corner)

    elif aperture[0] < 0 and aperture[1] < 0:
        # if two values were passed it's elliptical
        aperture = cg.Cylinder(radius=np.abs(aperture[0]) / 2, min_height=-thickness / 2, max_height=thickness / 2)
        aperture.scale_y(aperture[1] / aperture[0])

    else:
        raise TypeError(f"Could not deduce an aperture from {aperture}")


@_lens
def biconvex_lens(r1: float, r2: float, thickness: float, **kwargs):
    # create an aperture from the aperture arguments
    aperture_shape = _create_aperture(kwargs.get('aperture'), thickness)
    left_side = cg.Sphere(r2).move_z(r1 - thickness / 2)
    right_side = cg.Sphere(r1).move_z(-(r1 - thickness / 2))

    # assign materials from the kwargs
    material = kwargs.get('material')
    aperture_shape.material = material
    left_side.material = material
    right_side.material = material

    # perform csg to get the final lens
    lens = cg.csg.intersect(cg.csg.intersect(left_side, right_side),
                            aperture_shape)

    # now we need to rotate the lens so it's orientated in the YZ plane
    return lens


@_lens
def plano_convex_lens(r: float, thickness: float, **kwargs) -> cg.csg.CSGSurface:
    # create an aperture from the aperture arguments
    aperture = _create_aperture(kwargs.get('aperture'), thickness)
    right_side = cg.Sphere(r).move_z(-(r - thickness / 2))

    # assign materials from the kwargs
    material = kwargs.get('material')
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

    def wrapper_function(*args, **kwargs):
        lens_arguments = {
            'aperture': 1,
            'material': matl.mirror,
            'off_axis': (0, 0)
        }
        # update default values with any supplied by the user
        lens_arguments.update(kwargs)
        return func(*args, **lens_arguments).rotate_y(90).rotate_x(90)

    return wrapper_function


@_mirror
def plane_mirror(thickness: float, **kwargs) -> cg.csg.CSGSurface:
    off_axis = kwargs.get('off_axis')
    mirror = _create_aperture(kwargs.get('aperture'), thickness).move(*off_axis,
                                                                      0)  # move the mirror to it's off axis pt
    mirror.material = kwargs.get('material')
    return mirror


@_mirror
def spherical_mirror(radius: float, thickness: float, **kwargs) -> cg.csg.CSGSurface:
    off_axis = kwargs.get('off_axis')
    material = kwargs.get('material')
    aperture = kwargs.get('aperture')

    # need to calculate aperture thickness based on the off-axis value
    l = np.sqrt(off_axis[0] ** 2 + off_axis[1] ** 2)
    if hasattr(aperture, '__len__'):
        # if it's a rectangular aperture the dl is the norm of he rectangle
        dl = np.linalg.norm(aperture) / 2
    else:
        # otherwise if it's circular it's just the radius
        dl = aperture / 2

    aperture_front_thickness = abs(radius) - np.sqrt(radius ** 2 - (l + dl) ** 2)
    total_thickness = aperture_front_thickness + thickness  # the total aperture thickness

    aperture = _create_aperture(kwargs.get('aperture'), thickness + aperture_front_thickness)
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


@_mirror
def elliptical_mirror(major_radius: float, minor_radius: float, thickness: float, **kwargs) -> cg.csg.CSGSurface:
    off_axis = kwargs.get('off_axis')
    material = kwargs.get('material')
    aperture = kwargs.get('aperture')
    aperture_thickness = thickness + minor_radius

    aperture = _create_aperture(aperture, aperture_thickness)
    aperture.material = matl.absorber
    aperture.move(*off_axis, 0)
    aperture.move_z(minor_radius / 2 - thickness)

    mirror_surface = cg.Sphere(minor_radius, material=material)
    mirror_surface.scale_y(major_radius / minor_radius)
    mirror_surface.move_z(minor_radius)
    mirror = cg.csg.difference(aperture, mirror_surface)
    return mirror


@_mirror
def parabolic_mirror(focus: float, thickness: float, **kwargs) -> cg.csg.CSGSurface:
    off_axis = kwargs.get('off_axis')
    material = kwargs.get('material')
    aperture = kwargs.get('aperture')

    # need to calculate aperture thickness based on the off-axis value
    if hasattr(aperture, '__len__'):
        # if it's a rectangular aperture the dl is the norm of he rectangle
        furthest_point = np.linalg.norm(np.abs(np.asarray(off_axis)) + np.asarray(aperture) / 2)
    else:
        # otherwise if it's circular it's just the radius
        furthest_point = np.linalg.norm(np.asarray(off_axis)) + aperture

    front_thickness = 1 / (
                4 * focus) * furthest_point ** 2  # the front thickness is the parabola value at the furthest point
    total_thickness = thickness + front_thickness

    aperture_shape = _create_aperture(aperture, total_thickness).move(*off_axis,0)
    aperture_shape.material = matl.absorber  # assign the material
    aperture_shape.move_z(total_thickness / 2 - thickness)

    # now make the parabolic mirror, pad the height a bit, since it's slicing away from the aperture
    mirror_surface = cg.Paraboloid(focus, height=1.5 * front_thickness, material=material)
    mirror = cg.csg.difference(aperture_shape, mirror_surface)

    # center the focus off the mirror at the origin
    mirror.move_z(-focus)
    return mirror


def baffle(aperture: Tuple[float, float]):
    return cg.XYPlane(aperture[0], aperture[1], material=matl.absorber)
