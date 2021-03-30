import tinygfx.g3d as cg
import pyrayt.materials as matl
from typing import Union


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
        return func(*args, **lens_arguments)

    return wrapper_function


def _create_aperture(aperture: Union[float, tuple], thickness):
    if not hasattr(aperture, '__len__'):
        # if a single value was passed, it's a circular aperture so a cylinder should be returned
        return cg.Cylinder(radius=aperture / 2, min_height=-thickness / 2, max_height=thickness / 2)

    elif aperture[0] > 0 and aperture[1] > 0:
        min_corner = (-aperture[0], -aperture[1], -thickness / 2)
        max_corner = (aperture[0], aperture[1], thickness / 2)
        return cg.Cuboid(min_corner, max_corner)

    elif aperture[0] < 0 and aperture[1] < 0:
        # if two values were passed it's elliptical
        aperture = cg.Cylinder(radius=aperture[0], min_height=-thickness / 2, max_height=thickness / 2)
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
    lens.rotate_y(90).rotate_x(90)
    return lens


@_lens
def plano_convex_lens(r: float, thickness: float, **kwargs):
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
    lens.rotate_y(90).rotate_x(90)
    return lens
