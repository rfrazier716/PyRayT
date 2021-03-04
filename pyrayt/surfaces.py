import pyrayt.simple_cg as cg
import abc
import numpy as np


class Aperture(cg.WorldObject):
    _shape: cg.Shape2D

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO

    def points_in_aperture(self, points):
        """
        compares if points are in the aperture
        :return:
        """
        # transform points into the aperture's coordinate system
        local_points = np.matmul(self._get_object_transform(), points)

        # the default object exists in the YZ Plane, so those are the coordinates sent to check for intersection
        return self._shape.point_in_shape(local_points[1:3])


class CircularAperture(Aperture):
    def __init__(self, radius, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        self._radius = radius
        self._shape = cg.Disk(radius)

    @property
    def radius(self):
        return self._radius


class EllipticalAperture(Aperture):
    def __init__(self, major_radius, minor_radius, *args, **kwargs):
        # an ellipse is just a disk that has had a scale matrix applied to it
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        self._radii = (major_radius, minor_radius)
        self._shape = cg.Disk(1)
        self.scale(1, major_radius, minor_radius)

    @property
    def radii(self):
        return self._radii


class RectangularAperture(Aperture):
    def __init__(self, y_length, z_length, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        self._side_lengths = (y_length, z_length)
        # the default shape is in the XY plane but aperture exist in the YZ Plane
        self._shape = cg.Rectangle(y_length, z_length)

    @property
    def side_lengths(self):
        return self._side_lengths


class SquareAperture(RectangularAperture):
    """
    Special case of a Rectangular Aperture
    """

    def __init__(self, side_length, *args, **kwargs):
        # initialize the parent constructor
        super().__init__(side_length, side_length, *args, **kwargs)


class TracerSurface(cg.WorldObject, abc.ABC):
    _aperture: Aperture = None  # initialize a new aperture for the surface

    def __init__(self, surface_args, material=None, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        self._surface_primitive = type(self).surface(*surface_args)  # create a surface primitive from the provided args
        self._material = material
        self._normal_scale = 1  # a multiplier used when normals are inverted

    def invert_normals(self):
        self._normal_scale = -1

    def reset_normals(self):
        self._normal_scale = 1

    def intersect(self, rays):
        """
        Intersect the set of rays with the surface, returns a 1d array of euclidean distances between the ray and
        surface. if the surface does not intersect -1 is returned instead
        :param rays:
        :return:
        """
        local_ray_set = np.matmul(self._get_object_transform(),
                                  np.atleast_3d(rays))  # translate the rays into object space

        # get the hits matrix, which is mxn where n is the number of rays propagated
        hits = self._surface_primitive.intersect(local_ray_set)
        if self._aperture is not None:
            hits = np.where(np.isfinite(hits), hits, -1)  # mask the hits so that anywhere it's np.inf it's cast as -1
            # if an aperture is composed with the object go through a second check to make sure the hits fall in that
            # aperture
            hit_points = local_ray_set[0] + hits * local_ray_set[1]
            # any hits that aren't in the aperture get masked with a -1
            hits = np.where(self._aperture.points_in_aperture(hit_points), hits, -1)

        else:
            hits = np.min(np.where(hits >= 0, hits, np.inf), axis=0)  # retuce the hits to a 1xn array of minimum hits
            hits = np.where(np.isfinite(hits), hits, -1)  # mask the hits so that anywhere it's np.inf it's cast as -1

        return hits

    def shade(self, rays, *shader_args):
        """
        propagates a set of rays incident on the surface. returns a new set of rays and refractive indices representing
        the rays after transmitting/reflecting off the surface.
        """

        # translate rays to object space
        return self._material.shade(self, rays, *shader_args)

    def get_world_normals(self, positions):
        """
        returns the normal vectors of the surface for the given positions

        :param positions: a 4xn array of homogeneous point coordinates.
        :return: a 4xn array of unit vectors representing the surface's normal vector at the corresponding input
            position
        """
        local_ray_set = self.to_object_coordinates(positions)
        local_normals = self._surface_primitive.normal(local_ray_set)
        world_normals = np.matmul(self._get_object_transform().T, local_normals)
        world_normals[-1] = 0  # wipe out any w fields caused by the transpose of the transform
        world_normals /= np.linalg.norm(world_normals, axis=0)
        return world_normals * self._normal_scale  # return the normals, flipped if the object has them inverted

    @property
    def aperture(self):
        return self._aperture

    @aperture.setter
    def aperture(self, value):
        if issubclass(type(value), Aperture):
            self._aperture = value
        else:
            raise ValueError("aperture type must be a subclass of 'Aperture'")


class Sphere(TracerSurface):
    surface = cg.Sphere

    def __init__(self, radius, material=None, *args, **kwargs):
        super().__init__(surface_args=(radius,), material=material, *args, **kwargs)


class Paraboloid(TracerSurface):
    surface = cg.Paraboloid

    def __init__(self, focus, material=None, *args, **kwargs):
        super().__init__(surface_args=(focus,), material=material, *args, **kwargs)


class YZPlane(TracerSurface):
    surface = cg.Plane

    def __init__(self, material=None, *args, **kwargs):
        super().__init__(surface_args=(), material=material, *args, **kwargs)


class Cuboid(TracerSurface):
    surface = cg.Cube

    def __init__(self, material=None, *args, **kwargs):
        super().__init__(surface_args=(), material=material, *args, **kwargs)

    @classmethod
    def from_lengths(cls, x=1, y=1, z=1):
        return cls().scale(x / 2, y / 2, z / 2)  # divide each by 2 because a regular cube extends from -1,1

    @classmethod
    def from_corners(cls, corner_0=np.array((-1, -1, -1)), corner_1=np.array((1, 1, 1))):
        corner_0 = np.asarray(corner_0)
        corner_1 = np.asarray(corner_1)
        if np.any(corner_1 < corner_0):
            raise ValueError("Second Corner must be greater than first corner at each dimension")

        center = 0.5 * (corner_0 + corner_1)
        scale_values = 0.5 * (corner_1 - corner_0)

        # return a cube object spanning those points
        return cls().scale(*scale_values[:3]).move(*center[:3])
