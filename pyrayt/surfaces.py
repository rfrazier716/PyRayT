import pyrayt.simple_cg as cg
import abc
import numpy as np


class _Aperture(object):
    def __init__(self, *args, **kwargs):
        self.shape = 0
        self.offset = cg.Point(0, 0, 0)


class TracerSurface(cg.WorldObject, abc.ABC):
    def __init__(self, surface_args, material=None, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        self._surface_primitive = type(self).surface(*surface_args)  # create a surface primitive from the provided args
        self._material = material
        self.aperture = _Aperture()  # initialize a new aperture for the surface

        self._normal_scale = 1 # a multiplier used when normals are inverted

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
        hits = self._surface_primitive.intersect(local_ray_set)
        return np.where(np.isfinite(hits), hits, -1)  # mask the hits so that anywhere it's np.inf it's cast as -1

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
        return world_normals*self._normal_scale  # return the normals, flipped if the object has them inverted


class Sphere(TracerSurface):
    surface = cg.Sphere

    def __init__(self, radius, material=None, *args, **kwargs):
        super().__init__(surface_args=(radius,), material=material, *args, **kwargs)


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
    def from_corners(cls, corner_0=np.array((-1,-1,-1)), corner_1=np.array((1,1,1))):
        corner_0 = np.asarray(corner_0)
        corner_1 = np.asarray(corner_1)
        if np.any(corner_1<corner_0):
            raise ValueError("Second Corner must be greater than first corner at each dimension")

        center = 0.5*(corner_0 + corner_1)
        scale_values = 0.5*(corner_1 - corner_0)

        # return a cube object spanning those points
        return cls().scale(*scale_values[:3]).move(*center[:3])

