import adpd.packaging.simple_cg as cg
import abc
import numpy as np

from collections import namedtuple


class
namedtuple("NKMaterial",["Type","n","k"],defaults = "")


class _Aperture(object):
    def __init__(self, *args, **kwargs):
        self.shape = 0
        self.offset = cg.Point(0, 0, 0)


class TracerSurface(cg.WorldObject, abc.ABC):
    def __init__(self, surface_args, material=None, *args, **kwargs):
        super().__init__(*args, **kwargs) # call the next constructor in the MRO
        self._surface_primitive = type(self).surface(*surface_args) # create a surface primitive from the provided args
        self._material = material
        self.aperture = _Aperture() # initialize a new aperture for the surface

    def intersect(self, rays):
        """
        Intersect the set of rays with the surface, returns a 1d array of euclidean distances between the ray and
        surface. if the surface does not intersect np.inf is returned instead
        :param rays:
        :return:
        """
        local_ray_set = np.matmul(self._get_object_transform(), np.atleast_3d(rays)) # translate the rays into object space
        hits = self._surface_primitive.intersect(local_ray_set)
        return hits

    def propagate(self, rays, refractive_indices, wavelengths):
        """
        propagates a set of rays incident on the surface. returns a new set of rays and refractive indices representing
        the rays after transmitting/reflecting off the surface.

        :param refractive_indices:
        :param rays:
        :return:
        """

        # translate rays to object space
        local_ray_set = np.matmul(self._get_object_transform(), np.atleast_3d(rays))
        normals = self._surface_primitive.normal(local_ray_set[0]) # get the normals at the intersection points
        if self._material.type == "Reflect":
            reflections = cg.reflect(rays[1], normals)
            return np.array([rays[0], reflections])

        # refract or reflect based on material
        # return results of refraction


class Sphere(TracerSurface):
    surface = cg.Sphere

    def __init__(self, radius, material=None, *args, **kwargs):
        super().__init__(surface_args=(radius,), material=material)



