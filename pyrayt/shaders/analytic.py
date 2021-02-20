import enum
import pyrayt.simple_cg as cg
import pyrayt.surfaces as surf
import numpy as np


class Material(enum.Enum):
    REFRACTIVE = 0
    REFLECTIVE = 1
    ABSORBING = 2


class NKShader(object):
    """
    An analytic shader used for ray traces. Propogates rays through and optical system, with shade function returning a
        new set of rays and indices.

    :param material:
    :param n:
    :param k:
    :param args:
    :param kwargs:
    """

    def __init__(self, material=Material.REFLECTIVE, n=1, k=0, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        self._material = material
        self._n = n
        self._k = k

    def get_n(self, wavelength=0):
        """
        Return the refractive index of the material for a given wavelength

        :param wavelength:
        :return:
        """
        if callable(self._n):
            return self._n(wavelength)
        else:
            return self._n

    def get_k(self, wavelength=0):
        """
        Return the absorption constant of the material for a given wavelength

        :param wavelength:
        :return:
        """
        if callable(self._k):
            return self._k(wavelength)
        else:
            return self._k

    def shade(self, surface: surf.TracerSurface, rays, wavelengths, indices):
        """
        propagate the rays through the surface, returning a new set of rays and refractive indices

        :param surface:
        :param rays:
        :param wavelengths:
        :param indices:
        :return:
        """

        normals = surface.get_world_normals(rays[0])  # get the normal vectors from the surface

        # for reflective material return the reflective vectors
        if self._material == Material.REFLECTIVE:
            new_directions = cg.reflect(rays[1], normals)
            new_indices = indices

        elif self._material == Material.REFRACTIVE:
            new_directions, new_indices = cg.refract(rays[1], normals, indices, self.get_n(wavelengths))

        elif self._material == Material.ABSORBING:
            new_directions = np.zeros(rays[1].shape)  # if the material is absorbing all rays are terminated
            new_indices = indices

        else:
            raise ValueError(f"Shader does not know how to handle material {self._material}")

        return np.array((rays[0], new_directions)), new_indices


mirror = NKShader(Material.REFLECTIVE)
absorber = NKShader(Material.ABSORBING)