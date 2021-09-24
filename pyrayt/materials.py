from typing import Union
import numpy as np
from pyrayt._pyrayt import RaySet
from tinygfx.g3d import materials as cg_matl
import tinygfx.g3d as cg
import abc
from functools import lru_cache


class TracableMaterial(cg_matl.gooch.Material):
    def __init__(self, base_material=cg_matl.gooch.BLACK, *args, **kwargs):
        """Base class for any material that can be traced with RayTracer objects

        :param base_material: The base material to render the object with. used by the tinygfx module.
        """
        super().__init__(*args, **kwargs)  # call the next constructor
        self._base_material = (
            base_material  # the material to call whenever the object is being rendered
        )

    def shade(
        self, rays: np.ndarray, normals: np.ndarray, light_positions: np.ndarray
    ) -> np.ndarray:
        return self._base_material.shade(rays, normals, light_positions)

    @abc.abstractmethod
    def trace(self, surface, ray_set: RaySet) -> RaySet:
        """Trace Function for the material. Calculates the interaction of a :class:`RaySet` with a material, modifying the ray set in place.

        :param surface: The surface the material is attached to, used to calculate surface normals
        :type surface: cg.TracerSurface
        :param ray_set: the set of rays whose interaction is being calculated
        :type ray_set: RaySet
        :return: a reference to the original ray_set with parameters updated to accurately represent the state of the ray after interacting with the surface.
        :rtype: RaySet
        """
        pass


class _AbsorbingMaterial(TracableMaterial):
    def __init__(self, *args, **kwargs):
        """An Ideal absorber. Any ray interacting with an absorber will have it's velocity vector set to <0,0,0>, which will signal to the RayTrace object to terminate the ray."""
        super().__init__(
            cg_matl.gooch.BLACK, *args, **kwargs
        )  # call the parent constructor

    def trace(self, surface, ray_set: RaySet) -> RaySet:
        # an absorbing material will kill the ray direction vector
        ray_set.rays[1] = 0
        return ray_set


class _ReflectingMaterial(TracableMaterial):
    def __init__(self, *args, **kwargs):
        """An Ideal reflector, Any ray interacting with this reflector will be reflected with no change to refractive index or intensity."""
        super().__init__(cg_matl.gooch.BLUE, *args, **kwargs)

    def trace(self, surface: cg.TracerSurface, ray_set: RaySet) -> RaySet:
        # a reflecting material will
        normals = surface.get_world_normals(ray_set.rays[0])
        ray_set.rays[1] = cg.reflect(ray_set.rays[1], normals)
        return ray_set


class Glass(TracableMaterial):
    def __init__(self, *args, **kwargs):
        """Abstract base class for glasses with convenience functions for common parameters."""
        super().__init__(base_material=cg_matl.gooch.BLUE, *args, **kwargs)

    def trace(self, surface: cg.TracerSurface, ray_set: RaySet) -> RaySet:
        normals = surface.get_world_normals(ray_set.rays[0])
        ray_set.rays[1], ray_set.index = cg.refract(
            ray_set.rays[1], normals, ray_set.index, self.index_at(ray_set.wavelength)
        )
        return ray_set

    @lru_cache(100)
    def abbe(self) -> float:
        """
        Calculates the `Abbe number <https://en.wikipedia.org/wiki/Abbe_number>`_ of the material.
        """
        n_short = self.index_at(0.4861)
        n_center = self.index_at(0.5893)
        n_long = self.index_at(0.6563)

        return (n_center - 1) / (n_short - n_long)

    @abc.abstractmethod
    def index_at(
        self, wavelength: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Calculates the refractive index of the glass for the given wavelength.

        :param wavelength: The wavelength to sample, can be either a single float or a numpy array. Expected units are microns.
        :type wavelength: Union[float, np.ndarray]
        :return: The refractive index of the material, the shape of the returned value will match the shape/type of the wavelength argument.
        :rtype: Union[float, np.ndarray]
        """
        pass


class BasicRefractor(Glass):
    def __init__(self, refractive_index: float, *args, **kwargs):
        """A simplified refractive model for nondispersive materials

        :param refractive_index: refractive index of the material.
        :type refractive_index: float
        """
        self._refractive_index = refractive_index
        super().__init__()

    def index_at(
        self, wavelength: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        if isinstance(wavelength, np.ndarray):
            return np.full(wavelength.shape, self._refractive_index)
        else:
            return self._refractive_index


class SellmeierRefractor(Glass):
    def __init__(self, b1=0, b2=0, b3=0, c1=0, c2=0, c3=0):
        """A dispersive Refractive index model based on the `Sellmeier equation <https://en.wikipedia.org/wiki/Sellmeier_equation>`_. The six arguments (b1->b3, c1->c3) follow the expected convention found in literature.

        Sellmeier Coefficients for common glasses can be found at `refractiveindex.info <https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT>`_.
        """
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        super().__init__()  # Call the parent constructor

    def index_at(
        self, wavelength: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:

        return np.sqrt(
            1
            + (self.b1 * wavelength ** 2) / (wavelength ** 2 - self.c1)
            + (self.b2 * wavelength ** 2) / (wavelength ** 2 - self.c2)
            + (self.b3 * wavelength ** 2) / (wavelength ** 2 - self.c3)
        )


absorber = _AbsorbingMaterial()  # an
"""A bulk absorbing material"""

# instance of the absorbing material class to call
mirror = _ReflectingMaterial()
"""A perfectly reflecting material"""

glass = {
    "ideal": BasicRefractor(1.5),
    "BK7": SellmeierRefractor(
        1.03961212,
        0.231792344,
        1.01046945,
        6.00069867e-3,
        2.00179144e-2,
        1.03560653e02,
    ),
    "SF5": SellmeierRefractor(
        1.52481889, 0.187085527, 1.42729015, 0.011254756, 0.0588995392, 129.141675
    ),
    "SF2": SellmeierRefractor(
        1.40301821, 0.231767504, 0.939056586, 0.0105795466, 0.0493226978, 112.405955
    ),
}
"""A Dictionary of common glasses.
"""
