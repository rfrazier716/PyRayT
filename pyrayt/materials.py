from typing import Union
import numpy as np
from pyrayt._pyrayt import RaySet
from tinygfx.g3d import materials as cg_matl
import tinygfx.g3d as cg
import abc
from functools import lru_cache

class TracableMaterial(cg_matl.gooch.Material):
    def __init__(self, base_material, *args, **kwargs):
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
        """
        Traces Rays through the surface, returning an updated RaySet Object
        """
        pass


class _AbsorbingMaterial(TracableMaterial):
    def __init__(self, *args, **kwargs):
        super().__init__(
            cg_matl.gooch.BLACK, *args, **kwargs
        )  # call the parent constructor

    def trace(self, surface, ray_set: RaySet) -> RaySet:
        # an absorbing material will kill the ray direction vector
        ray_set.rays[1] = 0
        return ray_set


class _ReflectingMaterial(TracableMaterial):
    def __init__(self, *args, **kwargs):
        super().__init__(cg_matl.gooch.BLUE, *args, **kwargs)

    def trace(self, surface: cg.TracerSurface, ray_set: RaySet) -> RaySet:
        # a reflecting material will
        normals = surface.get_world_normals(ray_set.rays[0])
        ray_set.rays[1] = cg.reflect(ray_set.rays[1], normals)
        return ray_set

class _Glass(TracableMaterial):
    def __init__(self, *args, **kwargs):
        super().__init__( base_material=cg_matl.gooch.BLUE, *args, **kwargs)

    def trace(self, surface: cg.TracerSurface, ray_set: RaySet) -> RaySet:
        normals = surface.get_world_normals(ray_set.rays[0])
        ray_set.rays[1], ray_set.index = cg.refract(
            ray_set.rays[1], normals, ray_set.index, self.index_at(ray_set.wavelength)
        )
        return ray_set

    @lru_cache
    def abbe(self) -> float:
        """
        Calculates the Abbe number of the material 
        """
        n_short = self.index_at(0.4861)
        n_center = self.index_at(0.5893)
        n_long = self.index_at(0.6563)

        return (n_center -1)/(n_short - n_long)

    @abc.abstractmethod
    def index_at(self, wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass

class BasicRefractor(_Glass):
    def __init__(self, refractive_index, *args, **kwargs):
        self._refractive_index = refractive_index
        super().__init__()

    def index_at(self, wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(wavelength, np.ndarray):
            return np.full(wavelength.shape, self._refractive_index)
        else:
            return self._refractive_index


class SellmeierRefractor(_Glass):
    def __init__(self, b1=0, b2=0, b3=0, c1=0, c2=0, c3=0):
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        super().__init__() # Call the parent constructor

    def index_at(self, wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:

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
        1.52481889,
        0.187085527,
        1.42729015,
        0.011254756,
        0.0588995392,
        129.141675
    )
}
