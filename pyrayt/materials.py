import numpy as np
from pyrayt import RaySet
from tinygfx.g3d import materials as cg_matl
import abc


class TracableMaterial(cg_matl.gooch.Material):
    def __init__(self, base_material, *args, **kwargs):
        super().__init__(*args, **kwargs) # call the next constructor
        self._base_material = base_material # the material to call whenever the object is being rendered

    def shade(self, rays: np.ndarray, normals: np.ndarray, light_positions: np.ndarray) -> np.ndarray:
        return self._base_material.shade(rays, normals, light_positions)

    @abc.abstractmethod
    def trace(self, ray_set: RaySet) -> RaySet:
        """
        Traces Rays through the surface, returning an updated RaySet Object
        """
        pass

class _AbsorbingMaterial(TracableMaterial):

    def __init__(self, *args, **kwargs):
        super().__init__(cg_matl.gooch.BLACK, *args, **kwargs) # call the parent constructor

    def trace(self, ray_set: RaySet) -> RaySet:
        # an absorbing material will kill the ray direction vector
        ray_set.rays[1] = 0
        return ray_set


absorber = _AbsorbingMaterial() # an instance of the absorbing material class to call