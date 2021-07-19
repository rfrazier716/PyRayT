import numpy as np
from pyrayt._pyrayt import RaySet
from tinygfx.g3d import materials as cg_matl
import tinygfx.g3d as cg
import abc


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


class BasicRefractor(TracableMaterial):
    def __init__(self, refractive_index, *args, **kwargs):
        self._index = refractive_index
        super().__init__(cg_matl.gooch.BLUE, *args, **kwargs)

    def trace(self, surface: cg.TracerSurface, ray_set: RaySet) -> RaySet:
        # a reflecting material will
        normals = surface.get_world_normals(ray_set.rays[0])
        ray_set.rays[1], ray_set.index = cg.refract(
            ray_set.rays[1], normals, ray_set.index, self._index
        )
        return ray_set


absorber = _AbsorbingMaterial()  # an
"""A bulk absorbing material"""

# instance of the absorbing material class to call
mirror = _ReflectingMaterial()
glass = BasicRefractor(1.5)
