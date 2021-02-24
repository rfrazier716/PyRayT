import pyrayt.simple_cg as cg
import numpy as np
import abc


class Source(cg.WorldObject, abc.ABC):

    def __init__(self, wavelength=0.633, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wavelength = wavelength

    def generate_rays(self, n_rays: int) -> cg.RaySet:
        ray_set = self._local_ray_generation(n_rays)
        ray_set.rays = np.matmul(self._world_coordinate_transform, ray_set.rays)  # transform rays to world space
        ray_set.rays[1] /= np.linalg.norm(ray_set.rays[1], axis=0)  # normalize the direction vector
        return ray_set

    @abc.abstractmethod
    def _local_ray_generation(self, n_rays: int) -> cg.RaySet:
        pass

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value


class LineOfRays(Source):

    def __init__(self, spacing=1, wavelength=0.633, *args, **kwargs):
        super().__init__(wavelength, *args, **kwargs)
        self._spacing = spacing

    def _local_ray_generation(self, n_rays: int) -> cg.RaySet:
        """
        creates a line of rays directed towards the positive x-axis along the y-axis

        :param n_rays:
        :return:
        """
        set = cg.RaySet(n_rays)
        # if we want more than one ray, linearly space them, otherwise default position is fine
        if n_rays > 1:
            ray_position = np.linspace(-self._spacing / 2, self._spacing / 2, n_rays)
            set.rays[0, 1] = ray_position  # space rays along the y-axis
        set.rays[1, 0] = 1  # direct rays along positive x
        set.wavelength = self._wavelength
        return set
