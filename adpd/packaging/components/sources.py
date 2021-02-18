import adpd.packaging.simple_cg as cg
import numpy as np
import abc


class Source(cg.WorldObject, abc.ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_rays(self, n_rays):
        local_rays = self._local_ray_generation(n_rays)
        world_rays = np.matmul(self._world_coordinate_transform, local_rays)  # transform rays to world space
        world_rays[1] /= np.linalg.norm(world_rays[1], axis=0)  # normalize the direction vector
        return world_rays

    @abc.abstractmethod
    def _local_ray_generation(self, n_rays):
        pass

    @abc.abstractmethod
    def get_wavelength(self, n_rays):
        pass


class LineOfRays(Source):

    def __init__(self, spacing=1, wavelength=0.633, *args, **kwargs):
        self._spacing = spacing
        self._wavelength = wavelength
        super().__init__(*args, **kwargs)

    def _local_ray_generation(self, n_rays):
        """
        creates a line of rays directed towards the positive x-axis along the y-axis
        :param n_rays:
        :return:
        """
        rays = cg.bundle_of_rays(n_rays)
        # if we want more than one ray, linearly space them, otherwise default position is fine
        if n_rays > 1:
            ray_position = np.linspace(-self._spacing / 2, self._spacing / 2, n_rays)
            rays[0, 1] = ray_position  # space rays along the y-axis
        rays[1, 0] = 1  # direct rays along positive x
        return rays

    def get_wavelength(self, n_rays):
        return self._wavelength
