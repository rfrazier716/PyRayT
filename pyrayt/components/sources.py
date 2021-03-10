import tinygfx.g3d.primitives as primitives
import numpy as np
import abc


class Source(primitives.WorldObject, abc.ABC):

    def __init__(self, wavelength=0.633, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wavelength = wavelength

    def generate_rays(self, n_rays: int) -> primitives.RaySet:
        ray_set = self._local_ray_generation(n_rays)
        ray_set.rays = np.matmul(self._world_coordinate_transform, ray_set.rays)  # transform rays to world space
        ray_set.rays[1] /= np.linalg.norm(ray_set.rays[1], axis=0)  # normalize the direction vector
        return ray_set

    @abc.abstractmethod
    def _local_ray_generation(self, n_rays: int) -> primitives.RaySet:
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

    def _local_ray_generation(self, n_rays: int) -> primitives.RaySet:
        """
        creates a line of rays directed towards the positive x-axis along the y-axis

        :param n_rays:
        :return:
        """
        set = primitives.RaySet(n_rays)
        # if we want more than one ray, linearly space them, otherwise default position is fine
        if n_rays > 1:
            ray_position = np.linspace(-self._spacing / 2, self._spacing / 2, n_rays)
            set.rays[0, 1] = ray_position  # space rays along the y-axis
        set.rays[1, 0] = 1  # direct rays along positive x
        set.wavelength = self._wavelength
        return set


class OrthoGraphicCamera(primitives.WorldObject):
    """
    A camera oriented along the z-axis pointing along the x-axis
    """

    def __init__(self, h_pixel_count: int, h_width: float, aspect_ratio: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) # call the next constructor in the MRO
        self._h_pixels = h_pixel_count
        self._h_width = h_width
        self._v_width = aspect_ratio * h_width
        self._v_pixels = int(aspect_ratio * self._h_pixels)

    def get_resolution(self):
        return (self._h_pixels, self._v_pixels)

    def get_span(self):
        return (self._h_width, self._v_width)

    def generate_rays(self) -> primitives.RaySet:
        ray_set = self._local_ray_generation()
        ray_set.rays = np.matmul(self._world_coordinate_transform, ray_set.rays)  # transform rays to world space
        ray_set.rays[1] /= np.linalg.norm(ray_set.rays[1], axis=0)  # normalize the direction vector
        return ray_set

    def _local_ray_generation(self) -> primitives.RaySet:
        h_steps = np.linspace(-self._h_width / 2, self._h_width / 2, self._h_pixels)
        v_steps = np.linspace(self._v_width / 2, -self._v_width / 2, self._v_pixels)

        set = primitives.RaySet(self._h_pixels * self._v_pixels)
        ys, zs = np.meshgrid(h_steps, v_steps)
        set.rays[0, 1] = ys.reshape(-1)
        set.rays[0, 2] = zs.reshape(-1)
        set.rays[1, 0] = 1  # position rays along positive z axis

        return set
