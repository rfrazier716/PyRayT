import pyrayt
import tinygfx.g3d.primitives as primitives
import numpy as np
import abc
from typing import Tuple
from functools import lru_cache

import tinygfx.g3d as cg


class Source(cg.WorldObject, abc.ABC):

    def __init__(self, wavelength=0.633, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wavelength = wavelength

    def generate_rays(self, n_rays: int) -> pyrayt.RaySet:
        ray_set = self._local_ray_generation(n_rays)
        ray_set.rays = np.matmul(self._world_coordinate_transform, ray_set.rays)  # transform rays to world space
        ray_set.rays[1] /= np.linalg.norm(ray_set.rays[1], axis=0)  # normalize the direction vector
        return ray_set

    @abc.abstractmethod
    def _local_ray_generation(self, n_rays: int) -> pyrayt.RaySet:
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

    def _local_ray_generation(self, n_rays: int) -> pyrayt.RaySet:
        """
        creates a line of rays directed towards the positive x-axis along the y-axis

        :param n_rays:
        :return:
        """
        rayset = pyrayt.RaySet(n_rays)
        # if we want more than one ray, linearly space them, otherwise default position is fine
        if n_rays > 1:
            ray_position = np.linspace(-self._spacing / 2, self._spacing / 2, n_rays)
            rayset.rays[0, 1] = ray_position  # space rays along the y-axis
        rayset.rays[1, 0] = 1  # direct rays along positive x
        rayset.wavelength = self._wavelength
        return rayset


class ConeOfRays(Source):

    def __init__(self, cone_angle: float, wavelength=0.633, *args, **kwargs):
        super().__init__(wavelength, *args, **kwargs)
        self._angle = cone_angle * np.pi / 180.0

    def _local_ray_generation(self, n_rays: int) -> pyrayt.RaySet:
        """
        creates a line of rays directed towards the positive x-axis along the y-axis

        :param n_rays:
        :return:
        """
        rayset = pyrayt.RaySet(n_rays)
        # if we want more than one ray, change them to have the desired cone angle
        if n_rays > 1:
            angles = 2 * np.pi * np.arange(0, n_rays) / n_rays
            rayset.rays[1, 1] = np.sin(self._angle) * np.sin(angles)
            rayset.rays[1, 2] = np.sin(self._angle) * np.cos(angles)
        # the position in the x-direction is the cosine of the ray angle
        rayset.rays[1, 0] = np.cos(self._angle)
        rayset.wavelength = self._wavelength
        return rayset


class Lamp(Source):
    def __init__(self, width: float, length: float, max_angle: float = 90, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._max_angle = max_angle * np.pi / 180  # convert the max angle to radians internally
        self._width = width  # width is how far the lamp extends in the y-direction
        self._length = length  # length is how far the lamp extends in the z-direction

    def _local_ray_generation(self, n_rays: int) -> pyrayt.RaySet:
        rayset = pyrayt.RaySet(n_rays)
        rayset.wavelength = self._wavelength  # set the wavelength
        theta, phi = _sphere_sample(n_rays, self._max_angle)

        # randomly distribute the rays about the pd surface
        rayset.rays[0, 1] = self._width * (np.random.random_sample(n_rays) - 0.5)
        rayset.rays[0, 2] = self._length * (np.random.random_sample(n_rays) - 0.5)

        # orient the ray angles based on the theta/phi values
        rayset.rays[1, 0] = np.cos(theta)
        rayset.rays[1, 1] = np.sin(theta) * np.cos(phi)
        rayset.rays[1, 2] = np.sin(theta) * np.sin(phi)

        # set the intensity based on the theta value
        rayset.intensity = 100.0 * np.cos(theta)

        return rayset  # return the rotated rayset


class StaticLamp(Lamp):

    @lru_cache(10)
    def generate_rays(self, n_rays: int) -> pyrayt.RaySet:
        return super().generate_rays(n_rays)


def _sphere_sample(n_pts: int, max_angle: float) -> np.ndarray:
    """
    Creates N random somples on the unit sphere where theta ranges from [0,max_angle] and phi ranges [0,2*pi]

    :param n_pts:
    :param max_angle: the max angle (in radians) of the theta value
    :return: a 2xn array of sampled angle values, the first row is theta, and the second is phi
    """
    # going to use inverse cdf sampling to generate theta and phi points
    uv_samples = np.random.random_sample((2, n_pts))

    # convert row 0 to theta samples
    uv_samples[0] = np.arccos(1 - uv_samples[0] * (1 - np.cos(max_angle)))
    uv_samples[1] *= (2 * np.pi)  # convert row 1 to phi samples
    return uv_samples
