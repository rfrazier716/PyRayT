import numpy as np
from dataclasses import dataclass, field


class RGBAColor(np.ndarray):
    def __new__(cls, *args, **kwargs):
        # creates an array with the homogeneous coordinates
        obj = np.zeros(4, dtype=float).view(cls)
        return obj

    def __init__(self, r: float = 0, g: float = 0, b: float = 0, a: float = 1) -> None:
        # assign initialization
        self[0] = r
        self[1] = g
        self[2] = b
        self[3] = a

    @property
    def r(self):
        return self[0]

    @r.setter
    def r(self, x):
        self[0] = x

    @property
    def g(self):
        return self[1]

    @g.setter
    def g(self, y):
        self[1] = y

    @property
    def b(self):
        return self[2]

    @b.setter
    def b(self, z):
        self[2] = z

    @property
    def a(self):
        return self[3]

    @a.setter
    def a(self, w):
        self[3] = w


WHITE = RGBAColor(1, 1, 1)
BLACK = RGBAColor()
RED = RGBAColor(1, 0, 0)
GREEN = RGBAColor(0, 1, 0)
BLUE = RGBAColor(0, 0, 1)
YELLOW = RGBAColor(1, 1, 0)
ORANGE = RGBAColor(1,0.5,0)


@dataclass
class GoochMaterial(object):
    base_color: RGBAColor = field(default_factory=RGBAColor)
    warm_color: RGBAColor = field(default_factory=RGBAColor)
    cool_color: RGBAColor = field(default_factory=RGBAColor)

    alpha: float = 0.2
    beta: float = 0.6

    def shade(self, rays: np.ndarray, normals: np.ndarray, light_positions: np.ndarray) -> np.ndarray:
        # define the hot and cold shades based on alpha
        shade_warm = self.warm_color
        shade_cool = self.cool_color

        # make a list of padded rays so you can get the every light vector normal at each position
        rays = np.atleast_3d(rays)
        normals = normals[:3, np.newaxis] if normals.ndim == 1 else normals[:3]

        if light_positions.ndim == 1:
            light_vectors = (light_positions[:3, np.newaxis] - rays[0, :3])[np.newaxis, ...]

        else:
            light_vectors = light_positions[:3, :, np.newaxis] - np.tile(rays[0, :3], (light_positions.shape[-1], 1, 1))

        # normalize the light vector
        light_vectors /= np.linalg.norm(light_vectors, axis=1)
        # take the dot product of each slice of the light vectors with the normal vector
        light_normal_dot = np.einsum('ijk,jk->ik', light_vectors, normals)

        # calculate the add_mixture ratio for the shader based on the dot product
        mixture_ratio = 0.5 * (1 + light_normal_dot)
        # take the average mixture ratio for all lights
        all_light_mixture = np.mean(mixture_ratio, axis=0)
        pixel_value = np.einsum('i,j->ij', shade_warm, all_light_mixture) + np.einsum('i,j->ij', shade_cool,
                                                                                      1 - all_light_mixture)
        return pixel_value


g_white = GoochMaterial(base_color=WHITE, warm_color=ORANGE, cool_color=BLUE)
