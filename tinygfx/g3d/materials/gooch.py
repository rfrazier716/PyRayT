import numpy as np
from dataclasses import dataclass, field
import abc

from tinygfx.g3d.materials.color import RGBAColor
import tinygfx.g3d.materials.color as color


class Material(abc.ABC):
    """
    A base class for all materials
    """

    @abc.abstractmethod
    def shade(
        self, rays: np.ndarray, normals: np.ndarray, light_positions: np.ndarray
    ) -> np.ndarray:
        pass


@dataclass
class GoochMaterial(Material):
    base_color: RGBAColor = field(default_factory=color.RGBAColor)
    warm_color: RGBAColor = field(default_factory=color.RGBAColor)
    cool_color: RGBAColor = field(default_factory=color.RGBAColor)

    alpha: float = 0.3
    beta: float = 0.3

    def shade(
        self, rays: np.ndarray, normals: np.ndarray, light_positions: np.ndarray
    ) -> np.ndarray:
        # define the hot and cold shades based on alpha
        # shade_warm = np.minimum(self.warm_color+self.alpha*self.base_color,1)
        # shade_cool = np.minimum(self.cool_color+self.beta*self.base_color,1)
        shade_warm = (1 - self.alpha) * self.warm_color + self.alpha * self.base_color
        shade_cool = (1 - self.beta) * self.cool_color + self.beta * self.base_color

        # make a list of padded rays so you can get the every light vector normal at each position
        rays = np.atleast_3d(rays)
        normals = normals[:3, np.newaxis] if normals.ndim == 1 else normals[:3]

        if light_positions.ndim == 1:
            light_vectors = (light_positions[:3, np.newaxis] - rays[0, :3])[
                np.newaxis, ...
            ]

        else:
            light_vectors = light_positions[:3, :, np.newaxis] - np.tile(
                rays[0, :3], (light_positions.shape[-1], 1, 1)
            )

        # normalize the light vector
        light_vectors /= np.linalg.norm(light_vectors, axis=1)
        # take the dot product of each slice of the light vectors with the normal vector
        light_normal_dot = np.einsum("ijk,jk->ik", light_vectors, normals)

        # calculate the add_mixture ratio for the shader based on the dot product
        mixture_ratio = 0.5 * (1 + light_normal_dot)
        # take the average mixture ratio for all lights
        all_light_mixture = np.mean(mixture_ratio, axis=0)
        pixel_value = np.einsum("i,j->ij", shade_warm, all_light_mixture) + np.einsum(
            "i,j->ij", shade_cool, 1 - all_light_mixture
        )
        return pixel_value


def _blue_yellow_gooch(base_color):
    return GoochMaterial(
        base_color=base_color, warm_color=color.ORANGE, cool_color=color.BLUE
    )


WHITE = _blue_yellow_gooch(color.WHITE)
RED = _blue_yellow_gooch(color.RED)
GREEN = _blue_yellow_gooch(color.GREEN)
BLUE = GoochMaterial(
    base_color=color.BLUE, warm_color=color.YELLOW, cool_color=color.BLUE, alpha=0.2
)
YELLOW = _blue_yellow_gooch(color.YELLOW)
ORANGE = _blue_yellow_gooch(color.ORANGE)
BLACK = _blue_yellow_gooch(color.BLACK)
