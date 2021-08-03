from typing import Tuple
import numpy as np


def wavelength_to_rgb(wavelength: Tuple[float, np.ndarray], gamma=0.8) -> np.ndarray:

    """
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html

    This code adapted from noah.org
    http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    """
    wavelength = np.asarray(wavelength)  # convert the wavelength into a numpy array
    color = np.empty((3, wavelength.shape[0]))  # an empty matrix to generate color

    # calculate the color for 380 -> 440 nm
    zone_min = 0.38
    zone_max = 0.44
    clipped_wave = np.maximum(wavelength, zone_min)
    attenuation = 0.3 + 0.7 * (clipped_wave - zone_min) / (zone_max - zone_min)
    red = (
        np.abs((-(clipped_wave - zone_max) / (zone_max - zone_min) * attenuation))
        ** gamma
    )
    green = np.full(wavelength.shape, 0.0)
    blue = np.abs((1.0 * attenuation)) ** gamma
    zone = np.vstack(
        (red, green, blue)
    )  # the color calculations based on the first zone
    color = np.where(
        wavelength < zone_max, zone, color
    )  # fill the color array at the appropriate spots

    # calculate the color for 440 -> 490 nm
    zone_min = 0.44
    zone_max = 0.49
    red = np.full(wavelength.shape, 0.0)
    green = np.abs(((wavelength - zone_min) / (zone_max - zone_min))) ** gamma
    blue = np.full(wavelength.shape, 1.0)
    zone = np.vstack(
        (red, green, blue)
    )  # the color calculations based on the first zone
    color = np.where(
        np.logical_and(wavelength >= zone_min, wavelength < zone_max), zone, color
    )  # fill the color array at the appropriate spots

    # calculate the color for 490 -> 510 nm
    zone_min = 0.49
    zone_max = 0.51
    red = np.full(wavelength.shape, 0.0)
    green = np.full(wavelength.shape, 1.0)
    blue = np.abs(((zone_max - wavelength) / (zone_max - zone_min))) ** gamma
    zone = np.vstack(
        (red, green, blue)
    )  # the color calculations based on the first zone
    color = np.where(
        np.logical_and(wavelength >= zone_min, wavelength < zone_max), zone, color
    )  # fill the color array at the appropriate spots

    # calculate the color for 510 -> 580
    zone_min = 0.51
    zone_max = 0.58
    red = np.abs(((wavelength - zone_min) / (zone_max - zone_min))) ** gamma
    green = np.full(wavelength.shape, 1.0)
    blue = np.full(wavelength.shape, 0)
    zone = np.vstack(
        (red, green, blue)
    )  # the color calculations based on the first zone
    color = np.where(
        np.logical_and(wavelength >= zone_min, wavelength < zone_max), zone, color
    )  # fill the color array at the appropriate spots

    # calculate the color for 580-645
    zone_min = 0.58
    zone_max = 0.645
    red = np.full(wavelength.shape, 1.0)
    green = np.abs(((zone_max - wavelength) / (zone_max - zone_min))) ** gamma
    blue = np.full(wavelength.shape, 0)
    zone = np.vstack(
        (red, green, blue)
    )  # the color calculations based on the first zone
    color = np.where(
        np.logical_and(wavelength >= zone_min, wavelength < zone_max), zone, color
    )  # fill the color array at the appropriate spots

    # calculate the color for >645nm
    zone_min = 0.645
    zone_max = 0.75
    clipped_wave = np.minimum(wavelength, zone_max)
    attenuation = 0.3 + 0.7 * (zone_max - clipped_wave) / (zone_max - zone_min)
    zone = np.zeros(
        (3, wavelength.shape[0])
    )  # the color calculations based on the first zone
    zone[0] = np.abs(attenuation) ** gamma
    color = np.where(
        wavelength >= zone_min, zone, color
    )  # fill the color array at the appropriate spots

    return color.T


def lensmakers_equation(r1: float, r2: float, n_lens: float, thickness: float) -> float:
    """
    Calculate the focal length of a thick spherical lens using the lensmaker's equation.

    :param r1: the first radius of curvature, positive for convex, negative for concave
    :param r2: the second radius of curvature, negative for convex, positive for concave
    :param n_lens: the refractive index of the lens
    :param thickness: The thickness of the lens

    :return: The focal length of the lens based on the paraxial approximation
    """

    p = (n_lens - 1) * (1 / r1 - 1 / r2 + (n_lens - 1) * thickness / (n_lens * r1 * r2))
    return 1 / p
