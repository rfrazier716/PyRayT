=======================
Changelog
=======================

v0.3.0
-------
* Added dispersive materials based on Sellmeier Coefficients.
* Added :code:`Glass` base class for refractive materials.
* RayTracer :code:`show` function has additional keyword arguments for shading rays.

  * *wavelength*: shades each ray based on the wavelength of the ray.
  * *source*: shades each ray based on which source generated the ray.

* New :mod:`pyrayt.utils` module with convenience functions for lens design.