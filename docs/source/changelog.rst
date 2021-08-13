=======================
Changelog
=======================

v0.3.0
-------

* Added dispersive materials based on `Sellmeier Coefficients <https://en.wikipedia.org/wiki/Sellmeier_equation>`_.
* Added :class:`~pyrayt.materials.Glass` base class for refractive materials.
* Multiple new components:

  * :func:`~pyrayt.components.equilateral_prism` 
  * :func:`~pyrayt.components.thick_lens` for arbitrary spherical lenses.

* RayTracer :meth:`show` function has additional keyword arguments for shading rays.

  * *wavelength*: shades each ray based on the wavelength of the ray.
  * *source*: shades each ray based on which source generated the ray.
* RayTracer has new method :meth:`~pyryat._pyrayt.RayTracer.calculate_source_ids` which calculates and appends a *source_id* column to the results dataframe. 'source_id' is the index of the source that generated the ray.
* New :mod:`pyrayt.utils` module with convenience functions for lens design.
* Added :class:`~pyrayt._pyrayt.pin` context manager that preserves an object's position.
* Various minor changes to the :mod:`tinygfx` package
* Additional examples.