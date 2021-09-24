=======================
Release Notes
=======================

v0.3.1
-------

* Fixed minor bug where package would not run on Python 3.7

v0.3.0
-------

* Added dispersive materials based on `Sellmeier Coefficients <https://en.wikipedia.org/wiki/Sellmeier_equation>`_.
* Added :class:`~pyrayt.materials.Glass` base class for refractive materials.
* New components:

  * :func:`~pyrayt.components.aperture`
  * :func:`~pyrayt.components.equilateral_prism` 
  * :func:`~pyrayt.components.thick_lens`

* New Sources

  * :class:`~pyrayt.components.CircleOfRays`
  * :class:`~pyrayt.components.WedgeOfRays`


* Updates to :class:`~pyrayt._pyrayt.RayTracer` class.

  * :meth:`~pyrayt._pyrayt.RayTracer.show` function has additional keyword arguments for shading rays.

    * **wavelength**: Shades each ray based on the wavelength of the ray.
    * **source**: Shades each ray based on which source generated the ray.

  * :meth:`~pyrayt._pyrayt.RayTracer.calculate_source_ids` calculates and appends a *source_id* column to the results dataframe. **source_id** is the index of the source that generated the ray.

* New :mod:`pyrayt.utils` module with convenience functions for lens design.
* Added :class:`~pyrayt._pyrayt.pin` context manager that preserves an object's position.
* Various minor changes to the :mod:`tinygfx` package
* Additional examples.