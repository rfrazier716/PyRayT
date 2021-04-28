#################
 Materials
#################

While surface primitives are used for calculating ray intersections, materials are what dictate how a ray is
propagated at the intersection. Every traceable component has an associate material which does not need to be unique to
the object (for example you can make a glass material and assign the same one to multiple lenses). materials can be changed
after an objects construction through the :attr:`~tinygfx.g3d.TracerSurface.material` attribute.

.. note::

    Dispersive materials are not currently supported, but are being worked on for a future release.

Material Base Classes
======================

The following base classes can be used to define custom materials.

* :class:`~pyrayt.materials.TracableMaterial` - The base class for all materials. To create a custom material inherit from this class and implement the :meth:`~pyrayt.materials.TracableMaterial.trace` method.

* :class:`~pyrayt.materials.BasicRefractor` - A refractive material with a constant refractive index across all wavelengths.


Built In Materials
===================

A few common materials are defined for convenience. These are not captured by Sphinx but all exist in the :code:`pyrayt.materials` module

* :code:`absorber` - A bulk absorber that absorbs every ray incident to it.
* :code:`reflector` - An ideal reflector with 100% transmission coefficient
* :code:`glass` - A purely refractive material with n=1.5