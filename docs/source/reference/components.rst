######################
 Traceable Components
######################

Sources
========

Basic Sources
``````````````

* :class:`~pyrayt.components.LineOfRays`
* :class:`~pyrayt.components.ConeOfRays`

Additional Sources
```````````````````

* :class:`~pyryayt.components.Lamp`
* :class:`~pyrayt.components.StaticLamp`

Built in Parts
====================

.. _Apertures:
Specifying Apertures
`````````````````````
All optical component factor functions accept an :code:`aperture` keyword argument. Internally, the

Spherical Lenses
`````````````````
PyRayT includes factory functions for spherical lenses that return :func:`tinygfx.g3d.csg.CSGSurface` objects. Additional
keyword arguments can be passed to each function.

* :func:`~pyrayt.components.biconvex_lens`
* :func:`~pyrayt.components.plano_convex_lens`

Mirrors
````````

* :func:`~pyrayt.components.spherical_mirror`
* :func:`~pyrayt.components.parabolic_mirror`
* :func:`~pyrayt.components.plane_mirror`
* :func:`~pyrayt.components.elliptical_mirror`

Miscellaneous
``````````````

* :func:`~pyrayt.components.baffle`


