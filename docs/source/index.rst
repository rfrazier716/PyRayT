.. adi_optics documentation master file, created by
   sphinx-quickstart on Thu Feb  4 15:39:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyRayT's documentation!
======================================

PyRayT is a geometric ray tracing package for optical system design. Unlike most ray tracers which focus on rendering scenes 3D scenes, PyRayT tracks every ray as it propagates through the system, storing refractive index, intersection, and collision information. This information is saved in a Pandas dataframe which can be filtered and analyzed after a trace is complete.

All components are objects that can be manipulated in 3D space, and numpy is used extensively under the hood to enable fast simulations, even when simulating 100k+ rays.

Explore the Docs
`````````````````

.. toctree::
   :maxdepth: 1

   install
   tutorial
   reference/index
   examples
   license
   generated/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
