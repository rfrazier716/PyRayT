.. adi_optics documentation master file, created by
   sphinx-quickstart on Thu Feb  4 15:39:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyRayT's documentation!
======================================

.. raw:: html

   <div class="badges" style="margin-bottom: 1em">
   <a href="https://github.com/rfrazier716/pyrayt">
      <img src="https://img.shields.io/badge/VCS-Github-brightgreen?logo=github&style=flat">
   </a>
   <a href="https://pypi.org/project/pyrayt/">
      <img src="https://img.shields.io/pypi/v/pyrayt">
   </a>
   <a href="https://app.circleci.com/pipelines/github/rfrazier716/PyRayT">
      <img src="https://circleci.com/gh/rfrazier716/PyRayT.svg?style=shield">
   </a>
   </div>

`PyRayT <https://github.com/rfrazier716/pyrayt>`_ is a geometric ray tracing package for optical system design. Unlike most ray tracers which focus on rendering scenes 3D scenes, PyRayT tracks every ray as it propagates through the system, storing refractive index, intersection, and collision information. This information is saved in a Pandas dataframe which can be filtered and analyzed after a trace is complete.

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
   release
   generated/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
