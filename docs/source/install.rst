##############
 Installation
##############

PyRayT requires Python 3.7 or 3.8, and currently only supports 32-bit interpreters due to how object id's are stored. If you do not already have a Python environment configured on your computer, please see the instructions for installing the full `scientific Python stack <https://scipy.org/install.html>`_.

Below we assume you have the default Python environment already configured on your computer and you intend to install :code:`pyrayt` inside of it. If you want to create and work with Python virtual environments, please follow instructions on `venv`_ and `virtual environments`_.

.. _`venv`: https://docs.python.org/3/library/venv.html
.. _`virtual environments`:https://docs.python-guide.org/dev/virtualenvs/

First, make sure you have the latest version of :code:`pip` (the Python package manager) installed. If you do not, refer to the `Pip documentation <https://pip.pypa.io/en/stable/installing/>`_ and install :code:`pip` first.

Installing the Released Version
================================
Before installing pyrayt, check that your python environment is running a 32-bit interpreter:

.. prompt:: python >>> auto

    >>> import struct
    >>> print(struct.calcsize("P") * 8)
    32

If the output from the print statement is 64 instead of 32, the environment is using a 64-bit interpreter and pyrayt will fail to simulate propertly. Install a 32-bit interpreter and recreate the environment.

Install the current release of :code:`pyrayt` with :code:`pip`:

.. code:: shell

    py -m pip install pyrayt

To upgrade to a newer release use the :code:`--upgrade` flag:

.. code:: shell

    py -m pip install --upgrade pyrayt

If you do not have permission to install software systemwide, you can install into your user directory using the --user flag:

.. code:: shell

    py -m pip install --user pyrayt

Verify the installation by opening a python prompt and importing the top-level modules:

.. prompt:: python >>> auto

    >>> import pyrayt
    >>> import tinygfx.g3d

If both commands run without error, PyRayT has been successfully installed.