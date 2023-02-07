============
c3s_sm
============


.. image:: https://github.com/TUW-GEO/c3s_sm/workflows/Automated%20Tests/badge.svg?branch=master
   :target: https://github.com/TUW-GEO/c3s_sm/actions

.. image:: https://coveralls.io/repos/github/TUW-GEO/c3s_sm/badge.svg?branch=master
    :target: https://coveralls.io/github/TUW-GEO/c3s_sm?branch=master

.. image:: https://badge.fury.io/py/c3s-sm.svg
    :target: https://badge.fury.io/py/c3s-sm
 
.. image:: https://readthedocs.org/projects/c3s_sm/badge/?version=latest
    :target: http://c3s_sm.readthedocs.io/en/latest/?badge=latest

Reading and reshuffling of C3S soil moisture Written in Python.

Installation
============

Setup of a complete environment with `conda
<http://conda.pydata.org/miniconda.html>`_ can be performed using the following
commands:

.. code-block:: shell

  git clone git@github.com:TUW-GEO/c3s_sm.git c3s_sm
  cd c3s_sm
  conda env create -f environment.yml
  source activate c3s_sm

Tutorials
=========

We provide (general) tutorials on using the C3S Soil Moisture data:

- `Tutorial 1: DataAccess from CDS & Anomaly computation <T1_DataAccess_Anomalies.html>`_

These tutorials are designed to run on `mybinder.org <mybinder.org/>`_
You can find the code for all examples in
`this repository <https://github.com/TUW-GEO/c3s_sm-tutorials>`_.

Supported Products
==================

At the moment this package supports C3S soil moisture data
in netCDF format (reading and time series creation)
with a spatial sampling of 0.25 degrees.

Contribute
==========

We are happy if you want to contribute. Please raise an issue explaining what
is missing or if you find a bug. We will also gladly accept pull requests
against our master branch for new features or bug fixes.

Development setup
-----------------

For Development we also recommend a ``conda`` environment. You can create one
including test dependencies and debugger by running
``conda env create -f environment.yml``. This will create a new ``c3s_sm``
environment which you can activate by using ``source activate c3s_sm``.

Guidelines
----------

If you want to contribute please follow these steps:

- Fork the c3s_sm repository to your account
- Clone the repository, make sure you use ``git clone --recursive`` to also get
  the test data repository.
- make a new feature branch from the c3s_sm master branch
- Add your feature
- Please include tests for your contributions in one of the test directories.
  We use py.test so a simple function called test_my_feature is enough
- submit a pull request to our master branch

Note
====

This project has been set up using PyScaffold 2.5. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
