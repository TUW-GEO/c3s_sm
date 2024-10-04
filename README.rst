============
c3s_sm
============

|ci| |cov| |pip| |doc|

.. |ci| image:: https://github.com/TUW-GEO/c3s_sm/actions/workflows/ci.yml/badge.svg?branch=master
   :target: https://github.com/TUW-GEO/c3s_sm/actions

.. |cov| image:: https://coveralls.io/repos/TUW-GEO/c3s_sm/badge.png?branch=master
  :target: https://coveralls.io/r/TUW-GEO/c3s_sm?branch=master

.. |pip| image:: https://badge.fury.io/py/c3s_sm.svg
    :target: http://badge.fury.io/py/c3s-sm

.. |doc| image:: https://readthedocs.org/projects/c3s_sm/badge/?version=latest
   :target: http://c3s-sm.readthedocs.org/


Processing tools and tutorials for users of the C3S satellite soil moisture
service ( https://doi.org/10.24381/cds.d7782f18 ). Written in Python.

Installation
============

The c3s_sm package and all required dependencies can be installed via

.. code-block:: shell

    pip install c3s_sm

On macOS if you get ``ImportError: Pykdtree failed to import its C extension``, then it
might be necessary to install the pykdtree package from conda-forge

.. code-block:: shell

    conda install -c conda-forge pykdtree

Tutorials
=========

We provide tutorials on using the C3S Soil Moisture data:

- `Tutorial 1: DataAccess from CDS & Anomaly computation <https://c3s-sm.readthedocs.io/en/latest/_static/T1_DataAccess_Anomalies.html>`_

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

Guidelines
----------

If you want to contribute please follow these steps:

- Fork the c3s_sm repository to your account
- Clone the repository, make sure you use ``git clone --recursive`` to also get
  the test data repository.
- make a new feature branch from the c3s_sm master branch
- Add your feature
- Please include tests for your contributions in one of the test directories.
- submit a pull request to our master branch

