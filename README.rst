======
c3s_sm
======

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

API Key
-------
In order to download C3S soil moisture data from CDS, this package uses the
CDS API (https://pypi.org/project/cdsapi/). You can
either pass your credentials directly on the command line (which might be
unsafe) or set up a `.cdsapirc` file in your home directory (recommended).
Please see the description at https://cds.climate.copernicus.eu/how-to-api.

Quickstart
==========
Download image data from CDS using the c3s_sm shell command

.. code-block:: shell

    c3s_sm download /tmp/c3s/img -s 2023-09-01 -e 2023-10-31 -v v202212

... and convert them to time series

.. code-block:: shell

    c3s_sm reshuffle /tmp/c3s/img /tmp/c3s/ts

Finally, in python, read the time series data for a location as pandas
DataFrame.

.. code-block:: python

    >> from c3s_sm.interface import C3STs
    >> ds = C3STs('/tmp/c3s/ts')
    >> ts = ds.read(18, 48)

                      sm  sm_uncertainty  flag  ...  mode  sensor            t0
    2023-09-01  0.222125        0.014661     0  ...     2     544  19601.100348
    2023-09-02  0.213480        0.011166     0  ...     3   38432  19602.051628
    2023-09-03  0.197324        0.014661     0  ...     3   33312  19602.945730
                  ...             ...   ...  ...   ...     ...           ...
    2023-10-29  0.265275        0.013192     0  ...     3   37408  19658.955236
    2023-10-30  0.256964        0.011166     0  ...     3   38432  19660.085144
    2023-10-31  0.241187        0.014661     0  ...     3   33312  19660.945730


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

Build Docker image
==================

For operational implementations, this package and be installed in a
docker container.

- Check out the repo at the branch/tag/commit you want build
- Make sure you have docker installed and run the command (replace the tag `latest`
  with something more meaningful, e.g. a matching version number)

.. code::

    docker build -t c3s_sm:latest . 2>&1 | tee docker_build.log

This will execute the commands from the Dockerfile. I.e. install a new environment
with the checked out version of the c3s_sm package.

To build and publish the image online, we have a GitHub Actions workflow in
``.github/workflows/docker.yml``


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
