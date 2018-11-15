Reading C3S SM images
---------------------

Reading of the C3S SM raw netcdf files can be done in two ways.

Reading by file name
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    from datetime import datetime
    from c3s_sm.interface import C3SImg
    import numpy.testing as nptest

    # read several parameters
    parameter = ['sm', 'sm_uncertainty']
    # the class is initialized with the exact filename.
    image_path = os.path.join(os.path.dirname(__file__), 'tests', 'c3s_sm-test-data',
                              'img', 'ICDR', '060_dailyImages', 'combined', '2017')
    image_file = 'C3S-SOILMOISTURE-L3S-SSMV-COMBINED-DAILY-20170701000000-ICDR-v201706.0.0.nc'
    img = C3SImg(os.path.join(image_path, image_file), parameter=parameter)

    # reading returns an image object which contains a data dictionary
    # with one array per parameter. The returned data is a global 0.25 degree
    # image/array.
    image = img.read()

    assert image.data['sm'].shape == (720, 1440)
    assert image.lon.shape == (720, 1440)
    assert image.lon.shape == image.lat.shape
    assert image.lon[0, 0] == -179.875
    assert image.lat[0, 0] == 89.875
    assert sorted(image.data.keys()) == sorted(parameter)
    assert(image.metadata['sm']['long_name'] == 'Volumetric Soil Moisture')
    nptest.assert_almost_equal(image.data['sm'][167, 785], 0.14548, 4)


Reading by date
~~~~~~~~~~~~~~~

All the C3S SM data in a directory structure can be accessed by date.
The filename is automatically built from the given date.

.. code-block:: python

    from c3s_sm.interface import C3S_Nc_Img_Stack

    parameter = 'sm'
    img = C3S_Nc_Img_Stack(data_path=os.path.join(os.path.dirname(__file__),
                                                    'tests', 'c3s_sm-test-data', 'img',
                                                    'ICDR', '061_monthlyImages', 'passive'),
                              parameter=parameter)

    image = img.read(datetime(2017, 7, 1, 0))

    nptest.assert_almost_equal(image.data['sm'][167, 785], 0.23400, 4)


For reading all image between two dates the
:py:meth:`c3s_sm.interface.C3S_Nc_Img_Stack.iter_images` iterator can be
used.
