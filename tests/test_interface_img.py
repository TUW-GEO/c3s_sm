# -*- coding: utf-8 -*-

from c3s_sm.interface import C3SImg
import os
import numpy.testing as nptest
from c3s_sm.grid import C3SLandGrid, C3SCellGrid

# lat=48.125, lon=16.375
def test_C33Ts_tcdr_combined_daily():
    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'TCDR', '060_dailyImages', 'combined', '2014',
                        'C3S-SOILMOISTURE-L3S-SSMV-COMBINED-DAILY-20140101000000-TCDR-v201801.0.0.nc'))


    ds = C3SImg(file, mode='r', parameters='sm', array_1D=False)
    image= ds.read()

    nptest.assert_almost_equal(image.data['sm'][167, 785], 0.34659, 4)
    assert(image.metadata['sm']['long_name'] == 'Volumetric Soil Moisture')
    #assert(file_meta['creator_name'] == 'Earth Observation Data Center (EODC)')



def test_C33Ts_tcdr_active_monthly():

    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'TCDR', '061_monthlyImages', 'active',
                        'C3S-SOILMOISTURE-L3S-SSMS-ACTIVE-MONTHLY-20140101000000-TCDR-v201801.0.0.nc'))


    ds = C3SImg(file, mode='r', parameters='sm', array_1D=False)
    image = ds.read()

    nptest.assert_almost_equal(image.data['sm'][167, 785], 47.69982, 4)
    assert(image.metadata['sm']['_FillValue'] == -9999.)
    assert(image.metadata['sm']['long_name'] == 'Percent of Saturation Soil Moisture')
    #assert(file_meta['creator_name'] == 'Earth Observation Data Center (EODC)')


def test_C33Ts_tcdr_passive_decadal():
    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'TCDR', '062_dekadalImages', 'passive',
                        'C3S-SOILMOISTURE-L3S-SSMV-PASSIVE-DEKADAL-20140101000000-TCDR-v201801.0.0.nc'))


    ds = C3SImg(file, mode='r', parameters='sm', array_1D=False)
    image = ds.read()

    nptest.assert_almost_equal(image['sm'][167, 784], 0.50875, 4)
    assert(image.metadata['sm']['long_name'] == 'Volumetric Soil Moisture')
    #assert(image.metadata['creator_name'] == 'Earth Observation Data Center (EODC)')

################################################################################

def test_C33Ts_icdr_combined_daily():
    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'ICDR', '060_dailyImages', 'combined', '2017',
                        'C3S-SOILMOISTURE-L3S-SSMV-COMBINED-DAILY-20170701000000-ICDR-v201706.0.0.nc'))


    ds = C3SImg(file, mode='r', parameters='sm', array_1D=False)
    image = ds.read()

    nptest.assert_almost_equal(image.data['sm'][167, 785], 0.14548, 4)
    assert(image.metadata['sm']['long_name'] == 'Volumetric Soil Moisture')
    #assert(image['creator_name'] == 'Earth Observation Data Center (EODC)')



def test_C33Ts_icdr_active_monthly():

    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'ICDR', '061_monthlyImages', 'active',
                        'C3S-SOILMOISTURE-L3S-SSMS-ACTIVE-MONTHLY-20170701000000-ICDR-v201706.0.0.nc'))


    ds = C3SImg(file, mode='r', parameters='sm', array_1D=False)
    image = ds.read()

    nptest.assert_almost_equal(image.data['sm'][167, 785], 65.00162, 4)
    assert(image.metadata['sm']['_FillValue'] == -9999.)
    assert(image.metadata['sm']['long_name'] == 'Percent of Saturation Soil Moisture')
    #assert(file_meta['creator_name'] == 'Earth Observation Data Center (EODC)')


def test_C33Ts_icdr_passive_decadal():
    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'ICDR', '062_dekadalImages', 'passive',
                        'C3S-SOILMOISTURE-L3S-SSMV-PASSIVE-DEKADAL-20170701000000-ICDR-v201706.0.0.nc'))


    ds = C3SImg(file, mode='r', parameters='sm', array_1D=False)
    image = ds.read()

    nptest.assert_almost_equal(image.data['sm'][167, 785], 0.21000, 4)
    assert(image.metadata['sm']['long_name'] == 'Volumetric Soil Moisture')
    #assert(image.metadata['creator_name'] == 'Earth Observation Data Center (EODC)')


def test_1Dreading():
    ''' Test 1D reading with and without land grid, and if the results are the same'''
    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'ICDR', '060_dailyImages', 'combined', '2017',
                        'C3S-SOILMOISTURE-L3S-SSMV-COMBINED-DAILY-20170701000000-ICDR-v201706.0.0.nc'))

    ds = C3SImg(file, mode='r', parameters='sm', array_1D=True)
    image = ds.read()

    assert ds.grid.find_nearest_gpi(75.625, 14.625) == (602942, 0)
    ref_sm = image.data['sm'][434462]
    ref_lat = image.lat[434462]
    ref_lon = image.lon[434462]

    assert ref_lat == 14.625
    assert ref_lon == 75.625
    nptest.assert_almost_equal(ref_sm, 0.360762, 5)
    assert(image.metadata['sm']['long_name'] == 'Volumetric Soil Moisture')

    land_grid = C3SLandGrid()

    ds = C3SImg(file, mode='r', parameters='sm', array_1D=True, subgrid=land_grid)
    image = ds.read()

    assert ds.grid.find_nearest_gpi(75.625, 14.625) == (602942, 0)

    sm = image.data['sm'][177048]
    lat = image.lat[177048]
    lon = image.lon[177048]

    assert ref_lat == lat
    assert ref_lon == lon
    nptest.assert_almost_equal(ref_sm, sm, 5)

    assert(image.metadata['sm']['long_name'] == 'Volumetric Soil Moisture')








if __name__ == '__main__':
    test_1Dreading()
    test_C33Ts_tcdr_combined_daily()
    test_C33Ts_tcdr_active_monthly()
    test_C33Ts_tcdr_passive_decadal()

    test_C33Ts_icdr_combined_daily()
    test_C33Ts_icdr_active_monthly()
    test_C33Ts_icdr_passive_decadal()
