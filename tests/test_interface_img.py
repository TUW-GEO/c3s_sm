# -*- coding: utf-8 -*-

from c3s_sm.interface import C3SImg
import os
import numpy.testing as nptest
from smecv_grid.grid import SMECV_Grid_v052
import numpy as np
import pytest


def test_C3STs_tcdr_combined_daily():
    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'TCDR', '060_dailyImages', 'combined', '2014',
                        'C3S-SOILMOISTURE-L3S-SSMV-COMBINED-DAILY-20140101000000-TCDR-v201801.0.0.nc'))


    ds = C3SImg(file, mode='r', parameters=None, flatten=False, fillval={'sm': np.nan})
    image= ds.read()

    test_loc_lonlat = (16.375, 48.125)
    row, col = np.where((image.lon==test_loc_lonlat[0]) & (image.lat==test_loc_lonlat[1]))

    nptest.assert_almost_equal(image.data['sm'][row, col], 0.34659, 4)
    assert(image.metadata['sm']['long_name'] == 'Volumetric Soil Moisture')


def test_C3STs_tcdr_active_monthly():
    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'TCDR', '061_monthlyImages', 'active',
                        'C3S-SOILMOISTURE-L3S-SSMS-ACTIVE-MONTHLY-20140101000000-TCDR-v201801.0.0.nc'))

    ds = C3SImg(file, mode='r', parameters='sm', flatten=False, fillval=None,
                subgrid=SMECV_Grid_v052(None).subgrid_from_bbox(-181,-91, 181,91))

    image = ds.read()
    
    test_loc_lonlat = (16.375, 48.125)
    row, col = np.where((image.lon==test_loc_lonlat[0]) & (image.lat==test_loc_lonlat[1]))

    assert image.data['sm'].shape == (720,1440)
    nptest.assert_almost_equal(image.data['sm'][row, col], 47.69982, 4)
    assert(image.metadata['sm']['_FillValue'] == -9999.)
    assert image.data['sm'].min() == image.metadata['sm']['_FillValue']
    assert(image.metadata['sm']['long_name'] == 'Percent of Saturation Soil Moisture')


def test_C3STs_tcdr_passive_decadal():
    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'TCDR', '062_dekadalImages', 'passive',
                        'C3S-SOILMOISTURE-L3S-SSMV-PASSIVE-DEKADAL-20140101000000-TCDR-v201801.0.0.nc'))

    ds = C3SImg(file, mode='r', flatten=False, fillval={'nobs': -1, 'sm': np.nan},
                subgrid=SMECV_Grid_v052('landcover_class', subset_value=[10,11,60,70]).subgrid_from_bbox(-14, 30, 44, 73))
    image = ds.read()

    test_loc_lonlat = (16.125, 48.125)
    row, col = np.where((image.lon==test_loc_lonlat[0]) & (image.lat==test_loc_lonlat[1]))

    assert image['nobs'].min() == -1
    assert np.any(np.isnan(image['sm']))
    nptest.assert_almost_equal(image['sm'][row, col], 0.50875, 4)
    assert(image.metadata['sm']['long_name'] == 'Volumetric Soil Moisture')

def test_C3STs_icdr_combined_daily():
    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'ICDR', '060_dailyImages', 'combined', '2017',
                        'C3S-SOILMOISTURE-L3S-SSMV-COMBINED-DAILY-20170701000000-ICDR-v201706.0.0.nc'))


    ds = C3SImg(file, mode='r', parameters=['sm', 't0'], flatten=False,
                subgrid=SMECV_Grid_v052('land'))
    image = ds.read()

    test_loc_lonlat = (16.375, 48.125)
    row, col = np.where((image.lon==test_loc_lonlat[0]) & (image.lat==test_loc_lonlat[1]))

    nptest.assert_almost_equal(image.data['sm'][row, col], 0.14548, 4)
    assert(image.metadata['t0']['long_name'] == 'Observation Timestamp')

def test_C3STs_icdr_active_monthly():

    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'ICDR', '061_monthlyImages', 'active',
                        'C3S-SOILMOISTURE-L3S-SSMS-ACTIVE-MONTHLY-20170701000000-ICDR-v201706.0.0.nc'))

    ds = C3SImg(file, mode='r', parameters=['sm', 'sensor'], flatten=False, fillval=-1)
    image = ds.read()

    assert image['sensor'].min() == image['sm'].min() == -1
    test_loc_lonlat = (16.375, 48.125)
    row, col = np.where((image.lon==test_loc_lonlat[0]) & (image.lat==test_loc_lonlat[1]))
    nptest.assert_almost_equal(image.data['sm'][row, col], 65.00162, 4)
    assert(image.metadata['sm']['_FillValue'] == -9999.)
    assert(image.metadata['sm']['long_name'] == 'Percent of Saturation Soil Moisture')


def test_C3STs_icdr_passive_decadal():
    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'ICDR', '062_dekadalImages', 'passive',
                        'C3S-SOILMOISTURE-L3S-SSMV-PASSIVE-DEKADAL-20170701000000-ICDR-v201706.0.0.nc'))

    ds = C3SImg(file, mode='r', parameters='sm', flatten=False, fillval=np.nan)
    image = ds.read()

    test_loc_lonlat = (16.375, 48.125)
    row, col = np.where((image.lon==test_loc_lonlat[0]) & (image.lat==test_loc_lonlat[1]))

    nptest.assert_almost_equal(image.data['sm'][row, col], 0.21000, 4)
    assert(image.metadata['sm']['long_name'] == 'Volumetric Soil Moisture')

@pytest.mark.parametrize("subgrid,",
                         [(SMECV_Grid_v052(None)),
                          (SMECV_Grid_v052('land')),
                          (SMECV_Grid_v052('landcover_class', subset_value=[10,11])),
                          (SMECV_Grid_v052('land').subgrid_from_bbox(74, 13, 78, 15))])
def test_1Dreading(subgrid):
    # Test 1D reading with and without land grid, and if the results are the same

    file = os.path.join(os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'ICDR', '060_dailyImages', 'combined', '2017',
                        'C3S-SOILMOISTURE-L3S-SSMV-COMBINED-DAILY-20170701000000-ICDR-v201706.0.0.nc'))

    ds = C3SImg(file, mode='r', parameters=None, flatten=True, subgrid=subgrid)
    image = ds.read()

    test_loc_lonlat = (75.625, 14.625)
    idx = np.where((image.lon==test_loc_lonlat[0]) & (image.lat==test_loc_lonlat[1]))[0]

    assert ds.subgrid.find_nearest_gpi(*test_loc_lonlat) == (602942, 0)
    ref_sm = image.data['sm'][idx]
    ref_lat = image.lat[idx]
    ref_lon = image.lon[idx]

    assert ref_lat == test_loc_lonlat[1]
    assert ref_lon == test_loc_lonlat[0]
    nptest.assert_almost_equal(ref_sm, 0.360762, 5)
    assert(image.metadata['sm']['long_name'] == 'Volumetric Soil Moisture')
