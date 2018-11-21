# -*- coding: utf-8 -*-

import os
import glob
import tempfile
import numpy as np
import numpy.testing as nptest

from c3s_sm.reshuffle import main, parse_filename
from c3s_sm.interface import C3STs

def test_parse_filename():
    inpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "c3s_sm-test-data", "img2ts", "ICDR", "combined")

    file_args, file_vars = parse_filename(inpath)


    assert file_args['product'] == 'C3S'
    assert file_args['data_type'] == 'SSMV'
    assert file_args['sensor_type'] == 'COMBINED'
    assert file_args['temp_res'] == 'MONTHLY'
    assert file_args['sub_prod'] == 'ICDR'
    assert file_args['version'] == 'v201706'
    assert file_args['sub_version'] == '0.0'

    assert file_vars == [u'lat', u'lon', u'time', u'nobs', u'sensor', u'freqbandID', u'sm']




def test_reshuffle_TCDR_daily():
    inpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "c3s_sm-test-data", "img2ts", "TCDR", "active")
    startdate = "1991-08-05"
    enddate = "1991-08-08"
    #parameters = ["sm", "sm_uncertainty", "dnflag", "flag", "freqbandID", "mode", "sensor", "t0"]
    parameters = ['--parameters', 'sm', 'sm_uncertainty']
    land_points = 'True'

    ts_path = tempfile.mkdtemp()
    args = [inpath, ts_path, startdate, enddate]  + \
           parameters + ['--land_points', land_points]
    main(args)

    assert len(glob.glob(os.path.join(ts_path, "*.nc"))) == 1002

    ds = C3STs(ts_path, remove_nans=True)
    ts = ds.read(75.625, 14.625)
    ts_sm_values_should = np.array([66.0677, np.nan, 80.7060, 70.5648], dtype=np.float32)
    nptest.assert_allclose(ts['sm'].values, ts_sm_values_should, rtol=1e-5)

    ts_uncert_values_should = np.array([np.nan, np.nan, np.nan, np.nan],
                                       dtype=np.float32)
    nptest.assert_allclose(ts['sm_uncertainty'].values, ts_uncert_values_should,rtol=1e-5)

    nptest.assert_almost_equal(ts['sm'].values, ds.read(602942)['sm'].values)

    #nptest.assert_allclose(ts['dnflag'].values, [1,0,2,1],rtol=1e-5)
    #nptest.assert_allclose(ts['flag'].values, [0,127,0,0],rtol=1e-5)
    #nptest.assert_allclose(ts['freqbandID'].values, [2,0,2,2],rtol=1e-5)
    #nptest.assert_allclose(ts['mode'].values, [2,0,1,2],rtol=1e-5)
    #nptest.assert_allclose(ts['sensor'].values, [128,0, 128,128],rtol=1e-5)
    #nptest.assert_almost_equal(ts['t0'].values, [7886.22702546, -3440586.5, 7887.7328588,
     #                                     7889.22702546])

    ds.close()

def test_reshuffle_ICDR_monthly():
    inpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "c3s_sm-test-data", "img2ts", "ICDR", "combined")
    startdate = "2018-05-01"
    enddate = "2018-08-01"
    parameters = ['--parameters', "sm", "sensor"]

    land_points = 'False'

    ts_path = tempfile.mkdtemp()
    args = [inpath, ts_path, startdate, enddate] \
           + parameters + ['--land_points', land_points]
    main(args)


    assert len(glob.glob(os.path.join(ts_path, "*.nc"))) == 2593

    ds = C3STs(ts_path, remove_nans=True)
    ts = ds.read(-159.625, 65.875)
    ts_sm_values_should = np.array([0.23628984, 0.33424062, np.nan, 0.26261818], dtype=np.float32)

    nptest.assert_allclose(ts['sm'].values, ts_sm_values_should, rtol=1e-5)

    ts_sensor_values_should = np.array([768, 768, 768, 768 ], dtype=np.float32)
    nptest.assert_allclose(ts['sensor'].values, ts_sensor_values_should,rtol=1e-5)

    ds.close()



if __name__ == '__main__':
    test_parse_filename()
    test_reshuffle_TCDR_daily()
    test_reshuffle_ICDR_monthly()

