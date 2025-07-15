# -*- coding: utf-8 -*-
import datetime
import os
import glob
from tempfile import TemporaryDirectory
import numpy as np
import numpy.testing as nptest
from netCDF4 import Dataset

from c3s_sm.misc import img_infer_file_props, read_summary_yml, _default_template
from c3s_sm.interface import C3STs
from c3s_sm.reshuffle import img2ts, extend_ts
import pandas as pd
import pytest
import subprocess

testdata_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "c3s_sm-test-data")

def test_parse_filename():
    file_args = img_infer_file_props(
        os.path.join(testdata_path, 'img2ts', 'combined'),
        start_from='first')

    assert file_args['unit'] == 'V'
    assert file_args['product'] == 'COMBINED'
    assert file_args['freq'] == 'MONTHLY'
    assert file_args['record'] == 'TCDR'
    assert file_args['version'] == 'v201912'
    assert file_args['subversion'] == '0.0'

@pytest.mark.parametrize("interface", ["cmd", "python"])
def test_reshuffle_TCDR_daily_multiple_params(interface):
    inpath = os.path.join(testdata_path, "img2ts", "active")
    startdate = "1991-08-05"
    enddate = "1991-08-08"
    n_proc = 2

    with TemporaryDirectory() as ts_path:
        os.environ["C3S_SM_NO_IMAGE_BASE_CONNECTION"] = "1"

        if interface == "python":
            img2ts(inpath, ts_path, startdate, enddate,
                   ['sm', 'sm_uncertainty'], land_points=True,
                   bbox=[70, 10, 80, 20], n_proc=n_proc, ignore_meta=False)
            extend_ts(inpath, ts_path)
        elif interface == "cmd":
            args = [inpath, ts_path] \
                   + ['-s', startdate] \
                   + ['-e', enddate] \
                   + ['-p', 'sm', '-p', 'sm_uncertainty'] \
                   + ['--land', 'True'] \
                   + ['--ignore_meta', 'False'] \
                   + ['--bbox', '70', '10', '80', '20'] \
                   + ['--n_proc', str(n_proc)]
            subprocess.call(['c3s_sm', 'reshuffle', *args])
            subprocess.call(['c3s_sm', 'update_ts', *[inpath, ts_path]])
        else:
            raise NotImplementedError()

        i = os.environ.pop("C3S_SM_NO_IMAGE_BASE_CONNECTION")
        assert int(i) == 1

        assert len(glob.glob(os.path.join(ts_path, "*.nc"))) == 5

        ds = C3STs(ts_path, remove_nans=True,
                   parameters=['sm', 'sm_uncertainty'],
                   ioclass_kws={'read_bulk': True, 'read_dates': False})
        loc = 75.625, 14.625
        cell = ds.grid.gpi2cell(ds.grid.find_nearest_gpi(*loc)[0])
        with Dataset(os.path.join(ts_path, f'{cell:04}.nc')) as xrds:
            assert xrds['sm'].getncattr('units') == "percentage (%)"
            assert xrds['sm_uncertainty'].getncattr('name') == "sm_uncertainty"
        ts = ds.read(*loc)

        assert not any(ts['sm'] == 0)
        assert isinstance(ts.index, pd.DatetimeIndex)
        ts_sm_values_should = np.array([66.0677, np.nan, 80.7060, 70.5648, np.nan], dtype=np.float32)
        nptest.assert_allclose(ts['sm'].values, ts_sm_values_should, rtol=1e-3)

        ts_uncert_values_should = np.array([np.nan, np.nan, np.nan, np.nan, np.nan],
                                           dtype=np.float32)
        nptest.assert_allclose(ts['sm_uncertainty'].values, ts_uncert_values_should,rtol=1e-5)

        nptest.assert_almost_equal(ts['sm'].values, ds.read(602942)['sm'].values, 3)

        props = read_summary_yml(ts_path)
        #assert props['img2ts_kwargs']['startdate'] == pd.to_datetime(startdate).to_pydatetime()
        assert props['img2ts_kwargs']['enddate'] == datetime.datetime(1991, 8, 9)
        assert props['freq'] == "DAILY"
        assert props['version'] == 'v201801'
        assert props['sensor_type'] == 'active'

        ds.close()

@pytest.mark.parametrize("meta", ["ignore_meta", "include_meta"])
@pytest.mark.parametrize("interface", ["cmd", "python"])
def test_reshuffle_ICDR_monthly_single_param(meta, interface):
    inpath = os.path.join(testdata_path, "img2ts", "combined")

    if meta == 'ignore_meta':
        ignore_meta = True
    elif meta == 'include_meta':
        ignore_meta = False
    else:
        raise NotImplementedError("Unknown meta")

    startdate = "2019-10-01"
    enddate = "2019-12-01"   # last file 2020-01-01

    n_proc = 1

    a = subprocess.call(['c3s_sm', 'reshuffle', "--help"])
    assert a == 0

    os.environ["C3S_SM_NO_IMAGE_BASE_CONNECTION"] = "1"

    with TemporaryDirectory() as ts_path:
        freq = None if not ignore_meta else "MONTHLY"
        if interface == 'cmd':
            args = [inpath, ts_path] \
                   + ['-s', startdate] \
                   + ['-e', enddate] \
                   + ['--land', 'False'] \
                   + ['--bbox', '-10', '40', '10', '50'] \
                   + ['--ignore_meta', str(ignore_meta)] \
                   + ['--imgbuffer', '100'] \
                   + ['--n_proc', str(n_proc)]

            subprocess.call(['c3s_sm', 'reshuffle', *args])
            params = [inpath, ts_path]
            if freq is not None:
                params += ['--freq', str(freq)]
            subprocess.call(['c3s_sm', 'update_ts', *params])
            update_args = subprocess.run(
                ['c3s_sm', 'update_ts', *params, "--dry-run", "True"],
                     text=True, stdout=subprocess.PIPE)

            imgpath, tspath, tagg = update_args.stdout \
                .replace("\n", "") \
                .split(' ')
            assert imgpath == inpath
            assert tspath == ts_path
            assert tagg == str(freq)


        elif interface == "python":
            img2ts(inpath, ts_path, startdate, enddate,
                   bbox=[-10, 40, 10, 50], ignore_meta=ignore_meta,
                   imgbuffer=100, n_proc=n_proc, land_points=False)
            extend_ts(inpath, ts_path, freq=freq)

        i = os.environ.pop("C3S_SM_NO_IMAGE_BASE_CONNECTION")
        assert int(i) == 1

        assert len(glob.glob(os.path.join(ts_path, "*.nc"))) == 9

        ds = C3STs(ts_path, remove_nans=True, parameters=None,
                   ioclass_kws={'read_bulk': True, 'read_dates': False})
        loc = 4.125, 46.875
        cell = ds.grid.gpi2cell(ds.grid.find_nearest_gpi(*loc)[0])
        with Dataset(os.path.join(ts_path, f'{cell:04}.nc')) as xrds:
            if ignore_meta:
                with pytest.raises(AttributeError):
                    _ = xrds['sm'].getncattr('units')
            else:
                assert xrds['sm'].getncattr('units') == "m3 m-3"
                assert xrds['freqbandID'].getncattr('name') == "freqbandID"
        ts = ds.read(*loc)
        assert not np.any(ts['sm'] == 0)  # in corrupt file
        assert not np.any(ts['sensor'] < 0)  # in corrupt file
        assert isinstance(ts.index, pd.DatetimeIndex)
        assert ts.index.size == 3
        ts_sm_values_should = np.array([0.291388, 0.328116, 0.316130], dtype=np.float32)
        nptest.assert_allclose(ts['sm'].values, ts_sm_values_should, rtol=1e-5)

        ts_sensor_values_should = np.array([768, 768, 256], dtype=np.float32)
        nptest.assert_allclose(ts['sensor'].values, ts_sensor_values_should, rtol=1e-5)

        props = read_summary_yml(ts_path)
        #assert props['img2ts_kwargs']['startdate'] == pd.to_datetime(startdate).to_pydatetime()
        assert props['img2ts_kwargs']['enddate'] == datetime.datetime(2020,1,1)
        if ignore_meta:
            assert props['version'] == 'unknown'
            assert props['freq'] == 'unknown'
            assert props['sensor_type'] == 'unknown'

        ds.close()



