# -*- coding: utf-8 -*-

"""
Module to download c3s soil moisture data from the CDS
"""

import cdsapi
from datetime import datetime, timedelta
import calendar
import os
from zipfile import ZipFile
from glob import glob

import pandas as pd
from parse import parse
from dateutil.relativedelta import relativedelta
from cadati.dekad import day2dekad

from c3s_sm.const import fntempl as _default_template
from c3s_sm.const import variable_lut, freq_lut, api_ready, logger

def infer_file_props(path, fntempl=_default_template, start_from='last') -> dict:
    """
    Parse file names to retrieve properties from :func:`c3s_sm.const.fntempl`.
    """
    files = sorted(glob(os.path.join(path, '**', '*.nc')))
    if len(files) == 0:
        raise ValueError(f"No matching files for chosen template found in the directory {path}")
    else:
        if start_from.lower() == 'last':
            files = files[::-1]
        elif start_from.lower() == 'first':
            pass
        else:
            raise NotImplementedError(f"`start_from` must be one of: "
                                      f"`first`, `last`.")
        for f in files[::-1]:
            file_args = parse(fntempl,  os.path.basename(f))
            if file_args is None:
                continue
            return file_args.named

    raise ValueError(f"No matching files for chosen template found in the "
                     f"directory {path}")


def download_c3ssm(c, sensor, years, months, days, version, target_dir,
                   temp_filename, freq='daily', keep_original=False,
                   max_retries=5, dry_run=False):
    """
    Download c3s sm data for single levels of a defined time span
    Parameters. We will always try to download the CDR and ICDR!

    Parameters
    ----------
    c : cdsapi.Client
        Client to pass the request to
    sensor : str
        active, passive or combined. The sensor product to download
    years : list
        Years for which data is downloaded ,e.g. [2017, 2018]
    months : list
        Months for which data is downloaded, e.g. [4, 8, 12]
    days : list
        Days for which data is downloaded (range(31)=All days) e.g. [10, 20, 31]
    version: str
        Version string of data to download, e.g. 'v201706.0.0'
    variables : list, optional (default: None)
        List of variables to pass to the client, if None are passed, the default
        variables will be downloaded.
    target_dir : str
        Directory where the data is downloaded into
    temp_filename : str
        filename of the zip archive that will be downloaded
    freq : str
        daily, dekadal or monthly. Which of the three aggregated products to
        download.
    dry_run : bool, optional (default: False)
        Does not download anything, returns query, success is False

    Returns
    -------
    success : dict[str, bool]
        Indicates whether the download was successful, False for dry_run=True
    queries: dict[str, dict]
        icdr and cdr query that were submitted
    """

    if not api_ready:
        raise ValueError("Cannot establish connection to CDS. Please set up"
                         "your CDS API key as described at "
                         "https://cds.climate.copernicus.eu/api-how-to")


    if not os.path.exists(target_dir):
        raise IOError(f'Target path {target_dir} does not exist.')

    success = {'icdr': False, 'cdr': False}
    queries = {'icdr': None, 'cdr': None}

    for record in ['cdr', 'icdr']:
        dl_file = os.path.join(target_dir, temp_filename)
        os.makedirs(os.path.dirname(dl_file), exist_ok=True)

        i = 0
        while not success[record] and i <= max_retries:
            query = dict(
                name='satellite-soil-moisture',
                request={
                    'variable': variable_lut[sensor]['variable'],
                    'type_of_sensor': variable_lut[sensor]['type_of_sensor'],
                    'time_aggregation': freq_lut[freq],
                    'format': 'zip',
                    'year': [str(y) for y in years],
                    'month': [str(m).zfill(2) for m in months],
                    'day': [str(d).zfill(2) for d in days],
                    'version': version,
                    'type_of_record': record
                },
                target=dl_file
            )

            queries[record] = query

            if not dry_run:
                try:
                    c.retrieve(**query)
                    success[record] = True
                except Exception:
                    # delete the partly downloaded data and retry
                    if os.path.isfile(dl_file):
                        os.remove(dl_file)
                    success[record] = False
                finally:
                    i += 1
            else:
                success[record] = False
                break

        if success[record]:
            with ZipFile(dl_file, 'r') as zip_file:
                zip_file.extractall(target_dir)

            if not keep_original:
                os.remove(dl_file)

    return success, queries

def download_and_extract(target_path,
                         startdate=datetime(1978,1,1),
                         enddate=datetime.now(),
                         product='combined',
                         freq='daily',
                         version='v202212',
                         keep_original=False,
                         dry_run=False):
    """
    Downloads the data from the CDS servers and moves them to the target path.
    This is done in 30 day increments between start and end date.
    The files are then extracted into yearly folders under the target_path.

    Parameters
    ----------
    target_path : str
        Path where the files are stored to
    startdate: datetime, optional (default: datetime(1978,1,1))
        first day to download data for (if available)
    enddate: datetime, optional (default: datetime.now())
        last day to download data for (if available)
    product : str, optional (default: 'combined')
        Product (combined, active, passive) to download
    freq : str, optional (default: 'daily')
        'daily', 'dekadal' or 'monthly' averaged data to download.
    version : str, optional (default: 'v202212')
        Dataset version to download.
    keep_original: bool, optional (default: False)
        Keep the original downloaded data in zip format together with the unzipped
        files.
    dry_run : bool, optional (default: False)
        Does not download anything, returns query, success is False

    Returns:
    -------
    queries: list
        List[dict]: All submitted queries
    """

    product = product.lower()
    if product not in variable_lut.keys():
        raise ValueError(f"{product} is not a supported product. "
                         f"Choose one of {list(variable_lut.keys())}")

    freq = freq.lower()
    if freq not in freq_lut.keys():
        raise ValueError(f"{freq} is not a supported frequency. "
                         f"Choose one of {list(freq_lut.keys())}")

    os.makedirs(target_path, exist_ok=True)

    dl_logger = logger(os.path.join(target_path,
        f"download_{'{:%Y%m%d%H%M%S.%f}'.format(datetime.now())}.log"))

    c = cdsapi.Client(quiet=True,
                      url=os.environ.get('CDSAPI_URL'),
                      key=os.environ.get('CDSAPI_KEY'),
                      error_callback=dl_logger)
    queries = []

    if freq == 'daily':
        curr_start = startdate
        # download monthly zip archives
        while curr_start <= enddate:
            sy, sm, sd = curr_start.year, curr_start.month, curr_start.day
            sm_days = calendar.monthrange(sy, sm)[1]  # days in the current month
            y, m = sy, sm

            if (enddate.year == y) and (enddate.month == m):
                d = enddate.day
            else:
                d = sm_days

            curr_end = datetime(y, m, d)

            fname = (f"{curr_start.strftime('%Y%m%d')}_"
                     f"{curr_end.strftime('%Y%m%d')}.zip")

            target_dir_year = os.path.join(target_path, str(y))
            os.makedirs(target_dir_year, exist_ok=True)

            _, q = download_c3ssm(
                c, product, years=[y], months=[m],
                days=list(range(sd, d+1)), version=version,
                freq=freq, max_retries=3,
                target_dir=target_dir_year, temp_filename=fname,
                keep_original=keep_original, dry_run=dry_run)

            queries.append(q)
            curr_start = curr_end + timedelta(days=1)

    else:
        curr_year = startdate.year
        # download annual zip archives, this means that the day is ignored
        # when downloading monthly/dekadal data.
        if freq == 'monthly':
            ds = [1]
        else:
            ds = [1, 11, 21]

        while curr_year <= enddate.year:

            if curr_year == startdate.year:
                ms = [m for m in range(1, 13) if m >= startdate.month]
            elif curr_year == enddate.year:
                ms = [m for m in range(1, 13) if m <= enddate.month]
            else:
                ms = list(range(1,13))

            curr_start = datetime(curr_year, ms[0],
                startdate.day if curr_year == startdate.year else ds[0])

            while curr_start.day not in ds:
                curr_start += timedelta(days=1)

            curr_end = datetime(curr_year, ms[-1], ds[-1])

            target_dir_year = os.path.join(target_path, str(curr_year))
            os.makedirs(target_dir_year, exist_ok=True)

            fname = f"{curr_start.strftime('%Y%m%d')}_{curr_end.strftime('%Y%m%d')}.zip"

            _, q = download_c3ssm(
                c, product, years=[curr_year], months=ms,
                days=ds, version=version,
                freq=freq, max_retries=3,
                target_dir=target_dir_year, temp_filename=fname,
                keep_original=keep_original, dry_run=dry_run)

            queries.append(q)
            curr_year += 1

    return queries

def first_missing_date(last_date: str,
                       freq: str = 'daily') -> datetime:
    """
    For a product, based on the last available date, find the next
    expected one.
    """
    last_date = pd.to_datetime(last_date).to_pydatetime()
    assert freq in ['daily', 'dekadal', 'monthly'], \
        "Frequency must be daily, dekadal, or monthly"
    if freq == 'daily':
        next_date = last_date + relativedelta(days=1)
    elif freq == 'monthly':
        next_date = last_date + relativedelta(months=1)
    elif freq == 'dekadal':
        this_dekad = day2dekad(last_date.day)
        if last_date.day not in [1, 11, 21]:
            raise ValueError("Dekad day must be 1, 11 or 21")
        if (this_dekad == 1) or (this_dekad == 2):
            next_date = last_date + relativedelta(days=10)
        else:
            next_date = last_date + relativedelta(months=1)
            next_date = datetime(next_date.year, next_date.month, 1)

    return next_date


