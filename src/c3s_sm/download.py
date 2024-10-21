# -*- coding: utf-8 -*-
"""
Module to download c3s soil moisture data from the CDS
"""
import sys
import os
from datetime import datetime, timedelta
import calendar
from zipfile import ZipFile
import logging
import cdsapi
import pandas as pd
from dateutil.relativedelta import relativedelta
from cadati.dekad import day2dekad
from repurpose.process import parallel_process
import traceback

from c3s_sm.const import variable_lut, freq_lut, check_api_read
from c3s_sm.misc import update_image_summary_file, delete_empty_directories


def logger(fname, level=logging.DEBUG, verbose=False):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=fname,
        level=level,
        format='%(levelname)s %(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    if verbose:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.captureWarnings(True)

    assert os.path.exists(fname)

    return logger


def download_c3ssm(c,
                   sensor,
                   years,
                   months,
                   days,
                   version,
                   target_dir,
                   temp_filename,
                   freq='daily',
                   keep_original=False,
                   max_retries=5,
                   dry_run=False):
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
        Version string of data to download, e.g. 'v202212'
    target_dir : str
        Directory where the data is downloaded into
    temp_filename : str
        filename of the zip archive that will be downloaded
    freq : str, optional (default: daily)
        daily, dekadal or monthly. Which of the three aggregated products to
        download.
    keep_original: bool, optional (default: False)
        Whether the original file retrieved from CDS should be kept. If False,
        then only the extracted images are kept.
    max_retries: int, optional (default: 5)
        When a download failes, try again up to max_retries times
    dry_run : bool, optional (default: False)
        Does not download anything, returns query, success is False

    Returns
    -------
    success : dict[str, bool]
        Indicates whether the download was successful, False for dry_run=True
    queries: dict[str, dict]
        icdr and cdr query that were submitted
    """
    logger = logging.getLogger('dl_logger')

    if not dry_run:
        if not check_api_read():
            raise ValueError(
                "Cannot establish connection to CDS. Please set up"
                "your CDS API key as described at "
                "https://cds.climate.copernicus.eu/how-to-api")

        os.makedirs(target_dir, exist_ok=True)

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
                target=dl_file)

            queries[record] = query

            if not dry_run:
                try:
                    c.retrieve(**query)
                    success[record] = True
                except Exception as e:
                    logger.error(f"Error downloading file {dl_file}: {e}")
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
            logger.info(f"Chunk downloaded: {dl_file}")
            with ZipFile(dl_file, 'r') as zip_file:
                zip_file.extractall(target_dir)

            if not keep_original:
                os.remove(dl_file)

    return success, queries


def download_and_extract(target_path,
                         startdate=datetime(1978, 1, 1),
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

    Returns
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

    os.makedirs(os.path.join(target_path, '000_log'), exist_ok=True)

    dl_logger = logger(
        os.path.join(
            target_path, '000_log',
            f"download_{'{:%Y%m%d%H%M%S.%f}'.format(datetime.now())}.log"))

    if dry_run:
        c = None
    else:
        c = cdsapi.Client(
            quiet=True,
            url=os.environ.get('CDSAPI_URL'),
            key=os.environ.get('CDSAPI_KEY'),
            error_callback=dl_logger)

    STATIC_KWARGS = {
        'c': c,
        'keep_original': keep_original,
        'dry_run': dry_run,
        'sensor': product,
        'version': version,
        'freq': freq,
        'max_retries': 3
    }

    ITER_KWARGS = {
        'years': [],
        'months': [],
        'days': [],
        'target_dir': [],
        'temp_filename': []
    }

    if freq == 'daily':
        curr_start = startdate
        # download monthly zip archives
        while curr_start <= enddate:
            sy, sm, sd = curr_start.year, curr_start.month, curr_start.day
            sm_days = calendar.monthrange(sy,
                                          sm)[1]  # days in the current month
            y, m = sy, sm

            if (enddate.year == y) and (enddate.month == m):
                d = enddate.day
            else:
                d = sm_days

            curr_end = datetime(y, m, d)

            fname = (f"{curr_start.strftime('%Y%m%d')}_"
                     f"{curr_end.strftime('%Y%m%d')}.zip")

            target_dir_year = os.path.join(target_path, str(y))

            ITER_KWARGS['years'].append([y])
            ITER_KWARGS['months'].append([m])
            ITER_KWARGS['days'].append(list(range(sd, d + 1)))
            ITER_KWARGS['target_dir'].append(target_dir_year)
            ITER_KWARGS['temp_filename'].append(fname)

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

            if curr_year == startdate.year and curr_year != enddate.year:
                ms = [m for m in range(1, 13) if m >= startdate.month]
            elif curr_year == enddate.year and curr_year != startdate.year:
                ms = [m for m in range(1, 13) if m <= enddate.month]
            elif curr_year == startdate.year and curr_year == enddate.year:
                ms = [
                    m for m in range(1, 13)
                    if ((m >= startdate.month) and (m <= enddate.month))
                ]
            else:
                ms = list(range(1, 13))

            curr_start = datetime(
                curr_year, ms[0],
                startdate.day if curr_year == startdate.year else ds[0])

            while curr_start.day not in ds:
                curr_start += timedelta(days=1)

            curr_end = datetime(curr_year, ms[-1], ds[-1])

            target_dir_year = os.path.join(target_path, str(curr_year))

            fname = f"{curr_start.strftime('%Y%m%d')}_{curr_end.strftime('%Y%m%d')}.zip"

            ITER_KWARGS['years'].append([curr_year])
            ITER_KWARGS['months'].append(ms)
            ITER_KWARGS['days'].append(ds)
            ITER_KWARGS['target_dir'].append(target_dir_year)
            ITER_KWARGS['temp_filename'].append(fname)

            curr_year += 1

    results = parallel_process(
        download_c3ssm,
        STATIC_KWARGS=STATIC_KWARGS,
        ITER_KWARGS=ITER_KWARGS,
        n_proc=1,
        log_path=os.path.join(target_path, '000_log'),
        loglevel='INFO',
        backend='threading',
        logger_name='dl_logger',
        show_progress_bars=True)

    delete_empty_directories(target_path)

    try:
        update_image_summary_file(target_path)
    except ValueError as _:
        dl_logger.error(f"Could not update image summary. "
                        f"Error traceback: {traceback.format_exc()}")

    handlers = dl_logger.handlers[:]

    for handler in handlers:
        dl_logger.removeHandler(handler)
        handler.close()
    handlers.clear()

    success, queries = [r[0] for r in results], [r[1] for r in results]

    return queries


def first_missing_date(last_date: str, freq: str = 'daily') -> datetime:
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