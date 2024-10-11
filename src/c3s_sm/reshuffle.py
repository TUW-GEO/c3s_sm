# -*- coding: utf-8 -*-
"""
Module for a command line interface to convert the C3S data into a
time series format using the repurpose package
"""

import os
import shutil
import warnings
from datetime import datetime
import pandas as pd
import pygeogrids
from smecv_grid.grid import SMECV_Grid_v052
from parse import parse
from netCDF4 import Dataset
import numpy as np
from dateutil.relativedelta import relativedelta
from pathlib import Path

from repurpose.img2ts import Img2Ts
from repurpose.process import ImageBaseConnection

from c3s_sm.interface import C3S_Nc_Img_Stack
from c3s_sm.const import fntempl as _default_template
import c3s_sm.metadata as metadata
from c3s_sm.metadata import C3S_daily_tsatt_nc, C3S_dekmon_tsatt_nc
from c3s_sm.misc import (
    update_ts_summary_file,
    read_summary_yml,
    update_image_summary_file,
)


def reshuffle(*args, **kwargs):
    warnings.warn(
        "`c3s_sm.reshuffle.reshuffle` is deprecated, "
        "use `c3s_sm.reshuffle.img2ts`",
        category=DeprecationWarning)
    return img2ts(*args, **kwargs)


def parse_filename(data_dir, fntempl=_default_template):
    """
    Take the first file in the passed directory and use its file name to
    retrieve the product type, version number and variables in the file.

    Parameters
    ----------
    inroot : str
        Input root directory
    fntempl: str, optional (default: :const:`c3s_sm.const.fntempl`)
        Filename template

    Returns
    -------
    file_args : dict
        Parsed arguments from file name
    file_vars : list
        Names of parameters in the first detected file
    """

    for curr, subdirs, files in os.walk(data_dir):
        for f in sorted(files):
            file_args = parse(fntempl, f)
            if file_args is None:
                continue
            else:
                file_args = file_args.named
                file_args['datetime'] = '{datetime}'
                file_vars = Dataset(os.path.join(curr, f)).variables.keys()
                return file_args, list(file_vars)

    raise IOError('No file name in passed directory fits to template')


def extend_ts(img_path,
              ts_path,
              fntempl=_default_template,
              startdate=None,
              freq=None,
              n_proc=1):
    """
    Append any new data from the image path to the time series data.
    This function is only applied to time series file that were created
    using the img2ts function.
    This will use the start date from each cell, and process only
    parameters that are already present in the time series files.

    Parameters
    ----------
    img_path: str
        Path where the image files are stored
    ts_path: str
        Path where the time series files to append to are stored
    fntempl: str, optional (default: see :const:`c3s_sm.const.fntempl`)
        Template that image files follow, must contain a section {datetime}
        where the date is parsed from.
    startdate: str, optional (default: None)
        Date of the first image to append. If None, then we use the next
        available image to the last date in the time series.
    freq: str, optional (default: None)
        DAILY, DEKADAL or MONTHLY. If None is passed, freq must be inferable
        from metadata.
    n_proc: int, optional (default: 1)
        Number of parallel processes to read and write data.
    """
    ts_props = read_summary_yml(ts_path)

    try:
        img_props = read_summary_yml(img_path)
    except FileNotFoundError:
        update_image_summary_file(img_path)
        img_props = read_summary_yml(img_path)

    freq = ts_props['freq'] if freq is None else freq
    kwargs = ts_props['img2ts_kwargs']

    if startdate is None:
        if freq.upper() == 'DAILY':
            dt = relativedelta(days=1)
        elif freq.upper() == 'DEKADAL':
            dt = relativedelta(days=10)
        elif freq.upper() == 'MONTHLY':
            dt = relativedelta(months=1)
        else:
            raise ValueError(
                f'Unexpected frequency found: {freq}. One of daily, '
                f'dekadal, monthly is expected.')

        startdate = pd.to_datetime(kwargs['enddate']) + dt
    else:
        startdate = pd.to_datetime(startdate)

    enddate = pd.to_datetime(img_props['period_to'])

    kwargs["startdate"] = startdate
    kwargs["enddate"] = enddate
    kwargs["img_path"] = img_path
    kwargs['fntempl'] = fntempl

    img2ts(ts_path=ts_path, n_proc=n_proc, **kwargs)


def img2ts(img_path,
           ts_path,
           startdate,
           enddate,
           parameters=None,
           land_points=True,
           bbox=None,
           cells=None,
           ignore_meta=False,
           fntempl=_default_template,
           overwrite=False,
           imgbuffer=250,
           n_proc=1):
    """
    Reshuffle method applied to C3S data.

    Parameters
    ----------
    img_path: str
        input path where c3s images were downloaded.
    ts_path : str, optional (default: None)
        Output path.
    startdate : datetime or str
        Start date. If None is passed, then we will try to detect
        the start of the available image files in img_path
    enddate : datetime or str
        End date. If None is passed, then we will try to detect the date of
        the last available image file
    parameters: list, optional (default: None)
        parameters to read and convert to time series. If None is passed, then
        we use all available parameters from the input files.
    land_points : bool, optional (default: True)
        Use the land grid to calculate time series on.
        Leads to faster processing and smaller files.
    bbox : tuple, optional (default: None)
        Min lon, min lat, max lon, max lat
        BBox to read data for. Data outside the bbox is ignored in the
        conversion step. This option is incompatible with the cells keyword.
    cells: tuple, optional (default: None)
        To limit the processing to certain cells. This option is incompatible
        with the bbox keyword.
    ignore_meta : bool, optional (default: False)
        Ignore metadata and reshuffle only the values. Can be used e.g. if a
        version is not yet supported.
    fntempl: str, optional (default: see :const:`c3s_sm.const.fntempl`)
        Template that image files follow, must contain a section {datetime}
        where the date is parsed from.
    overwrite: bool, optional (default: False)
        If this option is activated, then any existing files on the output
        directory will be deleted before the conversion takes place.
        Otherwise the program will try to append new data to existing files.
        To extend existing time series, it is recommended to use the `extend_ts`
        function instead.
    imgbuffer: int, optional (default: 250)
        How many images to read at once before writing time series.
    n_proc: int, optional (default: 1)
        Number of parallel processes to read and write data.
    """
    if Path(img_path) in Path(ts_path).parents:
        raise ValueError("The time series directory can not be a subdirectory "
                         "of the image directory.")

    grid = SMECV_Grid_v052('land') if land_points else SMECV_Grid_v052(None)

    if (bbox is not None) and (cells is not None):
        raise ValueError(
            "Please either pass a bounding box or cells, not both")

    if bbox:
        grid = grid.subgrid_from_bbox(*bbox)
    if cells:
        grid = grid.subgrid_from_cells(cells)

    if parameters is None:
        file_args, file_vars = parse_filename(img_path, fntempl=fntempl)
        parameters = [p for p in file_vars if p not in ['lat', 'lon', 'time']]

    startdate = pd.to_datetime(startdate).to_pydatetime()
    enddate = pd.to_datetime(enddate).to_pydatetime()

    subpath_templ = ('%Y',) if os.path.isdir(
        os.path.join(img_path, str(startdate.year))) else None
    input_dataset = C3S_Nc_Img_Stack(
        data_path=img_path,
        parameters=parameters,
        subgrid=grid,
        flatten=True,
        fillval={
            'sm': np.nan,
            'flag': 2**8
        },
        fntempl=fntempl,
        subpath_templ=subpath_templ)

    props = {'freq': 'unknown', 'sensor_type': 'unknown', 'version': 'unknown'}

    if not ignore_meta:
        prod_args = input_dataset.fname_args
        freq = prod_args['freq']
        kwargs = {
            'sensor_type': prod_args['product'].lower(),
            'cdr_type': prod_args['record'],
            'freq': freq,
            'cls': getattr(metadata, f"C3S_SM_TS_Attrs"),
            'version': prod_args['version']
        }

        props['sensor_type'] = kwargs['sensor_type']
        props['version'] = prod_args['version']
        props['freq'] = freq.upper()

        if freq.upper() == 'DAILY':
            kwargs.pop('freq')
            attrs = C3S_daily_tsatt_nc(**kwargs)
        else:
            attrs = C3S_dekmon_tsatt_nc(**kwargs)

        ts_attributes = {}
        global_attributes = attrs.global_attr

        for var in parameters:
            ts_attributes[var] = attrs.ts_attributes[var]
    else:
        global_attributes = None
        ts_attributes = None

    if overwrite:
        if os.path.exists(ts_path):
            shutil.rmtree(ts_path)

    if not os.path.exists(ts_path):
        os.makedirs(ts_path)

    # secret switch to circumvent imagebase connection (to speed up tests)
    if os.environ.get("C3S_SM_NO_IMAGE_BASE_CONNECTION", "0") == "1":
        pass
    else:
        input_dataset = ImageBaseConnection(input_dataset)

    if isinstance(grid, pygeogrids.CellGrid):
        _cellsize = None
    else:
        _cellsize = 5

    reshuffler = Img2Ts(
        input_dataset=input_dataset,
        outputpath=ts_path,
        startdate=startdate,
        enddate=enddate,
        input_grid=grid,
        imgbuffer=imgbuffer,
        cellsize_lat=_cellsize,
        cellsize_lon=_cellsize,
        global_attr=global_attributes,
        zlib=True,
        unlim_chunksize=1000,
        ts_attributes=ts_attributes,
        n_proc=n_proc,
        backend='multiprocessing')

    reshuffler.calc()

    kwargs = {
        'parameters': list(parameters),
        'land_points': land_points,
        'enddate': enddate,
        'img_path': img_path,
        'cells': None if cells is None else list(cells),
        'bbox': None if bbox is None else list(bbox),
        "fntempl": fntempl,
        'ignore_meta': ignore_meta
    }

    props["img2ts_kwargs"] = kwargs

    update_ts_summary_file(ts_path, collect_cov=False, props=props)
