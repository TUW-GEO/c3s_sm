# -*- coding: utf-8 -*-
"""
Module for a command line interface to convert the C3S data into a
time series format using the repurpose package
"""

import os
import sys
import argparse
from datetime import datetime

import pandas as pd
from repurpose.img2ts import Img2Ts
from c3s_sm.interface import C3S_Nc_Img_Stack, fntempl
import c3s_sm.metadata as metadata
from c3s_sm.metadatac import C3S_daily_tsatt_nc, C3S_dekmon_tsatt_nc
from smecv_grid.grid import SMECV_Grid_v052
from parse import parse
from netCDF4 import Dataset

def mkdate(datestring):
    """
    Create date string.

    Parameters
    ----------
    datestring : str
        Date string.

    Returns
    -------
    datestr : datetime
        Date string as datetime.
    """
    if len(datestring) == 10:
        return datetime.strptime(datestring, '%Y-%m-%d')
    if len(datestring) == 16:
        return datetime.strptime(datestring, '%Y-%m-%dT%H:%M')

def str2bool(val):
    if val in ['True', 'true', 't', 'T', '1']:
        return True
    else:
        return False

def parse_filename(data_dir):
    """
    Take the first file in the passed directory and use its file name to
    retrieve the product type, version number and variables in the file.

    Parameters
    ----------
    inroot : str
        Input root directory

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
                file_vars = Dataset(os.path.join(curr,f)).variables.keys()
                return file_args, list(file_vars)

    raise IOError('No file name in passed directory fits to template')


def reshuffle(input_root, outputpath, startdate, enddate,
              parameters=None, land_points=True, bbox=None,
              ignore_meta=False, imgbuffer=500):
    """
    Reshuffle method applied to C3S data.

    Parameters
    ----------
    input_root: string
        input path where c3s images were downloaded.
    outputpath : string
        Output path.
    startdate : datetime
        Start date.
    enddate : datetime
        End date.
    parameters: list, optional (default: None)
        parameters to read and convert
    land_points : bool, optional (default: True)
        Use the land grid to calculate time series on.
        Leads to faster processing and smaller files.
    bbox : tuple, optional (default: None)
        Min lon, min lat, max lon, max lat
        BBox to read data for.
    ignore_meta : bool, optional (default: False)
        Ignore metadata and reshuffle only the values. Can be used e.g. if a
        version is not yet supported.
    imgbuffer: int, optional (default: 50)
        How many images to read at once before writing time series.
    """

    if land_points:
        grid = SMECV_Grid_v052('land')
    else:
        grid = SMECV_Grid_v052(None)

    if bbox:
        grid = grid.subgrid_from_bbox(*bbox)

    if parameters is None:
        file_args, file_vars = parse_filename(input_root)
        parameters = [p for p in file_vars if p not in ['lat', 'lon', 'time']]

    subpath_templ = ('%Y',) if os.path.isdir(os.path.join(input_root, str(startdate.year))) else None
    input_dataset = C3S_Nc_Img_Stack(data_path=input_root,
                                     parameters=parameters,
                                     subgrid=grid,
                                     flatten=True,
                                     fillval=None,
                                     subpath_templ=subpath_templ)

    if not ignore_meta:
        prod_args = input_dataset.fname_args

        kwargs = {'sensor_type': prod_args['prod'].lower(),
                  'cdr_type': prod_args['cdr'],
                  'product_temp_res':  prod_args['temp'],
                  'cls': getattr(metadata, f"C3S_SM_TS_Attrs_{prod_args['vers']}")}

        if prod_args['temp'].upper() == 'DAILY':
            kwargs.pop('product_temp_res')
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

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    reshuffler = Img2Ts(input_dataset=input_dataset, outputpath=outputpath,
                        startdate=startdate, enddate=enddate, input_grid=grid,
                        imgbuffer=imgbuffer, cellsize_lat=5.0,
                        cellsize_lon=5.0, global_attr=global_attributes, zlib=True,
                        unlim_chunksize=1000, ts_attributes=ts_attributes)
    reshuffler.calc()


def parse_args(args):
    """
    Parse command line parameters for C3S reshuffling.

    Parameters
    ----------
    args : list of str
        Command line parameters as list of strings.

    Returns
    -------
    args : argparse.Namespace
        Command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Convert C3s image data to time series format.")
    parser.add_argument("dataset_root",
                        help='Root of local filesystem where the '
                             'data is stored.')

    parser.add_argument("timeseries_root",
                        help='Root of local filesystem where the timeseries '
                             'should be stored.')

    parser.add_argument("start", type=mkdate,
                        help=("Startdate. In format YYYY-MM-DD"))

    parser.add_argument("end", type=mkdate,
                        help=("Enddate. In format YYYY-MM-DD"))

    parser.add_argument("--parameters", metavar="parameters", default=None,
                        nargs="+",
                        help=("Parameters to reshuffle into time series format. "
                              "E.g. sm for creating soil moisture time series."
                              "If None are passed, all variables from the first image file in the path are used."))

    parser.add_argument("--land_points", type=str2bool, default='False',
                        help=("Set True to convert only land points as defined"
                              " in the C3s land mask (faster and less/smaller files)"))

    parser.add_argument("--bbox", type=float, default=None, nargs=4,
                        help=("min_lon min_lat max_lon max_lat. "
                              "Bounding Box (lower left and upper right corner) "
                              "of area to reshuffle (WGS84)"))

    parser.add_argument("--ignore_meta", type=str2bool, default='False',
                        help=("Do not apply image metadata to the time series."
                              "E.g. for unsupported data versions."))

    parser.add_argument("--imgbuffer", type=int, default=200,
                        help=("How many images to read at once. Bigger "
                              "numbers make the conversion faster but "
                              "consume more memory."))

    args = parser.parse_args(args)
    # set defaults that can not be handled by argparse

    print(f"Converting data from {args.start.isoformat()} to"
          f" {args.end.isoformat()} into folder {args.timeseries_root}.")

    return args


def main(args):
    """
    Main routine used for command line interface.
    Parameters
    ----------
    args : list of str
        Command line arguments.
    """
    args = parse_args(args)

    reshuffle(args.dataset_root,
              args.timeseries_root,
              args.start,
              args.end,
              args.parameters,
              land_points=args.land_points,
              bbox=args.bbox,
              ignore_meta=args.ignore_meta,
              imgbuffer=args.imgbuffer)

def run():
    main(sys.argv[1:])
