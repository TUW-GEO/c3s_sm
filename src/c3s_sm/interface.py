# The MIT License (MIT)
#
# Copyright (c) 2018, TU Wien
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Readers for the C3S soil moisture products daily, dekadal (10-daily) and monthly
images as well as for timeseries generated using this module
'''

import pandas as pd
import os
import netCDF4 as nc
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from netCDF4 import num2date

from smecv_grid.grid import SMECV_Grid_v052
from pygeobase.object_base import Image
from pygeobase.io_base import ImageBase
from pygeobase.io_base import MultiTemporalImageBase
from pygeogrids.netcdf import load_grid

import warnings
from netCDF4 import Dataset
from pynetcf.time_series import GriddedNcOrthoMultiTs
from datetime import datetime
from parse import parse

try:
    import xarray as xr

    xr_supported = True
except ImportError:
    xr_supported = False

fntempl = "C3S-SOILMOISTURE-L3S-SSM{unit}-{prod}-{temp}-{datetime}-{cdr}-{vers}.{subvers}.nc"


class C3SImg(ImageBase):
    """
    Class to read a single C3S image (for one time stamp)
    """

    def __init__(self,
                 filename,
                 parameters=None,
                 mode='r',
                 subgrid=SMECV_Grid_v052(None),
                 flatten=False,
                 fillval=None):
        """
        Parameters
        ----------
        filename : str
            Path to the file to read
        parameters : str or Iterable, optional (default: 'sm')
            Names of parameters in the file to read.
            If None are passed, all are read.
        mode : str, optional (default: 'r')
            Netcdf file mode, choosing something different to r may delete data.
        subgrid : SMECV_Grid_v052
            A subgrid of points to read. All other GPIS are masked (2d reading)
            or ignored (when flattened).
        flatten: bool, optional (default: False)
            If set then the data is read into 1D arrays. This is used to e.g
            reshuffle the data for a subset of points.
        fillval : float or dict or None, optional (default: np.nan)
            Fill Value for masked pixels, if a dict is passed, this can be
            set for each parameter individually, otherwise it applies to all.
            Note that choosing np.nan can lead to a change in dtype for some
            (int) parameters. None will use the fill value from the netcdf file
        """
        self.path = os.path.dirname(filename)
        self.fname = os.path.basename(filename)

        super(C3SImg, self).__init__(os.path.join(self.path, self.fname), mode=mode)

        if parameters is None:
            parameters = []
        if type(parameters) != list:
            parameters = [parameters]

        self.parameters = parameters

        self.subgrid = subgrid  # subset to read
        self.grid = SMECV_Grid_v052(None)  # global input image

        self.flatten = flatten

        self.image_missing = False
        self.img = None  # to be loaded
        self.glob_attrs = None

        if isinstance(fillval, dict):
            self.fillval = fillval
            for p in self.parameters:
                if p not in self.fillval:
                    self.fillval[p] = None
        else:
            self.fillval = {p: fillval for p in self.parameters}

    def _read_flat_img(self) -> (dict, dict, dict, datetime):
        """
        Reads a single C3S image, flat with gpi0 as first element
        """
        with Dataset(self.filename, mode='r') as ds:
            timestamp = num2date(ds['time'], ds['time'].units,
                                 only_use_cftime_datetimes=True,
                                 only_use_python_datetimes=False)

            assert len(timestamp) == 1, "Found more than 1 time stamps in image"
            timestamp = timestamp[0]

            param_img = {}
            param_meta = {}

            if len(self.parameters) == 0:
                # all data vars, exclude coord vars
                self.parameters = [k for k in ds.variables.keys()
                                   if k not in ds.dimensions.keys()]

            parameters = list(self.parameters)

            for parameter in parameters:
                metadata = {}
                param = ds.variables[parameter]
                data = param[:][0]  # there is only 1 time stamp in the image

                self.shape = (data.shape[0], data.shape[1])

                # read long name, FillValue and unit
                for attr in param.ncattrs():
                    metadata[attr] = param.getncattr(attr)

                if parameter in self.fillval:
                    if self.fillval[parameter] is None:
                        self.fillval[parameter] = data.fill_value

                    common_dtype = np.find_common_type(
                        array_types=[data.dtype],
                        scalar_types=[type(self.fillval[parameter])])
                    self.fillval[parameter] = np.array([self.fillval[parameter]],
                                                       dtype=common_dtype)[0]

                    data = data.astype(common_dtype)
                    data = data.filled(self.fillval[parameter])
                else:
                    self.fillval[parameter] = data.fill_value
                    data = data.filled()

                data = np.flipud(data)
                data = data.flatten()

                metadata['image_missing'] = 0

                param_img[parameter] = data
                param_meta[parameter] = metadata

            global_attrs = ds.__dict__
            global_attrs['timestamp'] = str(timestamp)

        return param_img, param_meta, global_attrs, timestamp

    def _mask_and_reshape(self,
                          data: dict) -> dict:
        """
        Takes the grid and drops points that are not active.
        for flattened arrays that means that only the active gpis are kept.
        for 2 arrays inactive gpis are set to nan.

        Parameters
        ----------
        data: dict
            Variable names and flattened image data.
        shape_2d : tuple
            2d shape of the original image.

        Returns
        -------
        dat : dict
            Masked, reshaped data.
        """

        # check if flatten. if flatten, dont crop and dont reshape
        # if not flatten, reshape based on grid shape.

        # select active gpis
        for param, dat in data.items():
            if self.flatten:
                dat = dat[self.subgrid.activegpis]
            else:
                exclude = (~np.isin(self.grid.gpis, self.subgrid.activegpis))
                dat[exclude] = self.fillval[param]
                if len(self.shape) != 2:
                    raise ValueError(
                        "Reading 2d image needs grid with 2d shape"
                        "You can either use the global grid without subsets,"
                        "or make sure that you create a subgrid from bbox in"
                        "an area where no gpis are missing.")
                dat = dat.reshape(self.shape)

            data[param] = dat

        return data

    def read(self, timestamp=None):
        """
        Read a single C3S image, if it exists, otherwise fill an empty image.

        Parameters
        ----------
        timestamp : datetime, optional (default: None)
            Time stamp of the image, if this is passed, it is compared to
            the time stamp from the loaded file and must match
        """

        data, var_meta, glob_meta, img_timestamp = self._read_flat_img()

        if timestamp is not None:
            if img_timestamp is None:
                img_timestamp = timestamp
            assert img_timestamp == timestamp, "Time stamps do not match"

        # when flattened, this drops already all non-active gpis
        data = self._mask_and_reshape(data)

        if self.flatten:
            return Image(self.subgrid.activearrlon,
                         self.subgrid.activearrlat,
                         data,
                         var_meta,
                         timestamp)
        else:
            # also cut 2d case to active area
            min_lat, min_lon = self.subgrid.activearrlat.min(), \
                               self.subgrid.activearrlon.min()
            max_lat, max_lon = self.subgrid.activearrlat.max(), \
                               self.subgrid.activearrlon.max()

            corners = self.grid.gpi2rowcol([
                self.grid.find_nearest_gpi(min_lon, min_lat)[0],  # llc
                self.grid.find_nearest_gpi(max_lon, min_lat)[0],  # lrc
                self.grid.find_nearest_gpi(max_lon, max_lat)[0],  # urc
            ])

            rows = slice(corners[0][0], corners[0][2] + 1)
            cols = slice(corners[1][0], corners[1][1] + 1)

            return Image(self.grid.arrlon.reshape(*self.shape)[rows, cols],
                         np.flipud(self.grid.arrlat.reshape(*self.shape)[rows, cols]),
                         {k: np.flipud(v[rows, cols]) for k, v in data.items()},
                         var_meta,
                         timestamp)

    def write(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass


class C3S_Nc_Img_Stack(MultiTemporalImageBase):
    """
    Class for reading multiple images and iterate over them.
    """

    def __init__(self,
                 data_path,
                 parameters='sm',
                 subgrid=SMECV_Grid_v052(None),
                 flatten=False,
                 solve_ambiguity='sort_last',
                 fntempl=fntempl,
                 subpath_templ=None,
                 fillval=None):
        """
        Parameters
        ----------
        data_path : str
            Path to directory where C3S images are stored
        parameters : list or str,  optional (default: 'sm')
            Variables to read from the image files.
        subgrid : pygeogrids.CellGrid, optional (default: SMECV_Grid_v052(None)
            Subset of the image to read
        flatten : bool, optional (default: False)
            Flatten the read image to a 1D array instead of a 2D array
        solve_ambiguity : str, optional (default: 'latest')
            Method to solve ambiguous time stamps, e.g. if a reprocessing
            was performed.
                - error: raises error in case of ambiguity
                - sort_last (default): uses the last file when sorted by file
                    name, in case that multiple files are found.
                - sort_first: uses the first file when sorted by file name
                    in case that multiple files are found.
        fntempl: str, optional
            Filename template to parse datetime from.
        subpath_templ : list or None, optional (default: None)
            List of subdirectory names to build file paths. e.g. ['%Y'] if files
            in collected by years.
        fillval : float or dict or None, optional (default: None)
            Fill Value for masked pixels, if a dict is passed, this can be
            set for each parameter individually, otherwise it applies to all.
            Note that choosing np.nan can lead to a change in dtype for some
            parameters (int to float).
            None will use the fill value from the netcdf file
        """

        self.data_path = data_path
        ioclass_kwargs = {'parameters': parameters,
                          'subgrid': subgrid,
                          'flatten': flatten,
                          'fillval': fillval}

        self.fname_args = self._parse_filename(fntempl)
        self.solve_ambiguity = solve_ambiguity
        fn_args = self.fname_args.copy()
        fn_args['subvers'] = '*'
        fn_args['cdr'] = '*'
        filename_templ = fntempl.format(**fn_args)

        super(C3S_Nc_Img_Stack, self).__init__(path=data_path,
                                               ioclass=C3SImg,
                                               fname_templ=filename_templ,
                                               datetime_format="%Y%m%d%H%M%S",
                                               subpath_templ=subpath_templ,
                                               exact_templ=False,
                                               ioclass_kws=ioclass_kwargs)

    def _build_filename(self, timestamp: datetime, custom_templ: str = None,
                        str_param: dict = None):
        """
        This function uses _search_files to find the correct
        filename and checks if the search was unambiguous.

        Parameters
        ----------
        timestamp: datetime
            datetime for given filename
        custom_tmpl : string, optional
            If given the fname_templ is not used but the custom_templ. This
            is convenient for some datasets where not all file names follow
            the same convention and where the read_image function can choose
            between templates based on some condition.
        str_param : dict, optional
            If given then this dict will be applied to the fname_templ using
            the fname_templ.format(**str_param) notation before the resulting
            string is put into datetime.strftime.
        """
        filename = self._search_files(timestamp, custom_templ=custom_templ,
                                      str_param=str_param)
        if len(filename) == 0:
            raise IOError("No file found for {:}".format(timestamp.ctime()))
        if len(filename) > 1:
            filename = sorted(filename)
            if self.solve_ambiguity == 'sort_last':
                warnings.warn(f'Ambiguous file for {str(timestamp)} found.'
                              f' Sort and use last: {filename[-1]}, skipped {filename[:-1]}')
                filename = [filename[-1]]
            elif self.solve_ambiguity == 'sort_first':
                warnings.warn(f'Ambiguous file for {str(timestamp)} found.'
                              f' Sort and use first: {filename[0]}')
                filename = [filename[0]]
            else:
                raise IOError(
                    "File search is ambiguous {:}".format(filename))

        return filename[0]

    def _parse_filename(self, template):
        """
        Search a file in the passed directory and use the filename template to
        to read settings.

        Parameters
        -------
        template : str
            Template for all files in the passed directory.

        Returns
        -------
        parse_result : parse.Result
            Parsed content of filename string from filename template.
        """

        for curr, subdirs, files in os.walk(self.data_path):
            for f in files:
                file_args = parse(template, f)
                if file_args is None:
                    continue
                else:
                    file_args = file_args.named
                    file_args['datetime'] = '{datetime}'
                    return file_args

        raise IOError('No file name in passed directory fits to template')

    def tstamps_for_daterange(self, start_date, end_date):
        """
        Return dates in the passed period, with respect to the temp resolution
        of the images in the path.

        Parameters
        ----------
        start_date: datetime
            start of date range
        end_date: datetime
            end of date range

        Returns
        -------
        timestamps : list
            list of datetime objects of each available image between
            start_date and end_date
        """

        if self.fname_args['temp'] == 'MONTHLY':
            next = lambda date: date + relativedelta(months=1)
        elif self.fname_args['temp'] == 'DAILY':
            next = lambda date: date + relativedelta(days=1)
        elif self.fname_args['temp'] == 'DEKADAL':
            next = lambda date: date + relativedelta(days=10)
        else:
            raise NotImplementedError

        timestamps = [start_date]
        while next(timestamps[-1]) <= end_date:
            timestamps.append(next(timestamps[-1]))

        return timestamps

    def read(self, timestamp, **kwargs):
        """
        Return an image for a specific timestamp.

        Parameters
        ----------
        timestamp : datetime.datetime
            Time stamp.

        Returns
        -------
        image : object
            pygeobase.object_base.Image object
        """
        try:
            img = self._assemble_img(timestamp, **kwargs)
            return img
        except IOError:
            warnings.warn(f'Could not load image for {timestamp}.')
            raise IOError


class C3STs(GriddedNcOrthoMultiTs):
    """
    Module for reading C3S time series in netcdf format.
    """

    def __init__(self, ts_path, grid_path=None, remove_nans=False, drop_tz=True,
                 **kwargs):

        """
        Class for reading C3S SM time series after reshuffling.

        Parameters
        ----------
        ts_path : str
            Directory where the netcdf time series files are stored
        grid_path : str, optional (default: None)
            Path to grid file, that is used to organize the location of time
            series to read. If None is passed, grid.nc is searched for in the
            ts_path.
        remove_nans : bool or dict, optional (default: False)
            Replace fill values in SM time series. Either
                - dict of form {parameter: {val_to_replace: replacement_val}, ... }
                - dict of form {parameter : val_to_set_NaN ...}
                - True to replace -9999 with nan anywhere
                - False to do nothing
        drop_tz: bool, optional (default: True)
            Drop time zone information from time series

        Optional keyword arguments that are passed to the Gridded Base:
        ------------------------------------------------------------------------
            parameters : list, optional (default: None)
                Specific variable names to read, if None are selected, all are read.
            offsets : dict, optional (default:None)
                Offsets (values) that are added to the parameters (keys)
            scale_factors : dict, optional (default:None)
                Offset (value) that the parameters (key) is multiplied with
            ioclass_kws: dict
                Optional keyword arguments to pass to OrthoMultiTs class:
                ----------------------------------------------------------------
                    read_bulk : boolean, optional (default:False)
                        if set to True the data of all locations is read into memory,
                        and subsequent calls to read_ts read from the cache and not from disk
                        this makes reading complete files faster#
                    read_dates : boolean, optional (default:False)
                        if false dates will not be read automatically but only on specific
                        request useable for bulk reading because currently the netCDF
                        num2date routine is very slow for big datasets
        """

        if isinstance(remove_nans, dict):
            for var, is_should in remove_nans.copy().items():
                if not isinstance(is_should, dict):
                    remove_nans[var] = {is_should: np.nan}

        self.remove_nans = remove_nans

        if grid_path is None:
            grid_path = os.path.join(ts_path, "grid.nc")

        grid = load_grid(grid_path)

        self.drop_tz = drop_tz
        super(C3STs, self).__init__(ts_path, grid=grid, **kwargs)

    def _read_gp(self, gpi, **kwargs):
        """Read a single point from passed gpi or from passed lon, lat """
        # override the _read_gp function from parent class, to add dropna functionality

        ts = super(C3STs, self)._read_gp(gpi, **kwargs)
        if ts is None:
            return None

        if self.remove_nans:
            if self.remove_nans == True:
                ts = ts.replace(-9999.0000, np.nan)
            else:
                ts = ts.replace(self.remove_nans)

        if not self.drop_tz:
            ts.index = ts.index.tz_localize('UTC')
        else:
            if (hasattr(ts.index, 'tz') and (ts.index.tz is not None)):
                ts.index = ts.index.tz_convert(None)

        return ts

    def read_cell(self, cell, var='sm') -> pd.DataFrame:
        """
        Read all time series for a single variable in the selected cell.

        Parameters
        -------
        cell: int
            Cell number as in the c3s grid
        var : str, optional (default: 'sm')
            Name of the variable to read.
        """

        file_path = os.path.join(self.path, '{}.nc'.format("%04d" % (cell,)))
        with nc.Dataset(file_path) as ncfile:
            loc_id = ncfile.variables['location_id'][:]
            time = ncfile.variables['time'][:]
            unit_time = ncfile.variables['time'].units
            delta = lambda t: timedelta(t)
            vfunc = np.vectorize(delta)
            since = pd.Timestamp(unit_time.split('since ')[1])
            time = since + vfunc(time)

            variable = ncfile.variables[var][:]
            variable = np.transpose(variable)
            data = pd.DataFrame(variable, columns=loc_id, index=time)
            if self.remove_nans:
                if self.remove_nans == True:
                    data = data.replace(-9999, np.nan)
                else:
                    data = data.replace(self.remove_nans)
            return data

    def iter_ts(self, **kwargs):
        pass

    def write_ts(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    root = r"R:\Projects\C3S_312b\08_scratch\v202012_ts2img\060_daily_images\passive"
    old_template = r"C3S-SOILMOISTURE-L3S-SSMV-PASSIVE-{dt}000000-fv202012.nc"
    new_tempate = "C3S-SOILMOISTURE-L3S-SSMV-PASSIVE-DAILY-{dt}000000-TCDR-v202012.0.0.nc"

    for year in os.listdir(root):
        for f in os.listdir(os.path.join(root, year)):
            dt = parse(old_template, f)['dt']
            os.rename(os.path.join(root, year, f),
                      os.path.join(root, year, new_tempate.format(dt=dt)))
