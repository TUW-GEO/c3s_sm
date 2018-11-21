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
Readers for the C3S soil moisture proudct daily, dekadal (10-daily) and monthly images as well
as for timeseries generated using this module
'''

import pandas as pd
import os
import netCDF4 as nc
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from pygeobase.object_base import Image
from pygeobase.io_base import ImageBase
from pygeobase.io_base import MultiTemporalImageBase
from pygeogrids.netcdf import load_grid

from netCDF4 import Dataset
from pynetcf.time_series import GriddedNcOrthoMultiTs
from datetime import datetime
from parse import parse
from c3s_sm.grid import C3SCellGrid, C3SLandGrid


def c3s_filename_template():
    # this function can be used in case the filename changes at some point.
    return '{product}-SOILMOISTURE-L3S-{data_type}-{sensor_type}-{temp_res}-' \
           '{datetime}000000-{sub_prod}-{version}.{sub_version}.nc'


class C3STs(GriddedNcOrthoMultiTs):
    """
    Module for reading C3S time series in netcdf format.
    """

    def __init__(self, ts_path, grid_path=None, remove_nans=False):
        '''
        Parameters
        ----------
        ts_path : str
            Path to the netcdf time series files
        grid_path : str, optional (default: None)
            Path to the netcdf grid file.
            If None is passed, grid.nc is searched in ts_path.
        remove_nans : bool, optional (default: False)
            Replace -9999 with np.nan in time series
        '''
        self.remove_nans = remove_nans

        if grid_path is None:
            grid_path = os.path.join(ts_path, "grid.nc")

        grid = load_grid(grid_path)

        super(C3STs, self).__init__(ts_path, grid=grid)


    def _read_gp(self, gpi, **kwargs):
        """Read a single point from passed gpi or from passed lon, lat """
        # override the _read_gp function from parent class, to add dropna functionality

        ts = super(C3STs, self)._read_gp(gpi, **kwargs)
        if ts is None:
            return None

        if self.remove_nans:
            ts = ts.replace(-9999.0000, np.nan)

        ts.index = ts.index.tz_localize('UTC')

        return ts


    def read_cell(self, cell, var='sm'):
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
                data = data.replace(-9999.0000, np.nan)
            return data


class C3SImg(ImageBase):
    """
    Class to read a single C3S image (for one time stamp)
    """
    def __init__(self, filename, parameters='sm', mode='r', subgrid=None,
                 array_1D=False):
        # todo: get rid of mode?
        '''
        Parameters
        ----------
        filename : str
            Path to the file to read
        parameters : str or Iterable, optional (default: 'sm')
            Names of parameters in the file to read.
            If None are passed, all are read.
        mode : str, optional (default: 'r')
            Netcdf file mode, choosing something different to r may delete data.
        array_1D : bool, optional (default: False)
            Read image as one dimensional array, instead of a 2D array
            Use this when using a subgrid.
        '''

        super(C3SImg, self).__init__(filename, mode=mode)

        if not isinstance(parameters, list):
            parameters = [parameters]

        self.parameters = parameters
        self.grid = C3SCellGrid() if not subgrid else subgrid
        self.array_1D = array_1D

    def read(self, timestamp=None):
        """
        Reads a single C3S image.

        Parameters
        -------
        timestamp: datetime, optional (default: None)
            Timestamp for file to read. Pass None if file contains only 1 timestamp

        Returns
        -------
        image : Image
            Image object from netcdf content
        """

        ds = Dataset(self.filename, mode='r')

        param_img = {}
        img_meta = {'global': {}}

        if self.parameters[0] is None:
            parameters = ds.variables.keys()
        else:
            parameters = self.parameters

        for param in parameters:
            if param in ['lat', 'lon', 'time']: continue
            param_metadata = {}

            variable = ds.variables[param]

            for attr in variable.ncattrs():
                param_metadata.update({str(attr): getattr(variable, attr)})

            param_data = np.flipud(variable[0][:].filled()).flatten()

            param_img[str(param)] = param_data[self.grid.activegpis]
            img_meta[param] = param_metadata

        # add global attributes
        for attr in ds.ncattrs():
            img_meta['global'][attr] = ds.getncattr(attr)

        ds.close()

        if self.array_1D:
            return Image(self.grid.activearrlon, self.grid.activearrlat,
                         param_img, img_meta, timestamp)
        else:
            yres, xres = self.grid.shape
            for key in param_img:
                param_img[key] = param_img[key].reshape(xres, yres)

            return Image(self.grid.activearrlon.reshape(xres, yres),
                         self.grid.activearrlat.reshape(xres, yres),
                         param_img,
                         img_meta,
                         timestamp)


    def write(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass



class C3S_Nc_Img_Stack(MultiTemporalImageBase):
    '''Class for reading multiple images and iterate over them. '''
    def __init__(self, data_path, parameters='sm', subgrid=None, array_1D=False):
        '''
        Parameters
        ----------
        data_path : str
            Path to directory where C3S images are stored
        parameters : list or str,  optional (default: 'sm')
            Variables to read from the image files.
        subgrid : grid, optional (default: None)
            Subset of the image to read
        array_1D : bool, optional (default: False)
            Flatten the read image to a 1D array instead of a 2D array
        '''

        self.data_path = data_path
        ioclass_kwargs = {'parameters': parameters,
                          'subgrid' : subgrid,
                          'array_1D': array_1D}

        template = c3s_filename_template()
        self.fname_args = self._parse_filename(template)
        filename_templ = template.format(**self.fname_args)

        # todo: is this fixed? Daily data is organised differently than M and 10D?
        if self.fname_args['temp_res'] == 'DAILY':
            subpath_templ = ['%Y']
        else:
            subpath_templ = None

        super(C3S_Nc_Img_Stack, self).__init__(path=data_path, ioclass=C3SImg,
                                               fname_templ=filename_templ ,
                                               datetime_format="%Y%m%d",
                                               subpath_templ=subpath_templ,
                                               exact_templ=True,
                                               ioclass_kws=ioclass_kwargs)

    def _parse_filename(self, template):
        '''
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
        '''

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
        '''
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
        '''

        if self.fname_args['temp_res'] == 'MONTHLY':
            next = lambda date : date + relativedelta(months=1)
        elif self.fname_args['temp_res'] == 'DAILY':
            next = lambda date : date + relativedelta(days=1)
        elif self.fname_args['temp_res'] == 'DEKADAL':
            next = lambda date : date + relativedelta(days=10)
        else:
            raise NotImplementedError

        timestamps = [start_date]
        while next(timestamps[-1]) <= end_date:
            timestamps.append(next(timestamps[-1]))

        return timestamps





