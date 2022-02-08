# -*- coding: utf-8 -*-

import glob
from typing import Union
import numpy as np
from pygeogrids.grids import BasicGrid, CellGrid
from datetime import datetime
from parse import parse
from pygeobase.object_base import Image
import logging
from netCDF4 import Dataset
import pandas as pd
import os

from interface import fntempl

from smecv_grid.grid import SMECV_Grid_v052

try:
    import xarray as xr
    import dask
    from dask.diagnostics import ProgressBar

    xr_supported = True
except ImportError:
    xr_supported = False


class C3S_DataCube:
    """
    Builds a xarray stack / data cube of C3S images
    """

    def __init__(self,
                 data_root,
                 parameters='sm',
                 clip_dates=None,
                 chunks: Union[str, dict, tuple] = 'space',
                 parallel=True,
                 cell_size=5,  # deg
                 cell_chunksize_factor=5,  # * cell_size
                 log_to='std_out',
                 log_level=logging.WARNING,
                 grid=None,
                 **kwargs):
        """
        Parameters
        ----------
        data_root : str or Path
            where the C3S SM files are stored
        parameters : list or str, optional (default: 'sm')
            Parameter names to load from netcdf files
        clip_dates : tuple[str], optional (default: None)
            Start date and end date for files to load
        chunks : Union[str, dict, tuple], optional (default: 'space')
            Either a chunk definition that xarray/dask can use (int, tuple, dict)
            or the name of chunking approach that is already implemented
                - 'time' : Chunks are optimised for image reading
                - 'space' (default) : Chunks of size cell_size*cell_chunksize and
                                    unlimited in time
                - 'unlimited' or None : All dimensions are unlimited
        parallel : bool, optional (default: True)
            Parallel reading of input images.
        cell_size : int, optional (default: 5)
            Cell size in degrees
        cell_chunksize_factor : int, optional (default: 5)
            Chunk size (space) multiplication factor (* cell_size)
            1 means that the chunk size is the same as the cell size
        log_to: str, optional (default: 'std_out')
            (De)activate logging, to std_out, or to logfile.
            - None: to turn logging off completely.
            - 'std_out': to print log messages.
            - path to directory: to create a log file there
        kwargs:
            Additional kwargs are given to xarray open_mfdataset()
        """
        self._setup_logfile(log_to, log_level)

        if isinstance(chunks, str):
            if chunks.lower() == 'space':
                chunks = dict(lon=int((cell_size * cell_chunksize_factor) / 0.25),
                              lat=int((cell_size * cell_chunksize_factor) / 0.25),
                              time=None)  # time series optimised
            elif chunks.lower() == 'time':
                chunks = dict(time=365)
            elif chunks.lower() == 'unlimited':
                chunks = None
            else:
                raise ValueError("Pass 'space' or 'time' or 'unlimited'")

        self.parameters = list(np.atleast_1d(parameters))

        self.root_path = data_root

        if clip_dates is not None:
            start_date = pd.to_datetime(clip_dates[0])
            end_date = pd.to_datetime(clip_dates[1])
        else:
            start_date = end_date = None

        files = self._filter_files(start_date, end_date)  # todo : slow

        drop_vars = []
        with Dataset(files[0]) as ds0:
            for var in ds0.variables.keys():
                if var not in ds0.dimensions.keys() and var not in self.parameters:
                    drop_vars.append(var)

            if grid is not None:
                self.grid = grid
                self.chunks = grid.to_cell_grid(cellsize=cell_size)

            else:
                self.grid = self._gen_grid(ds0['lat'][:].filled(),
                                           ds0['lon'][:].filled(),
                                           cell_size)
                self.chunks = self._gen_grid(np.arange(-90, 90, cell_size)[::-1],
                                             np.arange(-180, 180, cell_size),
                                             cell_size * cell_chunksize_factor)

        with ProgressBar():
            self.ds = xr.open_mfdataset(files,
                                        data_vars='minimal',
                                        parallel=parallel,
                                        engine='netcdf4',
                                        chunks=chunks,
                                        drop_variables=drop_vars,
                                        **kwargs)

        self.active_chunk_id = None
        self.active_chunk_data = None

    @staticmethod
    def _setup_logfile(target='std_out', level=logging.WARNING):
        # set up logger
        logger = logging.getLogger('c3s')
        logger.setLevel(level)
        kwargs = dict(level=level,
                      format='%(levelname)s - %(asctime)s: %(message)s')
        if not target.lower() == 'std_out':
            kwargs['filename'] = os.path.join(target,
                                              f"c3slog_{datetime.now().strftime('%Y%m%d%H%M%S')}.log")

        logging.basicConfig(**kwargs)

    @staticmethod
    def _gen_grid(lats: np.array, lons: np.array, cellsize: float) -> CellGrid:
        lats, lons = np.meshgrid(lats, lons)
        grid = BasicGrid(lons.flatten(),
                         np.flipud(lats.flatten())) \
            .to_cell_grid(cellsize=cellsize)
        return grid

    @staticmethod
    def _cell_to_slice(grid, cell) -> (slice, slice):
        _, lons, lats = grid.grid_points_for_cell(cell)

        slice_lat = slice(max(lats), min(lats), None)
        slice_lon = slice(min(lons), max(lons), None)

        return slice_lat, slice_lon

    def _filter_files(self, start_date: datetime = None, end_date: datetime = None) -> list:
        if start_date is None:
            start_date = datetime(1978, 1, 1)
        if end_date is None:
            end_date = datetime.now()

        files = []
        allfiles = glob.glob(os.path.join(self.root_path, '**', '**.nc'))

        for fname in allfiles:
            fn_comps = parse(fntempl, os.path.basename(fname))
            dt = pd.to_datetime(fn_comps['datetime'])
            if start_date <= dt <= end_date:
                files.append(fname)

        return files

    def _read_gp(self,
                 gpi: int) -> pd.DataFrame:

        active_chunk_id = self.chunks.gpi2cell(gpi)
        # import pdb; pdb.set_trace()
        if active_chunk_id != self.active_chunk_id:
            slice_lat, slice_lon = self._cell_to_slice(self.chunks, active_chunk_id)
            logging.info('Extracting chunk')
            self.active_chunk_data = self.ds.sel({'lat': slice_lat, 'lon': slice_lon}).load()
            self.active_chunk_id = active_chunk_id

        lon, lat = self.grid.gpi2lonlat(gpi)
        logging.info('Reading TS from chunk')
        ts_data = self.active_chunk_data.sel({'lat': lat, 'lon': lon})
        ts_data = ts_data.drop_vars(('lat', 'lon')).to_dataframe()
        return ts_data

    def read_img(self,
                 time: Union[str, datetime]) -> Image:
        time = pd.to_datetime(time)
        data = self.ds.sel({'time': time})

        vars = [d for d in list(data.variables.keys()) if d not in list(data.coords.keys())]

        return Image(lon=data['lon'].values,
                     lat=data['lat'].values,
                     data={v: data[v].values for v in vars},
                     metadata={v: data[v].attrs for v in vars},
                     timestamp=time,
                     timekey='time')

    def read_ts(self,
                *args,
                max_dist=np.inf):

        if len(args) == 1:
            data = self._read_gp(args[0])
        elif len(args) == 2:
            gpi, dist = self.grid.find_nearest_gpi(args[0], args[1], max_dist=max_dist)
            if hasattr(gpi, '__len__') and (len(gpi) == 0):
                data = None
            else:
                data = self._read_gp(gpi)
        else:
            raise ValueError("Wrong number of arguments passed, either pass 1 gpi"
                             " or two coordinates (lon, lat)")

        return data

    def write_stack(self, out_file, **kwargs):
        vars = [v for v in list(self.ds.variables.keys()) if v not in self.ds.coords.keys()]

        encoding = {v: {'zlib': True, 'complevel': 6} for v in vars}

        with ProgressBar():
            self.ds.to_netcdf(out_file, encoding=encoding, **kwargs)


if __name__ == '__main__':
    ds = C3S_DataCube("/shares/pstradio/radar/Datapool/C3S/01_raw/v202012/ICDR/060_dailyImages/passive/",
                      chunks='unlimited', clip_dates=('2020-01-01', '2021-12-31'),
                      log_to='std_out', log_level=logging.INFO)

    # ds.write_stack(r"C:\Temp\c3s\stacks\combined_2000.nc")
    # ts = ds.read_ts(45,15)
    # ts1 = ds.read_ts(792743)
    # ts2 = ds.read_ts(792744)
    #
    # ts3 = ds.read_ts(356982)
    #
    # for t in ['2000-05-01', '2010-05-02', '2019-05-03']:
    #     img = ds.read_img(t)
    #     print(img)
