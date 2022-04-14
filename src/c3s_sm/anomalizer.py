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

"""
Anomalies calculator
"""
import os.path

import pandas as pd
import numpy as np
import xarray as xr

from smecv_grid.grid import SMECV_Grid_v052
from xrcube import C3S_DataCube
from interface import C3STs

from joblib import Parallel, delayed

from pytesmo.time_series.anomaly import calc_climatology, calc_anomaly
from datetime import datetime
from pynetcf.time_series import GriddedNcContiguousRaggedTs


class C3SreaderICDR_TCDR(GriddedNcContiguousRaggedTs):
    """

    Allows reading of anomalies from Interim CDR with a pre-set Thematic CDR baseline

    Parameters
    ----------
    icdr_path: str or Path
        directory with the ICDR data
    tcdr_path: str or Path
        directory with the TCDR data
    icdr_start: str
        start of the considered period. Format: 'YYYY-mm-dd'
        **Note**: defines size of the netCDF stack
    icdr_end: str
        end of the considered period. Format: 'YYYY-mm-dd'
        **Note**: defines size of the netCDF stack

    Attributes
    ----------

    """

    def __init__(
            self,
            icdr_path: str,
            tcdr_path: str,
            icdr_start: str = None,
            icdr_end: str = None,
            sm_name: str = "sm",
            grid=SMECV_Grid_v052(),
            stack_kwargs: dict = {},
            ts_kwargs: dict = {},
            mode='r',
            out_path=None,
    ):
        if out_path is None:
            out_path = os.path.join(icdr_path, f"default_out")

        self.reader_path = out_path

        super(C3SreaderICDR_TCDR, self).__init__(
            path=out_path,
            mode=mode,
            grid=grid,
        )

        self.grid = grid
        self.sm_name = sm_name

        # initialise the netcdf stack reader
        self.StackReader = C3S_DataCube(
            icdr_path,
            chunks='unlimited',
            clip_dates=(icdr_start, icdr_end),
            log_to='std_out',
            parameters=[sm_name],
            grid=grid,
            **stack_kwargs,
        )
        # initialise the timeseries reader
        self.TSReader = C3STs(
            tcdr_path,
            remove_nans=True,
            **ts_kwargs,
        )

    def _apply_funct(self, gpi, funct, name=None, **fun_kwargs) -> pd.DataFrame:
        """Apply a given function to icdr and tcdr data, returning a series"""
        if funct is None:
            raise ValueError(
                "Should provide value to 'funct'"
            )

        # extract time series data
        icdr_ts = self.StackReader.read_ts(gpi)[self.sm_name]
        tcdr_ts = self.TSReader.read(gpi)[self.sm_name]

        # apply the function which uses the icdr and tcdr
        out = funct(icdr_ts, tcdr_ts, **fun_kwargs)

        assert type(out) == pd.Series, "The given function should return a Series"

        if name is None:
            name = self.sm_name
        out_df = out.to_frame(name=name)

        return out_df

    def write_funct(self, funct, grid_subset=None, **fun_kwargs):
        """Write the output of a function to time series"""
        if grid_subset is None:
            grid_subset = self.grid

        print(
            f"write_funct has got the cell(s): {grid_subset.get_cells()} to process"
        )

        for gp in np.unique(grid_subset.activegpis):
            data = self._apply_funct(gp, funct, **fun_kwargs)
            self._write_gp(gp, data)


class AnomalyRepurpose(GriddedNcContiguousRaggedTs):
    """
    Class to repurpose the anomaly (or any data) files generated with the C3SreaderICDR_TCDR class
    """

    def __init__(self, *args, **kwargs):
        self._fillvalue_in = -777777.
        self._fillvalue_merge = -888888.

        super(AnomalyRepurpose, self).__init__(*args, **kwargs)

    def read_cell(
            self,
            cell,
            dt_index=None,
            dates=None,
            param='sm',
            freq="D",
            **kwargs
    ) -> xr.Dataset:
        """
        Reads a single variable for all points of a (global) cell.

        Parameters
        ----------
        cell: int
            Cell number, will look for a file <cell>.nc that must exist.
            The file must contain a variable `location_id` and `time`.
            Time must have an attribute of form '<unit> since <refdate>'
        dt_index: pd.DateTimeIndex, optional
            Index to apply to the output. Either this or 'dates' should be passed
        dates: list, optional
            start - end of the period to read. Format: [(Year, month, day), (Year, month, day)]
            Either this or 'dt_index' should be passed
        param: str, optional (default: 'sm')
            Variable to extract from files
        freq: str
            datetime index frequency

        Returns
        -------
        cellds: xr.Dataset
            Filled dataset values for a cell
        """
        if dt_index is None:
            if dates is None:
                raise ValueError(
                    "Either one of 'dt_index' or 'dates' should be given"
                )

            startdate, enddate = dates

            dt_index = pd.date_range(
                datetime(*startdate),
                datetime(*enddate),
                freq=freq
            )

        cell_data = []
        # get gpis from global grid and not self.grid
        gps, lons, lats = SMECV_Grid_v052(None).grid_points_for_cell(cell)
        for gp in gps:
            try:
                gp_data = self.read(gp)
                gp_data = gp_data.reindex(dt_index, axis=0)
                gp_data.rename(columns={param: gp}, inplace=True)

            # no values for this gpi
            # TODO: check why Indexes do not match
            except (
                    OSError,
                    AttributeError,
                    IndexError
            ):
                gp_data = pd.DataFrame(self._fillvalue_merge, index=dt_index, columns=[gp])

            cell_data.append(gp_data)

        data = pd.concat(cell_data, axis=1)
        data.fillna(self._fillvalue_in, inplace=True)

        # make xr.Dataset with grid size
        size1d = int(SMECV_Grid_v052().cellsize / SMECV_Grid_v052().resolution)

        gpis = data.columns.values.reshape(size1d, size1d)
        data_vars = {param: data.values.reshape((len(dt_index), size1d, size1d))}
        # define dataset variables
        data_vars = {k: (['time', 'lat', 'lon'], v) for k, v in data_vars.items()}
        data_vars['gpi'] = (['lat', 'lon'], gpis)

        cellds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': dt_index,
                'lon': np.array(np.unique(lons), np.float32),
                'lat': np.array(np.unique(lats), np.float32)
            }
        )

        return cellds

    def repurpose(self, path_out, **kwargs):
        """Convert time series dataset to images"""
        ds = []
        for cell in np.unique(self.grid.activearrcell):
            cellds = self.read_cell(cell, **kwargs)
            ds.append(cellds)

        print("Combining cell datasets in a global cube")
        ds = xr.combine_by_coords(ds)

        # TODO: encoding
        ds.to_netcdf(path_out)


# Functions to apply to ICDR and TCDR data
# =========================================

def calc_anomalies(
        icdr_ts: pd.Series,
        tcdr_ts: pd.Series,
        baseline_start=(1991, 1, 1),
        baseline_end=(2021, 1, 1),
) -> pd.Series:
    """Get anomalies from icdr and tcdr time series given a baseline"""
    clim = calc_climatology(
        tcdr_ts,
        timespan=[datetime(*baseline_start), datetime(*baseline_end)]
    )

    return calc_anomaly(icdr_ts, climatology=clim)


def get_percentile(
        icdr_ts,
        tcdr_ts,
        baseline_start=(1991, 1, 1),
        baseline_end=(2021, 1, 1),
) -> pd.Series:
    """
    Get the soil moisture percentile of the given icdr months based on a defined tcdr
    baseline
    """
    month_icdr = icdr_ts.groupby(pd.Grouper(freq='M')).mean()

    baseline_dates = (tcdr_ts.index < datetime(*baseline_end)) & (tcdr_ts.index > datetime(*baseline_start))
    tcdr_ts = tcdr_ts.loc[baseline_dates]
    tcdr_ts = tcdr_ts.groupby(pd.Grouper(freq='M')).mean()

    def month2percentile(value, baseline):
        # make apply function
        month = value.name.month
        month_baseline = baseline[baseline.index.month == month]

        # percentile
        return (month_baseline.values < value[0]).mean() * 100

    month_icdr = month_icdr.to_frame("icdr").apply(
        lambda x: month2percentile(x, tcdr_ts), axis=1
    )

    return month_icdr


# Function to parallelize the class methods
# =========================================

def parallel_writing(AnomObj, funct, n_cores=8, **fun_kwargs):
    """Parallelize the writing of anomalies over the specified cores number"""
    runs = [
        AnomObj.grid.subgrid_from_cells(cell) for cell in AnomObj.grid.get_cells()
    ]
    Parallel(n_jobs=n_cores)(
        delayed(AnomObj.write_funct)(funct, run, **fun_kwargs) for run in runs
    )


if __name__ == '__main__':

    # EXAMPLE USE
    # ===========
    subgrid = SMECV_Grid_v052()  # .subgrid_from_bbox(minlon, minlat, maxlon, maxlat)

    # generation of anomalies time series
    # -----------------------------------
    Anom = C3SreaderICDR_TCDR(
        "path/to/the/ICDR",
        "path/to/the/TCDR",
        icdr_start="yyyy-MM-DD",
        icdr_end="yyyy-MM-DD",
        grid=subgrid,
        mode='w',
        out_path="path/to/output"
    )

    # give function to the parallel withing, in this case 'calc_anomalies'
    parallel_writing(Anom, calc_anomalies, n_cores=8)

    # generation of stack from the timeseries of anomalies (or other)
    # ---------------------------------------------------------------
    Rep = AnomalyRepurpose(
        path="path/to/timeseries",
        grid=subgrid,
    )
    # Index the months are grouped to
    Rep.repurpose("path/to/output/nc")
