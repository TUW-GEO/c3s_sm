import os
import warnings
from glob import glob
import pandas as pd
from typing import Union, List
import yaml
from parse import parse
from c3s_sm.const import fntempl as _default_template
import xarray as xr
from repurpose.process import parallel_process


def collect_ts_cov(data_path: str, n_proc=1, progressbar=False):
    """
    Open all time series files in a directory (slow) and detect the
    temporal coverage.

    Parameters
    ----------
    data_path : str
        Path where the cell files are stored.
    progressbar: bool, optional (default: False)
        Show progress bar when looping through files.

    Returns
    -------
    periods: dict
        Periods coverged by the time series files.
        {(start, end): [cell, cell, ...], ...}
    """
    fl = glob(
        os.path.join(data_path, '**', "[0-9][0-9][0-9][0-9].nc"),
        recursive=True)

    if len(fl) == 0:
        raise ValueError(f"No matching files found in {data_path}")

    def _func(f: str) -> tuple:
        cell = int(os.path.basename(f).split('.')[0])
        ds = xr.open_dataset(f)
        start = pd.to_datetime(min(ds['time'].values)).to_pydatetime()
        end = pd.to_datetime(max(ds['time'].values)).to_pydatetime()
        parameters = list(ds.data_vars)

        return start, end, cell, parameters

    se = parallel_process(
        _func,
        ITER_KWARGS=dict(f=fl),
        show_progress_bars=progressbar,
        backend='threading',
        n_proc=n_proc)

    periods = {}
    parameters = None

    for start, end, cell, param in se:
        if (start, end) not in periods.keys():
            periods[(start, end)] = []
        periods[(start, end)].append(cell)
        if parameters is None:
            parameters = sorted(param)
        else:
            if sorted(param) != parameters:
                warnings.warn(f"Found different parameters for cell {cell}")

    return periods, parameters

def _parse(files, fntempl: List) -> (dict, str, str):
    if len(fntempl) == 0:
        raise ValueError("No templates given")

    for f in files:
        file_args = None
        for t in fntempl:
            file_args = parse(t, os.path.basename(f))
            if file_args is not None:
                break
        if file_args is None:
            continue
        return file_args.named, t, f

    return None, None, None


def img_infer_file_props(path: str,
                         fntempl: Union[str,List[str]] = _default_template,
                         start_from='last') -> dict:
    """
    Parse file names to retrieve properties from :func:`c3s_sm.const.fntempl`.
    Does not open any files.
    """
    if isinstance(fntempl, str):
        fntempl = [fntempl]

    files = glob(os.path.join(path, '**', '*.nc'), recursive=True)
    files.sort()
    if len(files) == 0:
        raise ValueError(f"No matching files for chosen template found in "
                         f"the directory {path}")
    else:
        if start_from.lower() == 'last':
            files = files[::-1]
        elif start_from.lower() == 'first':
            pass
        else:
            raise NotImplementedError(f"`start_from` must be one of: "
                                      f"`first`, `last`.")

        file_args, _, _ = _parse(files, fntempl)

        if file_args is not None:
            return file_args

    raise ValueError(f"No matching files for chosen template found in the "
                     f"directory {path}")


def read_summary_yml(path: str) -> dict:
    """
    Read image summary and return fields as dict.
    """
    path = os.path.join(path, 'overview.yml')

    with open(path, 'r') as stream:
        props = yaml.safe_load(stream)

    return props


def get_first_image_date(path: str, fntempl: str = _default_template) -> str:
    """
    Parse files in the given directory (or any subdir) using the passed
    filename template. props will contain all fields specified in the template.
    the `datetime` field is required and used to determine the first image date.

    Parameters
    ----------
    path: str
        Path to the directory containing the image files
    fntempl: str, optional
        The filename template used to parse image file names.
        Must contain a field `datetime` that indicates the location of the
        image time stamp in the filename.

    Returns
    -------
    date: str
        Parse date from the first found image file that matches `fntempl`.
    """
    try:
        props = img_infer_file_props(path, fntempl=fntempl, start_from='first')
        startdate = props['datetime']
    except ValueError:
        raise ValueError('Could not infer start date from image files. '
                         'Please specify startdate manually.')
    return startdate


def get_last_image_date(path: str, fntempl: str) -> str:
    """
    Parse files in the given directory (or any subdir) using the passed
    filename template. props will contain all fields specified in the template.
    the `datetime` field is required and used to determine the last image date.

    Parameters
    ----------
    path: str
        Path to the directory containing the image files
    fntempl: str (optional)
        The filename template used to parse image file names.
        Must contain a field `datetime` that indicates the location of the
        image time stamp in the filename.

    Returns
    -------
    date: str
        Parse date from the last found image file that matches `fntempl`.
    """
    try:
        props = img_infer_file_props(path, fntempl=fntempl, start_from='last')
        enddate = props['datetime']
    except ValueError:
        raise ValueError('Could not infer end date from image files. '
                         'Please specify enddate manually.')
    return enddate


def update_image_summary_file(data_path: str,
                              out_file=None,
                              fntempl: str = _default_template):
    """
    Summarize image metadata as yml file

    Parameters
    ----------
    data_path: str
        Root path to the image archive
    out_file: str, optional (default: None)
        Path to summary file. File will be created/updated.
        If not specified, then `data_path` is used. If a file already exists,
        it will be overwritten.
    fntempl: str, optional
        The filename template used to parse image file names.
        Must contain a field `datetime` that indicates the location of the
        image time stamp in the filename.
    """
    first_image_date = get_first_image_date(data_path, fntempl=fntempl)
    last_image_date = get_last_image_date(data_path, fntempl=fntempl)

    props = img_infer_file_props(data_path, start_from='first')
    _ = props.pop("datetime")
    props['period_from'] = str(pd.to_datetime(first_image_date).date())
    props['period_to'] = str(pd.to_datetime(last_image_date).date())

    if out_file is None:
        out_file = os.path.join(data_path, f"overview.yml")

    with open(out_file, 'w') as f:
        yaml.dump(props, f, default_flow_style=False)


def update_ts_summary_file(data_path, props=None, collect_cov=False, **kwargs):
    """
    Create a summary yml file that contains the most relevant information
    for a dataset. This information is then also used to update/extent the
    record.

    Parameters
    ----------
    data_path: str
        Path where the time series files are stored and where the file
        overview.yml will be stored.
    props: dict, optional
        Additional information to write down
    collect_cov: bool, optional (False)
        Loop through all available cell files and detect their tempora coverage.
        Will add one or multiple `periodX` fields to the summary file, which
        indicate the temporal coverage of each cell.
    kwargs:
        Additional keyword arguments are passed to `collect_ts_cov`
    """

    if collect_cov:
        periods = collect_ts_cov(data_path, **kwargs)
    else:
        periods = None

    try:
        old_props = read_summary_yml(data_path)
    except FileNotFoundError:
        old_props = {}

    props = {} if props is None else props
    if periods is not None:
        i = 1
        for startend, cells in periods.items():
            props[f'period{i}'] = dict(
                start=str(startend[0]),
                end=str(startend[1]),
                N=len(cells),
                cells=sorted(cells))

    out_file = os.path.join(data_path, f"overview.yml")

    # keep old props (except datetime)
    for k, v in old_props.items():
        if k not in props and not k.startswith('period'):
            props[k] = v

    for k, v in props.items():
        if isinstance(v, tuple):
            props[k] = list(v)

    with open(out_file, 'w') as f:
        yaml.dump(props, f, default_flow_style=False, sort_keys=False)

    return props


def delete_empty_directories(path: str):
    """
    Delete empty dirs in path
    """
    for root, dirs, files in os.walk(path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Check if the directory is empty
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
