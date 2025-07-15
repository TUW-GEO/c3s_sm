import os
from datetime import datetime
import pandas as pd
import click
from c3s_sm.download import download_and_extract, first_missing_date
from c3s_sm.misc import get_first_image_date, get_last_image_date, img_infer_file_props
from c3s_sm.reshuffle import img2ts, extend_ts
from c3s_sm.const import fntempl as _default_template, check_api_read, cds_api_url


@click.command(
    "download",
    context_settings={'show_default': True},
    short_help="Download C3S SM data from Climate Data Store.")
@click.argument("path", type=click.Path(writable=True))
@click.option(
    '--startdate',
    '-s',
    type=click.STRING,
    default='1978-11-01',
    help="Startdate in format YYYY-MM-DD. If not given, "
    "then the first available date of the product is used.")
@click.option(
    '--enddate',
    '-e',
    type=click.STRING,
    default=str(datetime.now().date()),
    help="Enddate in format YYYY-MM-DD. If not given, "
    "then the current date is used.")
@click.option(
    "-p",
    "--product",
    type=click.STRING,
    default='combined',
    help="The C3S SM sensor product to download. Choose one of "
    "combined, active, passive.")
@click.option(
    "-f",
    "--freq",
    type=click.STRING,
    default="daily",
    help="The C3S SM sensor product temporal sampling frequency to download. "
    "Choose one of: daily, dekadal, monthly.")
@click.option(
    "-v",
    '--version',
    type=click.STRING,
    default="v202312",
    help="The C3S SM product version to download. "
    "Choose one that is on the CDS: "
    "e.g. deprecated_v20191, v201706, v201812, "
    "v201912_1, v202012, v202212, v202312")
@click.option(
    "-k",
    "--keep",
    type=click.BOOL,
    default=False,
    help="Also keep the original, temporarily downloaded image stack "
    "instead of deleting it after extracting individual images.")
@click.option(
    "--cds_token",
    type=click.STRING,
    default=None,
    help="To identify with the CDS. Required if no .cdsapirc file exists in "
         "the home directory (see documentation). You can find your token/key "
         "on your CDS user profile page.")
def cli_download(path,
                 startdate,
                 enddate,
                 product,
                 freq,
                 version,
                 keep,
                 cds_token=None):
    """
    Download C3S SM data within a chosen period. NOTE: Before using this
    program, create a CDS account and set up a `.cdsapirc` file as described
    here: https://cds.climate.copernicus.eu/how-to-api

    \b
    Required Parameters
    -------------------
    PATH: string (required)
        Path where the downloaded C3S SM images are stored.
        Make sure to set up the CDS API for your account as describe in
        https://cds.climate.copernicus.eu/how-to-api
    """
    # The docstring above is slightly different to the normal python one to
    # display it properly on the command line.

    if cds_token is not None:
        os.environ["CDSAPI_KEY"] = cds_token
    check_api_read()

    startdate = pd.to_datetime(startdate)
    enddate = pd.to_datetime(enddate)

    print(f"Downloading C3S SM CDR/ICDR {freq} {product} {version} "
          f"from {startdate.isoformat()} to {enddate.isoformat()} "
          f"into {path}.")

    download_and_extract(
        path,
        startdate=startdate,
        enddate=enddate,
        product=product,
        freq=freq,
        version=version,
        keep_original=keep)


@click.command(
    "update_img",
    context_settings={'show_default': True},
    short_help="Extend an existing record by downloading new files.")
@click.argument("path", type=click.Path(writable=True))
@click.option(
    "--fntempl",
    type=click.STRING,
    default=None,
    help="In case files don't follow the usual naming convention, "
    "a custom template can be given here. Must contain fields "
    "`freq`, `prod`, `vers` and `datetime`")
@click.option(
    "--cds_token",
    type=click.STRING,
    default=None,
    help="To identify with the CDS. Required if no .cdsapirc file exists in "
         "the home directory (see documentation). You can find your token/key "
         "on your CDS user profile page.")
@click.option("--dry-run", is_flag=False, default=False,
              help="Do not actually download the images. Returns the "
                   "chosen options instead. Can be grabbed with eg."
                   "`read v1 v2 ... <<< $(c3s_sm update_img ...)`")
def cli_update_img(path, fntempl, cds_token=None, dry_run=False):
    """
    Extend a locally existing C3S SM record by downloading new files that
    don't yet exist locally.
    This will find the latest available local file, and download all
    available extensions.
    NOTE: Use the `c3s_sm download` program first do create a local record
    to update with this function.

    \b
    Required Parameters
    -------------------
    PATH: string
        Path where previously downloaded C3S SM images are stored.
        Make sure to set up the CDS API for your account as describe in
        https://cds.climate.copernicus.eu/how-to-api
    """
    # The docstring above is slightly different to the normal python one to
    # display it properly on the command line.

    if cds_token is not None:
        os.environ["CDSAPI_KEY"] = cds_token
    check_api_read()

    if fntempl is None:
        fntempl = _default_template

    props = img_infer_file_props(path, fntempl=fntempl, start_from='last')

    freq = props['freq'].lower()
    product = props['product'].lower()
    version = props['version']

    startdate = first_missing_date(props['datetime'], freq=freq)

    if not dry_run:
        print(
            f"Update C3S SM images: "
            f"Fetching latest data for C3S SM CDR/ICDR {freq} {product} {version} "
            f"after {startdate.isoformat()} into {path}.")

    _ = download_and_extract(
        path,
        startdate=startdate,
        freq=freq,
        version=version,
        product=product,
        keep_original=False,
        dry_run=dry_run)

    if dry_run:
        # You can grab those with eg ``
        print(path, str(startdate).replace(' ', 'T'), freq,
              version, product)

@click.command(
    "reshuffle",
    context_settings={'show_default': True},
    short_help="Convert C3S SM images into time series.")
@click.argument("img_path", type=click.Path(readable=True))
@click.argument("ts_path", type=click.Path(writable=True))
@click.option(
    '--startdate',
    '-s',
    type=click.STRING,
    default=None,
    help="Format YYYY-MM-DD | First image time stamp to include in the"
    "time series. [default: Date of the first available image]")
@click.option(
    '--enddate',
    '-e',
    type=click.STRING,
    default=None,
    help="Format YYYY-MM-DD | Last image time stamp to include in the"
    "time series. [default: Date of the last available image]")
@click.option(
    '--parameters',
    '-p',
    multiple=True,
    type=click.STRING,
    default=None,
    help="STRING | Data variable in images to include "
    "in time series. If not specified, then all variables are "
    "included. You can pass this option multiple times, "
    "e.g. `... -p sm -p flag ...`! "
    "[default: ALL parameters are included]")
@click.option(
    '--land',
    type=click.BOOL,
    default=True,
    help="True or False | Activating this flag will exclude grid "
    "cells over water are not converted to time series. "
    "Leads to faster processing and smaller files, but a varying "
    "number of time series in each file.")
@click.option(
    '--bbox',
    nargs=4,
    type=click.FLOAT,
    help="4 NUMBERS | min_lon min_lat max_lon max_lat. "
    "Set Bounding Box (lower left and upper right corner) "
    "of area to reshuffle (WGS84). [default: -180 -90 180 90]")
@click.option(
    '--ignore_meta',
    type=click.BOOL,
    default=False,
    help="True or False | Activate to NOT transfer netcdf attributes"
    " from images into time series files. E.g. for unsupported "
    "data versions.")
@click.option(
    "--fntempl",
    type=click.STRING,
    default=None,
    help="STRING CONTAINING {PLACEHOLDERS} | If image files don't "
    "follow the usual naming convention, a custom template can "
    "be given here. Must contain {placeholder} fields for "
    "{freq}, {product}, {version} and {datetime}.")
@click.option(
    "--overwrite",
    type=click.BOOL,
    default=False,
    help="True or False | If this is True, then any files that already"
    " exist in the `output_path`, will be replaced by the newly "
    "created ones. If this is False, we try to append data to "
    "existing files.")
@click.option(
    '--imgbuffer',
    '-b',
    type=click.INT,
    default=250,
    help="NUMBER | Number of images to read into memory at once before "
    "conversion to time series. A larger buffer means faster"
    " processing but requires more memory.")
@click.option(
    '--n_proc',
    '-n',
    type=click.INT,
    default=1,
    help="NUMBER | Number of parallel processes for reading and "
    "writing data.")
def cli_reshuffle(img_path, ts_path, startdate, enddate, parameters, land,
                  bbox, ignore_meta, fntempl, overwrite, imgbuffer, n_proc):
    """
    Convert C3S SM image data into a (5x5 degrees chunked) time series format
    following CF conventions for 'Orthogonal multidimensional array representation'
    This format is preferred for performant location-based reading of SM data
    over the full period. To read the generated time series data, you can then
    use the `c3s_sm.interface.C3STs` or `pynetcf.time_series.GriddedNcOrthoMultiTs`
    class.

    \b
    Required Parameters
    -------------------
    IMG_PATH: string
        Path where previously downloaded C3S SM images are stored. Use the
        `c3s_sm download` command to retrieve image data.
    TS_PATH: string
        Path where the newly created time series files should be stored.
    """
    # The docstring above is slightly different to the normal python one to
    # display it properly on the command line.
    if len(parameters) == 0:
        parameters = None

    if fntempl is None:
        fntempl = _default_template

    if startdate is None:
        startdate = get_first_image_date(img_path, fntempl)
    if enddate is None:
        enddate = get_last_image_date(img_path, fntempl)

    startdate = pd.to_datetime(startdate)
    enddate = pd.to_datetime(enddate)

    print(f"Creating time series for image data from {img_path}.")
    print(f"Parameters: {parameters} (`None` means all)")
    print(f"From: {startdate.isoformat()}, To: {enddate.isoformat()}")
    print(f"Into target directory: {ts_path}")

    img2ts(
        img_path,
        ts_path,
        startdate=startdate,
        enddate=enddate,
        parameters=parameters,
        land_points=land,
        bbox=bbox,
        ignore_meta=ignore_meta,
        fntempl=fntempl,
        overwrite=overwrite,
        imgbuffer=imgbuffer,
        n_proc=n_proc)


@click.command(
    "update_ts",
    context_settings={'show_default': True},
    short_help="Extend an existing time series record with "
    "available image data.")
@click.argument("img_path", type=click.Path(readable=True))
@click.argument("ts_path", type=click.Path(writable=True))
@click.option(
    "--freq",
    type=click.STRING,
    default=None,
    help="This option can be used to manually specify whether DAILY,"
    "MONTHLY or DEKADAL data are processed.")
@click.option(
    "--fntempl",
    type=click.STRING,
    default=None,
    help="In case image files don't follow the usual naming "
    "convention, a custom template can be given here. Must "
    "contain fields `freq`, `prod`, `vers` and `datetime`")
@click.option("--dry-run", is_flag=False, default=False,
              help="Do not actually reshuffle the images. Returns the "
                   "chosen options instead. Can be grabbed with eg."
                   "`read p1 p2 freq fn <<< $(c3s_sm update_ts ...)`")
def cli_update_ts(img_path, ts_path, freq, fntempl, dry_run=False):
    """
    Extend a locally existing C3S SM time series record by appending new data
    from the image files. This will detect the time range of the time series
    data and compare it against the available image data.
    NOTE: Use the `c3s_sm reshuffle` program first do create a time series
    record to update with this function.

    \b
    Required Parameters
    -------------------
    IMG_PATH: string
        Path where previously downloaded C3S SM images are stored.
    TS_PATH: string
        Path where the time series to update are stored
    """
    # The docstring above is slightly different to the normal python one to
    # display it properly on the command line.

    if not dry_run:
        print(f"Extend time series in {ts_path} with image data from {img_path}")

    if fntempl is None:
        fntempl = _default_template

    extend_ts(img_path, ts_path, fntempl=fntempl, freq=freq)

    if dry_run:
        print(img_path, ts_path, freq)

@click.group(short_help="C3S SM Command Line Programs.")
def c3s_sm():
    pass


c3s_sm.add_command(cli_download)
c3s_sm.add_command(cli_update_img)
c3s_sm.add_command(cli_reshuffle)
c3s_sm.add_command(cli_update_ts)