import os

from pathlib import Path
from datetime import datetime

# This can be overridden in the .cdsapirc file
cds_api_url = "https://cds.climate.copernicus.eu/api"

# CDSAPI_RC variable must be set or we use home dir
dotrc = os.environ.get('CDSAPI_RC', os.path.join(Path.home(), '.cdsapirc'))


def check_api_read() -> bool:
    if not os.path.isfile(dotrc):
        key = os.environ.get('CDSAPI_KEY')
        if "CDSAPI_URL" not in os.environ:
            os.environ['CDSAPI_URL'] = cds_api_url

        if key is None:
            raise ValueError(
                'Neither CDSAPI_KEY variable nor .cdsapirc file found, '
                'download will not work! '
                'Please create a .cdsapirc file with your API key. '
                'See: https://cds.climate.copernicus.eu/how-to-api'
            )
        else:
            api_ready = True
    else:
        if "CDSAPI_URL" in os.environ:
            os.environ.pop("CDSAPI_URL")   # Use URL from file
        api_ready = True

    return api_ready


variable_lut = {
    'combined': {
        'variable': 'surface_soil_moisture_volumetric',
        'type_of_sensor': 'combined'
    },
    'passive': {
        'variable': 'surface_soil_moisture_volumetric',
        'type_of_sensor': 'passive'
    },
    'active': {
        'variable': 'surface_soil_moisture_saturation',
        'type_of_sensor': 'active'
    },
    'ft': {
        'variable': 'freeze_thaw_classification',
        'type_of_sensor': None
    },
    'rzsm': {
        'variable': 'root_zone_soil_moisture_volumetric',
        'type_of_sensor': None
    }
}

freq_lut = {
    'daily': 'daily',
    'dekadal': '10_day_average',
    'monthly': 'month_average'
}

startdates = {
    'combined': datetime(1978, 11, 1),
    'ft': datetime(1978, 11, 1),
    'rzsm': datetime(1978, 11, 1),
    'passive': datetime(1978, 11, 1),
    'active': datetime(1991, 8, 5)
}

fntempl = [
    "C3S-SOILMOISTURE-L3S-SSM{unit}-{product}-{freq}-{datetime}-{record}-{version}.{subversion}.nc",
    "C3S-{product}-L3S-RZSM{unit}-{freq}-{datetime}-{record}-{version}.{subversion}.nc",
    "C3S-SOILMOISTURE-L3S-{product}-{freq}-{datetime}-{record}-{version}.{subversion}.nc"
]
