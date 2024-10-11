# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict


class C3S_SM_TS_Attrs(object):
    '''Default, common metadata for daily and monthly, dekadal products'''

    def __init__(self, sensor_type, version):
        '''
        Parameters
        ----------
        sensor_type : str
            Sensor type: active, passive, combined
        version : str
            Version name to read attributes for
        sub_version : str
            Sub version to read attributes for
        '''
        self.version = version

        self.product_datatype_str = {
            'active': 'SSMS',
            'passive': 'SSMV',
            'combined': 'SSMV'
        }

        self.sensor_type = sensor_type

        self.atts_sensor_type(sensor_type)

    def atts_sensor_type(self, sensor_type='active'):
        if sensor_type == "active":
            self.sm_units = "percentage (%)"
            self.sm_uncertainty_units = "percentage (%)"
            self.sm_full_name = 'Percent of Saturation Soil Moisture Uncertainty'
            self.sm_uncertainty_full_name = 'Percent of Saturation Soil Moisture Uncertainty'
        else:
            self.sm_units = "m3 m-3"
            self.sm_uncertainty_units = "m3 m-3"
            self.sm_full_name = 'Volumetric Soil Moisture'
            self.sm_uncertainty_full_name = 'Volumetric Soil Moisture Uncertainty'

    def dn_flag(self):
        dn_flag_dict = OrderedDict([
            (1, "day"),
            (2, 'night'),
        ])
        self.dn_flag_values = np.array(list(dn_flag_dict.keys()))
        self.dn_flag_meanings = np.array(list(dn_flag_dict.values()))

        return self.dn_flag_values, self.dn_flag_meanings

    def flag(self):
        flag_dict = OrderedDict([
            (1, 'snow_coverage_or_temperature_below_zero'),
            (2, 'dense_vegetation'),
            (3,
             'others_no_convergence_in_the_model_thus_no_valid_sm_estimates'),
            (4, 'soil_moisture_value_exceeds_physical_boundary'),
            (5, 'weight_of_measurement_below_threshold'),
            (6, 'all_datasets_deemed_unreliable'),
            (7, 'barren_ground_advisory_flag'),
            (8, 'not_used'),
        ])

        self.flag_values = np.array(list(flag_dict.keys()))
        self.flag_meanings = np.array(list(flag_dict.values()))

        return self.flag_values, self.flag_meanings

    def freqbandID_flag(self):
        freqbandID_flag_dict = OrderedDict([
            (1, 'L14'),
            (2, 'C53'),
            (3, 'C66'),
            (4, 'C68'),
            (5, 'C69'),
            (6, 'C73'),
            (7, 'X107'),
            (8, 'K194'),
            (9, 'MODEL'),
        ])

        self.freqbandID_flag_values = np.array(
            list(freqbandID_flag_dict.keys()))
        self.freqbandID_flag_meanings = np.array(
            list(freqbandID_flag_dict.values()))

        return self.freqbandID_flag_values, self.freqbandID_flag_meanings

    def sensor_flag(self):
        sensor_flag_dict = OrderedDict([
            (1, 'SMMR'),
            (2, 'SSMI'),
            (3, 'TMI'),
            (4, 'AMSRE'),
            (5, 'WindSat'),
            (6, 'AMSR2'),
            (7, 'SMOS'),
            (8, 'AMIWS'),
            (9, 'ASCATA'),
            (10, 'ASCATB'),
            (11, 'SMAP'),
            (12, 'MODEL'),
            (13, 'GPM'),
            (14, 'FY3B'),
            (15, 'FY3D'),
            (16, 'ASCATC'),
            (17, 'FY3C'),
        ])

        self.sensor_flag_values = np.array(list(sensor_flag_dict.keys()))
        self.sensor_flag_meanings = np.array(list(sensor_flag_dict.values()))

        return self.sensor_flag_values, self.sensor_flag_meanings

    def mode_flag(self):
        mode_flag_dict = OrderedDict([
            (1, 'ascending'),
            (2, 'descending'),
        ])
        self.mode_flag_values = np.array(list(mode_flag_dict.keys()))
        self.mode_flag_meanings = np.array(list(mode_flag_dict.values()))

        return self.mode_flag_meanings, self.mode_flag_values


class C3S_daily_tsatt_nc(C3S_SM_TS_Attrs):

    def __init__(self, cdr_type: str, sensor_type: str, version: str, cls):

        self.general_attrs = cls(sensor_type=sensor_type, version=version)

        self.version = self.general_attrs.version
        sensor_type = self.general_attrs.sensor_type

        self.freq = 'daily'
        self.cdr_type = cdr_type
        self.general_attrs.atts_sensor_type(sensor_type)
        self.general_attrs.dn_flag()
        self.general_attrs.flag()
        self.general_attrs.freqbandID_flag()
        self.general_attrs.mode_flag()
        self.general_attrs.sensor_flag()

        self.ts_attributes = {
            'dnflag': {
                'full_name': 'Day / Night Flag',
                'flag_values': self.general_attrs.dn_flag_values,
                'flag_meanings': self.general_attrs.dn_flag_meanings
            },
            'flag': {
                'full_name': 'Flag',
                'flag_values': self.general_attrs.flag_values,
                'flag_meanings': self.general_attrs.flag_meanings
            },
            'freqbandID': {
                'full_name': 'Frequency Band Identification',
                'flag_values': self.general_attrs.freqbandID_flag_values,
                'flag_meanings': self.general_attrs.freqbandID_flag_meanings
            },
            'mode': {
                'full_name': 'Satellite Mode',
                'flag_values': self.general_attrs.mode_flag_values,
                'flag_meanings': self.general_attrs.mode_flag_meanings
            },
            'sensor': {
                'full_name': 'Sensor',
                'flag_values': self.general_attrs.sensor_flag_values,
                'flag_meanings': self.general_attrs.sensor_flag_meanings
            },
            'sm': {
                'full_name': self.general_attrs.sm_full_name,
                'units': self.general_attrs.sm_units
            },
            'sm_uncertainty': {
                'full_name': self.general_attrs.sm_uncertainty_full_name,
                'units': self.general_attrs.sm_uncertainty_units
            },
            't0': {
                'full_name': 'Observation Timestamp',
                'units': 'days since 1970-01-01 00:00:00 UTC'
            }
        }

        _prod = sensor_type.upper()
        _freq = self.freq.upper()
        _cdr = self.cdr_type.upper()
        _vers = self.version

        product_name = " ".join([
            'C3S', 'SOILMOISTURE', 'L3S',
            self.general_attrs.product_datatype_str[sensor_type].upper(),
            _prod, _freq, _cdr, _vers
        ])

        self.global_attr = {
            'product_full_name': product_name,
            'product': str(_prod),
            'temporal_sampling': str(_freq),
            'cdr': str(_cdr),
            'version': str(_vers),
            'resolution': '0.25 degree'
        }


class C3S_dekmon_tsatt_nc(object):
    """
    Attributes for c3s dekadal and monthly for active, passive and combined
    tcdr and icdr timeseries files.
    """

    def __init__(self, freq: str, cdr_type: str, sensor_type: str, version: str,
                 cls):

        self.general_attrs = cls(sensor_type=sensor_type, version=version)

        self.version = self.general_attrs.version
        sensor_type = self.general_attrs.sensor_type

        self.freq = freq
        self.cdr_type = cdr_type
        self.general_attrs.atts_sensor_type(sensor_type)
        self.general_attrs.dn_flag()
        self.general_attrs.freqbandID_flag()

        self.general_attrs.sensor_flag()

        self.ts_attributes = {
            'freqbandID': {
                'full_name': 'Frequency Band Identification',
                'flag_values': self.general_attrs.freqbandID_flag_values,
                'flag_meanings': self.general_attrs.freqbandID_flag_meanings
            },
            'sensor': {
                'full_name': 'Sensor',
                'flag_values': self.general_attrs.sensor_flag_values,
                'flag_meanings': self.general_attrs.sensor_flag_meanings
            },
            'nobs': {
                'full_name': 'Number of valid observation'
            },
            'sm': {
                'full_name': self.general_attrs.sm_full_name,
                'units': self.general_attrs.sm_units
            }
        }

        _prod = sensor_type.upper()
        _freq = self.freq.upper()
        _cdr = self.cdr_type.upper()
        _vers = self.version

        product_name = " ".join([
            'C3S', 'SOILMOISTURE', 'L3S',
            self.general_attrs.product_datatype_str[sensor_type].upper(),
            _prod, _freq, _cdr, _vers
        ])

        self.global_attr = {
            'product_full_name': product_name,
            'product': str(_prod),
            'temporal_sampling': str(_freq),
            'cdr': str(_cdr),
            'version': str(_vers),
            'resolution': '0.25 degree'
        }