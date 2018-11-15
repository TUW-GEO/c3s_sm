# -*- coding: utf-8 -*-
from c3s_sm.interface import C3S_Nc_Img_Stack
from datetime import datetime
import os
import numpy.testing as nptest
from pygeobase.object_base import  Image


def test_c3s_timestamp_for_daterange():
    parameters = ['sm', 'sm_noise']

    path = os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'TCDR', '060_dailyImages', 'combined')

    ds = C3S_Nc_Img_Stack(path, parameters)


    tstamps = ds.tstamps_for_daterange(datetime(2000, 1, 1),
                                       datetime(2000, 1, 5))
    assert len(tstamps) == 5
    assert tstamps == [datetime(2000, 1, 1),
                       datetime(2000, 1, 2),
                       datetime(2000, 1, 3),
                       datetime(2000, 1, 4),
                       datetime(2000, 1, 5)]

def test_c3s_img_stack_single_img_reading():
    parameters = ['sm']

    path = os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'TCDR', '060_dailyImages', 'combined')

    ds = C3S_Nc_Img_Stack(path, parameters)

    img = ds.read(datetime(2014,1,1)) # type: Image

    nptest.assert_almost_equal(img.data['sm'][167, 785], 0.34659, 4)

def test_c3s_img_stack_multiple_img_reading_TCDR():
    startdate, enddate = datetime(2016,4,1), datetime(2016,6,1)

    parameters = ['sm']

    path = os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'TCDR', '061_monthlyImages', 'combined')

    ds = C3S_Nc_Img_Stack(path, parameters)

    images = ds.iter_images(startdate, enddate)

    for i, img in enumerate(images):
        if i == 0:
            nptest.assert_almost_equal(img.data['sm'][167, 785], 0.32004, 4)
        if i == 1:
            nptest.assert_almost_equal(img.data['sm'][167, 785], 0.31229, 4)
        if i == 2:
            nptest.assert_almost_equal(img.data['sm'][167, 785], 0.31059, 4)


def test_c3s_img_stack_multiple_img_reading_ICDR():
    startdate, enddate = datetime(2017,7,1), datetime(2017,12,1)

    parameters = ['sm']

    path = os.path.join(os.path.dirname(__file__),
                        'c3s_sm-test-data', 'img', 'ICDR', '061_monthlyImages', 'passive')

    ds = C3S_Nc_Img_Stack(path, parameters)

    images = ds.iter_images(startdate, enddate)

    for i, img in enumerate(images):
        if i == 0:
            nptest.assert_almost_equal(img.data['sm'][167, 785], 0.23400, 4)
        if i == 1:
            nptest.assert_almost_equal(img.data['sm'][167, 785], 0.22680, 4)
        if i == 2:
            nptest.assert_almost_equal(img.data['sm'][167, 785], 0.29522, 4)



if __name__ == '__main__':
    test_c3s_img_stack_multiple_img_reading_TCDR()
    test_c3s_img_stack_single_img_reading()
    test_c3s_img_stack_multiple_img_reading_ICDR()
    test_c3s_timestamp_for_daterange()

