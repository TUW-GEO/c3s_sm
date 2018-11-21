# -*- coding: utf-8 -*-

from c3s_sm.grid import C3SCellGrid
from c3s_sm.grid import C3SLandGrid
import numpy as np


def test_C3SCellGrid():
    grid = C3SCellGrid()
    gp, dist = grid.find_nearest_gpi(75.625, 14.625)
    assert gp == 602942
    lon, lat = grid.gpi2lonlat(602942)
    assert lon == 75.625
    assert lat == 14.625
    assert np.where(grid.get_grid_points()[0] == 602942)[0][0] == 434462 # index
    assert grid.get_grid_points()[1][434462] == lon
    assert grid.get_grid_points()[2][434462] == lat
    assert grid.gpi2cell(602942) == 1856
    assert grid.gpis.size == 1036800
    assert grid.gpis[0] == 1035360
    assert np.unique(grid.get_grid_points()[3]).size == 2592


def test_landgrid():
    grid = C3SLandGrid()
    gp, dist = grid.find_nearest_gpi(75.625, 14.625)
    assert gp == 602942
    lon, lat = grid.gpi2lonlat(602942)
    assert lon == 75.625
    assert lat == 14.625
    assert np.where(grid.get_grid_points()[0] == 602942)[0][0] == 177048  # index
    assert grid.get_grid_points()[1][177048] == lon
    assert grid.get_grid_points()[2][177048] == lat
    assert grid.gpi2cell(602942) == 1856
    assert grid.gpis.size == 1036800
    assert grid.activegpis.size == 244243
    assert grid.gpis[0] == 1035360
    assert grid.activegpis[0] == 999942
    assert np.unique(grid.get_grid_points()[3]).size == 1001
