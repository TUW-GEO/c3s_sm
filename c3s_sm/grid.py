# -*- coding: utf-8 -*-

from smecv_grid.grid import SMECV_Grid_v042

def C3SCellGrid():
    return SMECV_Grid_v042(None)

def C3SLandGrid():
    return SMECV_Grid_v042('land')


