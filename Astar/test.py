# -*- coding: utf-8 -*-
import ctypes
import numpy as np
import numpy.ctypeslib as npct

astar = ctypes.cdll.LoadLibrary('./AstarGED.so')

INT = ctypes.c_int
PINT = ctypes.POINTER(ctypes.c_int)
PPINT = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))

astar.init.restype = ctypes.c_void_p
astar.ged.restype = PINT

def int2ArrayToPointer(arr): #Converts a 2D numpy to ctypes 2D array.
    # Init needed data types
    ARR_DIMX = INT * arr.shape[1]
    ARR_DIMY = PINT * arr.shape[0]
    # Init pointer
    arr_ptr = ARR_DIMY()
    # Fill the 2D ctypes array with values
    for i, row in enumerate(arr):
        arr_ptr[i] = ARR_DIMX()
        for j, val in enumerate(row):
            arr_ptr[i][j] = val
    return arr_ptr

def CT(input): # convert type
    ctypes_map = {int: ctypes.c_int, float: ctypes.c_double, str: ctypes.c_char_p}
    input_type = type(input)
    if input_type is list:
        length = len(input)
        if length == 0:
            print("convert type failed...input is " + input)
            return None
        else:
            arr = (ctypes_map[type(input[0])] * length)()
            for i in range(length):
                arr[i] = bytes(input[i], encoding="utf-8") if (type(input[0]) is str) else input[i]
            return arr
    else:
        if input_type in ctypes_map:
            return ctypes_map[input_type](bytes(input, encoding="utf-8") if type(input) is str else input)
        else:
            print("convert type failed...input is " + input)
            return None


if __name__ == '__main__':
    filename = "D:/datasets/GED/AIDS/AIDS_txt.txt"
    upper_bound = 100
    topk = 2
    g1_id = "15200"
    g2_id = "16895"
    MO = np.array([0,1,2,3])
    matched_nodes = np.array([[0,1,2,3],[2,1,0,3]])
    matched_nodes = matched_nodes.T

    astar.init(CT(filename))
    res = astar.ged(CT(g1_id), CT(g2_id), CT(upper_bound),  npct.as_ctypes(MO), int2ArrayToPointer(matched_nodes), CT(topk))

    print("ged:{}, res:{}".format(res[0], res[1]))