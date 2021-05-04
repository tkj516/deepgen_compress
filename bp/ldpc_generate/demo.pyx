import cython
import numpy as np
cimport numpy as np

cdef extern from "demolibrary.h":
    void _scale "scale" (float*, int, float)

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def scale(np.ndarray x, float mult):
    _scale(<float*> np.PyArray_DATA(x), x.size, mult)
    return x

def foo(float x):
    cdef float y
    y = 2 * x
    return y

