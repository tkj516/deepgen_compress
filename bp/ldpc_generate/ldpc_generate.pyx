import cython
import numpy as np
cimport numpy as np

import scipy.sparse as sparse

cdef extern from "ldpc_generate1.h":
    ctypedef long long mwSize
    ctypedef long long mwIndex
    ctypedef int IOint
    ctypedef double IOdouble
    ctypedef unsigned char IOuint8;
    void _generate "ldpc_generate" (mwSize M, mwSize N, IOdouble t,
            mwIndex q, IOint seed, IOdouble **sparseValues,
            mwSize **sparseRows, mwSize **sparseCols, mwSize *nzmaxOut)
    void _h2g "h2g" (const mwSize M, const mwSize N, const IOdouble *const sr1,
            const mwSize *const irs1, const mwSize *const jcs1, const mwSize nz,
            IOdouble **Hvalues, mwSize **Hrows, mwSize **Hcols, mwSize *Hsize,
            IOuint8 **Gout, mwSize * Ksize)

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def generate(IOint M, IOint N, IOdouble t, mwIndex q, IOint seed):
    cdef IOdouble * vals
    cdef mwSize * rows
    cdef mwSize * cols
    cdef mwSize nzmax
    _generate(M, N, t, q, seed, &vals, &rows, &cols, &nzmax)

    cdef IOdouble[::1] valsView = <IOdouble[:nzmax]>vals
    cdef mwIndex[::1] rowsView = <mwIndex[:nzmax]>rows
    cdef mwIndex[::1] colsView = <mwIndex[:(N+1)]>cols
    return sparse.csc_matrix((np.array(valsView), rowsView, colsView))

@cython.boundscheck(False)
@cython.wraparound(False)
def generateGH(IOint M, IOint N, IOdouble t, mwIndex q, IOint seed):
    cdef IOdouble * vals
    cdef mwSize * rows
    cdef mwSize * cols
    cdef mwSize nzmax
    _generate(M, N, t, q, seed, &vals, &rows, &cols, &nzmax)
    cdef IOdouble * Hvals
    cdef mwSize * Hrows
    cdef mwSize * Hcols
    cdef mwSize Hsize
    cdef IOuint8 * G
    cdef mwSize K
    _h2g(M, N, vals, rows, cols, nzmax, &Hvals, &Hrows, &Hcols, &Hsize, &G, &K)

    cdef IOdouble[::1] HvalsView = <IOdouble[:Hsize]>Hvals
    cdef mwIndex[::1] HrowsView = <mwIndex[:Hsize]>Hrows
    cdef mwIndex[::1] HcolsView = <mwIndex[:(N+1)]>Hcols
    cdef IOuint8[::1] Gview = <IOuint8[:(K * N)]>G

    return (sparse.csc_matrix((np.array(HvalsView), HrowsView, HcolsView)),
            np.array(Gview, dtype=np.uint8).reshape((K, N), order='F'))
