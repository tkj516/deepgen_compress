#ifndef _LDPC_GENERATE1_H_
#define _LDPC_GENERATE1_H_

/* Convert Matlab-specific stuff to regular stuff */
typedef long long mwSize;
typedef long long mwIndex;

typedef int IOint;
typedef double IOdouble;

typedef unsigned char IOuint8;

/* TODO FIXME these two IOdouble arrays, Hvalues and sr1/sparseValues, they can
 * be uint8 just like Gout. */
void h2g(const mwSize M, const mwSize N, const IOdouble *const sr1,
         const mwSize *const irs1, const mwSize *const jcs1, const mwSize nz,
         IOdouble **Hvalues, mwSize **Hrows, mwSize **Hcols, mwSize *Hsize,
         IOuint8 **Gout, mwSize *Kout);

void ldpc_generate(const mwSize M, const mwSize N, const IOdouble t,
                   const mwIndex q, const IOint seed, IOdouble **sparseValues,
                   mwSize **sparseRows, mwSize **sparseCols, mwSize *nzmaxOut);

#endif


