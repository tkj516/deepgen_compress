/* Invert sparse binary H for  LDPC*/
/* Author : Igor Kozintsev   igor@ifp.uiuc.edu
   Please let me know if you find bugs in this code (I did test
   it but I still have some doubts). All other comments are welcome
   too :) !
   I use a simple algorithm to invert H.
   We convert H to [I | A]
                   [junk ]
   using column reodering and row operations (junk - a few rows of H
   which are linearly dependent on the previous ones)
   G is then found as G = [A'|I]
   G is stored as array of doubles in Matlab which is very inefficient.
   Internal representation in this programm is unsigned char. Please modify
   the part which writes G if you wish.
   */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "ldpc_generate1.h"

/* modification mostly on declaring and Forced Conversion on the data types of
 * the variables -Dong Meng*/

/* Array accessors */
#define idx(arr, row, col, numrows) (arr[row + col * numrows])
#define HHidx(row, col) (HH[row + col * M])
#define GGidx(row, col) (GG[row + col * K])

void h2g(const mwSize M, const mwSize N, const IOdouble *const sr1,
         const mwSize *const irs1, const mwSize *const jcs1, const mwSize nz,
         IOdouble **Hvalues, mwSize **Hrows, mwSize **Hcols, mwSize *Hsize,
         IOuint8 **Gout, mwSize *Kout) {
  IOuint8 *HH, *GG;
  mwIndex ii, jj, *ir, *jc, rdep, tmp, d;
  double *sr2;
  mwSize K;
  mwIndex i, j, k, kk, *irs2, *jcs2;

  /* create working array HH[row][column]*/
  HH = (IOuint8 *)calloc(M * N, sizeof(IOuint8 *));
  if (HH == NULL) {
    printf("ERROR: could not allocate memory for HH.");
    exit(1);
  }

  k = 0;
  for (j = 0; j < N; j++) {
    for (i = 0; i < (jcs1[j + 1] - jcs1[j]); i++) {
      ii = irs1[k];       /* index in column j*/
      HHidx(ii, j) = sr1[k]; /* put  nonzeros */
      k++;
    }
  }

  /* invert HH matrix here */
  /* row and column indices */
  ir = (mwIndex *)malloc(M * sizeof(mwIndex));
  jc = (mwIndex *)malloc(N * sizeof(mwIndex));
  if (ir == NULL || jc == NULL) {
    printf("ERROR: could not allocate memory for ir or jc.");
    exit(1);
  }
  for (i = 0; i < M; i++)
    ir[i] = i;
  for (j = 0; j < N; j++)
    jc[j] = j;

  /* perform Gaussian elimination on H, store reodering operations */
  rdep = 0; /* number of dependent rows in H*/
  d = 0;    /* current diagonal element */

  while ((d + rdep) < M) { /* cycle through independent rows of H */

    j = d; /* current column index along row ir[d] */
    while ((HHidx(ir[d], jc[j]) == 0) && (j < (N - 1)))
      j++;                  /* find first nonzero element in row i */
    if (HHidx(ir[d], jc[j])) { /* found nonzero element. It is "1" in GF2 */

      /* swap columns */
      tmp = jc[d];
      jc[d] = jc[j];
      jc[j] = tmp;

      /* eliminate current column using row operations */
      for (ii = 0; ii < M; ii++)
        if (HHidx(ir[ii], jc[d]) && (ir[ii] != ir[d]))
          for (jj = d; jj < N; jj++)
            HHidx(ir[ii], jc[jj]) =
                (HHidx(ir[ii], jc[jj]) + HHidx(ir[d], jc[jj])) % 2;
    } else {  /* all zeros -  need to delete this row and update indices */
      rdep++; /* increase number of dependent rows */
      tmp = ir[d];
      ir[d] = ir[M - rdep];
      ir[M - rdep] = tmp;
      d--; /* no diagonal element is found */
    }
    d++; /* increase the number of diagonal elements */
  }      /*while i+rdep*/
         /* done inverting HH */

  K = N - M + rdep; /* true K */

  /* create G matrix  G = [A'| I] if H = [I|A]*/
  GG = (IOuint8 *)malloc(K * N * sizeof(IOuint8 *));
  if (GG == NULL) {
    printf("ERROR: could not allocate memory for GG.");
    exit(1);
  }
  for (i = 0; i < K; i++)
    for (j = 0; j < (N - K); j++) {
      tmp = (N - K + i);
      GGidx(i, j) = HHidx(ir[j], jc[tmp]);
    }

  /* FIXME if GG was calloc'd, we wouldn't need to write all these zeros (just
   * the ones) */
  for (i = 0; i < K; i++)
    for (j = (N - K); j < N; j++)
      GGidx(i, j) = (i == (j - N + K));

  sr2 = (double *)calloc(nz, sizeof(double));
  irs2 = (mwSize *)calloc(nz, sizeof(mwSize));
  jcs2 = (mwSize *)calloc(N + 1, sizeof(mwSize));
  if (sr2 == NULL || irs2 == NULL || jcs2 == NULL) {
    printf("ERROR: could not allocate sparse array.");
    exit(1);
  }
  /* Write H_OUT swapping columns according to jc */
  k = 0;
  for (j = 0; (j < N); j++) {
    jcs2[j] = k;
    tmp = jcs1[jc[j] + 1] - jcs1[jc[j]];
    for (i = 0; i < tmp; i++) {
      kk = jcs1[jc[j]] + i;
      sr2[k] = sr1[kk];
      irs2[k] = irs1[kk];
      k++;
    }
  }
  jcs2[N] = k;

  /* Update outputs */
  *Hvalues = sr2;
  *Hrows = irs2;
  *Hcols = jcs2;
  *Hsize = nz;
  *Gout = GG;
  *Kout = K;

  /* free the memory */
  free(HH);
  free(ir);
  free(jc);
  return;
}

