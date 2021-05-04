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
#include "mex.h"

/* Input Arguments: tentative H matrix*/
#define H_IN prhs[0]

/* Output Arguments: final matrices*/
#define H_OUT plhs[0]
#define G_OUT plhs[1]

/* modification mostly on declaring and Forced Conversion on the data types of
 * the variables -Dong Meng*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  unsigned char **HH, **GG;
  mwIndex ii, jj, *ir, *jc, rdep, tmp, d;
  double *sr1, *sr2, *g;
  mwSize N, M, K, nz;
  mwIndex i, j, k, kk, *irs1, *jcs1, *irs2, *jcs2;

  /* Check for proper number of arguments */
  if (nrhs != 1) {
    mexErrMsgTxt("h2g requires one input arguments.");
  } else if (nlhs != 2) {
    mexErrMsgTxt("h2g requires two output arguments.");
  } else if (!mxIsSparse(H_IN)) {
    mexErrMsgTxt("h2g requires sparse H matrix.");
  }

  /* read sparse matrix H */
  sr1 = mxGetPr(H_IN);
  irs1 = mxGetIr(H_IN);  /* row */
  jcs1 = mxGetJc(H_IN);  /* column */
  nz = mxGetNzmax(H_IN); /* number of nonzero elements (they are ones)*/
  M = mxGetM(H_IN);
  N = mxGetN(H_IN);

  /* create working array HH[row][column]*/
  HH = (unsigned char **)mxMalloc(M * sizeof(unsigned char *));
  for (i = 0; i < M; i++) {
    HH[i] = (unsigned char *)mxMalloc(N * sizeof(unsigned char));
  }
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      HH[i][j] = 0; /* initialize all to zero */

  k = 0;
  for (j = 0; j < N; j++) {
    for (i = 0; i < (jcs1[j + 1] - jcs1[j]); i++) {
      ii = irs1[k];       /* index in column j*/
      HH[ii][j] = sr1[k]; /* put  nonzeros */
      k++;
    }
  }

  /* invert HH matrix here */
  /* row and column indices */
  ir = (mwIndex *)mxMalloc(M * sizeof(mwIndex));
  jc = (mwIndex *)mxMalloc(N * sizeof(mwIndex));
  for (i = 0; i < M; i++)
    ir[i] = i;
  for (j = 0; j < N; j++)
    jc[j] = j;

  /* perform Gaussian elimination on H, store reodering operations */
  rdep = 0; /* number of dependent rows in H*/
  d = 0;    /* current diagonal element */

  while ((d + rdep) < M) { /* cycle through independent rows of H */

    j = d; /* current column index along row ir[d] */
    while ((HH[ir[d]][jc[j]] == 0) && (j < (N - 1)))
      j++;                  /* find first nonzero element in row i */
    if (HH[ir[d]][jc[j]]) { /* found nonzero element. It is "1" in GF2 */

      /* swap columns */
      tmp = jc[d];
      jc[d] = jc[j];
      jc[j] = tmp;

      /* eliminate current column using row operations */
      for (ii = 0; ii < M; ii++)
        if (HH[ir[ii]][jc[d]] && (ir[ii] != ir[d]))
          for (jj = d; jj < N; jj++)
            HH[ir[ii]][jc[jj]] = (HH[ir[ii]][jc[jj]] + HH[ir[d]][jc[jj]]) % 2;
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
  GG = (unsigned char **)mxMalloc(K * sizeof(unsigned char *));
  for (i = 0; i < K; i++) {
    GG[i] = (unsigned char *)mxMalloc(N * sizeof(unsigned char));
  }
  for (i = 0; i < K; i++)
    for (j = 0; j < (N - K); j++) {
      tmp = (N - K + i);
      GG[i][j] = HH[ir[j]][jc[tmp]];
    }

  for (i = 0; i < K; i++)
    for (j = (N - K); j < N; j++)
      if (i == (j - N + K)) /* diagonal */
        GG[i][j] = 1;
      else
        GG[i][j] = 0;

  /* NOTE, it is very inefficient way to store G. Change to taste!*/
  G_OUT = mxCreateDoubleMatrix(K, N, mxREAL);
  /* Assign pointers to the output matrix */
  g = mxGetPr(G_OUT);
  for (i = 0; i < K; i++)
    for (j = 0; j < N; j++)
      g[i + j * K] = GG[i][j];

  H_OUT = mxCreateSparse(M, N, nz, mxREAL);
  sr2 = mxGetPr(H_OUT);
  irs2 = mxGetIr(H_OUT); /* row */
  jcs2 = mxGetJc(H_OUT); /* column */
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

  /* free the memory */
  for (j = 0; j < M; j++) {
    mxFree(HH[j]);
  }
  mxFree(HH);
  mxFree(ir);
  mxFree(jc);
  for (i = 0; i < K; i++) {
    mxFree(GG[i]);
  }
  mxFree(GG);
  return;
}

