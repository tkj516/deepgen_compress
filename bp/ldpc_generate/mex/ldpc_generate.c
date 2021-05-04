/* Generate H for  LDPC*/
/* Based on sparse.c by Matt Davey <mcdavey@mrao.cam.ac.uk> with his
 * permission*/
#include <math.h>
#include "mex.h"

/* Input Arguments: parameters*/
#define M_IN prhs[0]    /* number of parity checks */
#define N_IN prhs[1]    /* blocklength */
#define T_IN prhs[2]    /* mean column weight */
#define Q_IN prhs[3]    /* GF base  */
#define SEED_IN prhs[4] /* seed for random generator */

/* Output Arguments: matrices*/
#define H_OUT plhs[0]

/* modification mostly on declaring and Forced Conversion on the data types of
 * the variables -Dong Meng*/

void setupRand(int seed) { srand((unsigned int)seed); }

double uniformRand() { return ((double)rand() / (double)RAND_MAX); }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  short **M_list, **N_list, *M_target;
  double *pp, *sr, *s, *ss;
  mwSize N, M, nzmax;
  mwIndex q, i, j, k, *irs, *jcs, l, tr, tm, tc, done, redo, tmp, regime,
      tr_max, t_max, m_low;
  float t;
  long seed;
  void adjust(mwIndex *, mwIndex *, mwIndex *, mwIndex *);
  unsigned int K2, M2;
  char c;
  mxArray *arg_in[2], *arg_out[1]; /* to call rand generator of Matlab*/

  /* Check for proper number of arguments */
  if (nrhs != 5) {
    mexErrMsgTxt("GENERATE requires five input arguments.");
  } else if (nlhs > 1) {
    mexErrMsgTxt("GENERATE requires one output argument.");
  }

  pp = mxGetPr(N_IN);
  N = (mwSize)(*pp);
  pp = mxGetPr(M_IN);
  M = (mwSize)(*pp);
  pp = mxGetPr(T_IN);
  t = (float)(*pp);
  pp = mxGetPr(Q_IN);
  q = (int)(*pp);
  pp = mxGetPr(SEED_IN);
  seed = (int)(*pp);

  arg_in[0] = mxCreateString("state");
  arg_in[1] = mxCreateDoubleMatrix((mwSize)(1), (mwSize)(1), mxREAL);
  s = mxGetPr(arg_in[1]);
  s[0] = seed; /* this will be used to call rand*/

  /* initialize random generator */
  mexCallMATLAB(0, NULL, 2, arg_in, "rand"); /* rand('state',seed) */
  s[0] = 1;                                  /* from now on use s to store "1"*/
  setupRand(seed);

  /* I have no idea about the details of the following - please
  ask the author - :).  igor*/

  /* Generate some sparse matrices for error-correcting codes.  Supply
   * the following parameters:
   * 	N: Blocklength
   * 	M: Number of parity checks
   * 	t: Mean column weight.
   *
   * If t<3, then we generate weight 2 columns systematically in the
   * form of blocks of identity matrices to reduce probability of
   * getting short cycle lengths.  Can't have more than M weight 2
   * columns, though.
   *
   * Having generated any weight 2 columns, we fill the rest.  Calculate
   * number of columns to fill with floor(t) and number with ceiling(t).
   * Find mean row weight and calculate number of rows to fill with
   * floor(r) and ceiling(r).
   *
   * We fill rows as follows:
   * (Regime 0): Fill so that rows contain <= tr ones until M-tm rows
   * 	    contain tr ones.
   * (Regime 1): Fill so that rows contain <= (tr+1) ones until tm rows
   * 	    contain (tr+1) ones.
   * (Regime 2): Fill remaining rows containing < tr ones until done.
   */

  /* N=6112;M=4512;t=2.3;*/

  t_max = (mwSize)ceil(t);
  M_list = (short **)mxMalloc(N * sizeof(short *));
  M_target = (short *)mxMalloc(M * sizeof(short *));
  N_list = (short **)mxMalloc(M * sizeof(short *));
  for (i = 0; i < N; i++) {
    M_list[i] = (short *)mxMalloc((t_max + 1) * sizeof(short));
  }
  i = 0;
  /* Do we have any weight 2 columns?
   *
   * If so, do this first.  Remember that M mightn't have a large
   * power of two as a divisor, so might need to find some M'<M to use
   * as unit length.
   */
  K2 = 0;
  if (t < 3) {
    K2 = ceil((double)N * (3 - t));
    if (K2 > M)
      mexErrMsgTxt("GENERATE: Can't have more than M weight 2 columns.");
    j = 2;
    done = 0;
    for (i = 0; !done; i++) {
      M2 = floor((double)M / (double)j);
      if ((M2 * (j - 1)) >= K2)
        done = 1;
      j *= 2;
    }
    M2 *= (j / 4);
  }
  /*
   * i contains number of identity blocks we'll need.... */
  tr = ((short)floor((double)(t * N) / (double)M));
  /* Now we want `tr' to be final minimum row weight, `tr_max' to be
   * final maximum row weight, `tm' to be number of rows which will
   * have weight greater than `tr'.  'tc' will be a running count of
   * how many rows we still have to fill up to weight `tr'.  Once we
   * hit this many, we can start overfilling rows.
   */
  if (i > tr) {
    tr_max = i;
    /* If identity blocks make overheavy rows, we need to calculate
     * the minimum row weight.
     */
    done = 0;
    k = 1;
    j = floor((double)t * N) - 2 * K2; /* Number of ones left to distribute */
    for (i = 0; !done; i++) {
      /* (M-2*M2) rows will be empty after identity blocks */
      j -= ((M - 2 * M2) + (2 * M2 * (k - 1)) / k);
      if (j < 0) {
        done = 1;
      } else {
        k *= 2;
      }
    }
    tr = i - 1;
    tm = M + j;
  } else {
    /* This is easier! */
    tr_max = tr + 1;
    tm = (mwSize)floor((((double)t * N) / (double)M - tr) * M + 0.5);
  }
  tc = M - tm;
  for (i = 0; i < M; i++) {
    N_list[i] = (short *)mxMalloc((tr_max + 1) * sizeof(short));
  }
  for (i = 0; i < M; i++) {
    N_list[i][0] = 0;
  }

  regime = 0;
  /* Generate weight 2 columns.  First create two identity matrices on
   * top of each other, then two 1/2 size ones in the lower rows of
   * the matrix, and so on.
   *
   * j:   length of current identity block
   * k:   base of this block
   * i:   current column position
   */
  j = M2;
  k = 0;
  i = 0;
  while (i < K2) {
    for (; (i - k) < j && i < K2; i++) {
      M_list[i][0] = 2;
      M_list[i][1] = i;
      M_list[i][2] = i + j;
      N_list[i][0]++;
      if (N_list[i][0] == tr)
        adjust(&tm, &tr, &tc, &regime);
      N_list[i][N_list[i][0]] = i;
      N_list[i + j][0]++;
      if (N_list[i + j][0] == tr)
        adjust(&tm, &tr, &tc, &regime);
      N_list[i + j][N_list[i + j][0]] = i;
    }
    k = i;
    j /= 2;
  }

  /* Now fill the unsystematic columns, ensuring weight per row as even as poss.
   */
  i = K2;
  if (K2 == 0) {
    /* Fill low weight columns */
    for (i = 0; i < (mwSize)(N * (t_max - t) + 0.5); i++) {
      for (k = 1; k <= (mwSize)floor(t); k++) {
        done = 0;
        do {
          j = (short)floor(M * uniformRand());
          if (uniformRand() < (1 - (double)N_list[j][0] / (double)tr)) {
            done = 1;
            for (l = 1; l < k; l++)
              if (j == M_list[i][l])
                done = 0;
          }
        } while (!done);
        N_list[j][0]++;
        N_list[j][N_list[j][0]] = i;
        if (N_list[j][0] == tr)
          adjust(&tm, &tr, &tc, &regime);
        M_list[i][k] = j;
      }
      M_list[i][0] = k - 1;
    }
  }
  redo = 1;
  for (; i < N; i++) {
    fprintf(stderr, "%d\r", i);
    for (k = 1; k <= t_max; k++) {
      done = 0;
      do {
        /* find the lowest weight rows, and fill one of them */
        if (redo) {
          l = tr_max;
          for (j = 0; j < M; j++)
            if (N_list[j][0] < l)
              l = N_list[j][0];
          m_low = 0;
          for (j = 0; j < M; j++)
            if (N_list[j][0] == l) {
              M_target[m_low] = j;
              m_low++;
            }
        }
        j = M_target[tmp = (short)floor(m_low * uniformRand())];
        /*	if(ss[0]<(1-(double)N_list[j][0]/(double)tr)) {*/
        done = 1;
        for (l = 1; l < k; l++)
          if (j == M_list[i][l])
            done = 0;
        if (done == 1) {
          if (m_low == 1)
            redo = 1;
          else {
            for (; tmp < (m_low - 1); tmp++)
              M_target[tmp] = M_target[tmp + 1];
            m_low--;
            redo = 0;
          }
        }
        /*	}*/

      } while (!done);
      N_list[j][0]++;
      N_list[j][N_list[j][0]] = i;
      if (N_list[j][0] == tr)
        adjust(&tm, &tr, &tc, &regime);
      M_list[i][k] = j;
    }
    M_list[i][0] = k - 1;
  }
  tr = ((short)ceil((double)(3 * N - K2) / (double)M));

  for (i = 0; i < M; i++) {
    mxFree(N_list[i]);
  }
  mxFree(N_list);

  /* done generating H matrix */

  /* Allocate space for sparse matrix */
  nzmax = 0;
  for (j = 0; j < N; j++)
    nzmax += M_list[j][0];
  /* NOTE: The maximum number of non-zero elements cannot be less
     than the number of columns in the matrix. */
  if (N > nzmax) {
    nzmax = N;
  }
  plhs[0] = mxCreateSparse(M, N, nzmax, mxREAL);
  sr = mxGetPr(plhs[0]);
  irs = mxGetIr(plhs[0]); /* row */
  jcs = mxGetJc(plhs[0]); /* column */

  /* Copy nonzeros */
  k = 0;
  for (j = 0; (j < N); j++) {
    jcs[j] = k;
    for (i = 1; (i <= M_list[j][0]); i++) {
      sr[k] = 1;
      irs[k] = M_list[j][i];
      k++;
    }
  }
  jcs[N] = k;

  for (i = 0; i < N; i++) {
    mxFree(M_list[i]);
  }
  mxFree(M_list);

  return;
}

void adjust(mwIndex *tm, mwIndex *tr, mwIndex *tc, mwIndex *regime) {
  switch (*regime) {
  case 0:
    (*tc)--;
    if ((*tc) == 0) {
      *regime = 1;
      (*tr)++;
    }
    break;
  case 1:
    (*tm)--;
    if ((*tm) == 0) {
      *regime = 2;
      (*tr)--;
    }
    break;
  }
}

