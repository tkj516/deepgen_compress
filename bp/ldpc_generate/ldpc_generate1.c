/* Generate H for  LDPC*/
/* Based on sparse.c by Matt Davey <mcdavey@mrao.cam.ac.uk> with his
 * permission*/
#include "ldpc_generate1.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define mexPrintf printf
#define mexErrMsgTxt printf

/* modification mostly on declaring and Forced Conversion on the data types of
 * the variables -Dong Meng*/

/* Uniform random number generator API */
void setupRand(IOint seed) {
  srand((unsigned) seed);
}

IOdouble uniformRand() { return ((IOdouble)rand() / (IOdouble)RAND_MAX); }

/* Worker function */

/**
 * LDPC generator
 * @param[in] M number of parity checks
 * @param[in] N blocklength
 * @param[in] t mean column weight
 * @param[in] q Galois field (GF) base
 * @param[in] seed random number generator seed
 * @param[out] sparseValues pointer to array storing sparse values
 * @param[out] sparseRows pointer to array of row indexes (CSC format)
 * @param[out] sparseCols pointer to array of columnar elements (CSC format)
 * @param[out] nzmaxOut pointer to max number of non-zero elements
 *
 * The return values specify a sparse matrix in CSC (compressed sparse columns)
 * format of size M by N (M rows, N columns). `sparseCols` and `sparseRows` are
 * `nzmaxOut`-long, while `sparseCols` is `N + 1`-long. The `n`th entry in
 * `sparseValues` goes in row `sparseRows[n]`. The `j`th entry in `sparseCols`
 * indicates how many non-zero entries exist in columns `0:(j-1)`; therefore,
 * its first entry, `sparseCols[0]`, is always 0, and its last entry,
 * `sparseCols[N]`, is the total number of non-zero elements in the sparse
 * array. See [1--4] for description and usage of this format for sparse
 * matrixes.
 *
 * [1] http://stackoverflow.com/a/6098429/500207
 * [2] http://stackoverflow.com/a/20562225/500207
 * [3] Versioned Wikipedia entry: https://en.wikipedia.org/w/index.php?title=Sparse_matrix&oldid=731601498#Compressed_sparse_column_.28CSC_or_CCS.29
 * [4] http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html
 */
void ldpc_generate(const mwSize M, const mwSize N, const IOdouble t,
                   const mwIndex q, const IOint seed, IOdouble **sparseValues,
                   mwSize **sparseRows, mwSize **sparseCols, mwSize *nzmaxOut) {
  short **M_list, **N_list, *M_target;
  double *sr;
  mwSize nzmax;
  mwIndex i, j, k, *irs, *jcs, l, tr, tm, tc, done, redo, tmp, regime,
      tr_max, t_max, m_low;
  void adjust(mwIndex *, mwIndex *, mwIndex *, mwIndex *);
  unsigned int K2, M2;

  /* Set up RNG */
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
  M_list = (short **)malloc(N * sizeof(short *));
  M_target = (short *)malloc(M * sizeof(short *));
  N_list = (short **)malloc(M * sizeof(short *));
  if (M_list == NULL || M_target == NULL || N_list == NULL) {
    printf("ERROR: could not allocate lists.");
    exit(1);
  }
  for (i = 0; i < N; i++) {
    M_list[i] = (short *)malloc((t_max + 1) * sizeof(short));
    if (M_list[i] == NULL) {
      printf("ERROR: could not allocate sub-lists.");
      exit(1);
    }
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
    if (K2 > M) {
      mexErrMsgTxt("GENERATE: Can't have more than M weight 2 columns.");
      exit(1);
    }
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
    N_list[i] = (short *)malloc((tr_max + 1) * sizeof(short));
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
    fprintf(stderr, "%lld\r", i);
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
    free(N_list[i]);
  }
  free(N_list);

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

  sr = (double *)calloc(nzmax, sizeof(double));
  irs = (mwSize *)calloc(nzmax, sizeof(mwSize));
  jcs = (mwSize *)calloc(N + 1, sizeof(mwSize));
  if (sr == NULL || irs == NULL || jcs == NULL) {
    printf("ERROR: could not allocate sparse array.");
    exit(1);
  }

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

  /* Assign outputs */
  *sparseValues = sr;
  *sparseRows = irs;
  *sparseCols = jcs;
  *nzmaxOut = nzmax;

  /* Free */
  for (i = 0; i < N; i++) {
    free(M_list[i]);
  }
  free(M_list);

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

