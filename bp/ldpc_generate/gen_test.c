#include <stdio.h>
#include "ldpc_generate1.h"

int main() {
  mwSize M = 512;
  mwSize N = 1024;
  double t = 3.0;
  mwIndex q = 2;
  int seed = 123;

  double *values;
  mwSize *rows;
  mwSize *cols;
  mwSize nzmax;

  ldpc_generate(M, N, t, q, seed, &values, &rows, &cols, &nzmax);

  printf("nzmax = %lld;\n", nzmax);

  // VALUES
  printf("values = [");
  for (int i = 0; i < nzmax; i++) {
    printf("%g, ", values[i]);
  }
  printf("];\n");

  // ROWS
  printf("rows = [");
  for (int i = 0; i < nzmax; i++) {
    printf("%lld, ", rows[i]);
  }
  printf("];\n");

  // COLS
  printf("cols = [");
  for (int i = 0; i < N + 1; i++) {
    printf("%lld, ", cols[i]);
  }
  printf("];\n");

  return 0;
}

