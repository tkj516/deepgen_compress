#if 0
#include <stdio.h>
#include <stdlib.h>
#include "demo.h"

#define NUM 10
int main() {
  float *arr = (float *)malloc(sizeof(float) * NUM);
  int i;

  for (i = 0; i < NUM; i++) {
    arr[i] = i;
  }

  scale(arr, NUM, 0.5);

  for(i=0;i<NUM;i++){
    printf("arr[%02d] = %f\n", i, arr[i]);
  }
  return 0;
}

#endif

void scale(float* in, int N, float multiple) {
  for (int i = 0; i < N; i++) {
    in[i] *= multiple;
  }
}

