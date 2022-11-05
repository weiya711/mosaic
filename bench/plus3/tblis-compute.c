#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <immintrin.h>
#include "tblis.h"
#if _OPENMP
#include <omp.h>
#endif
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;
typedef struct {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  int32_t      csize;         // component size
  int32_t*     mode_ordering; // mode storage ordering
  taco_mode_t* mode_types;    // mode storage types
  uint8_t***   indices;       // tensor index data (per mode)
  uint8_t*     vals;          // tensor values
  uint8_t*     fill_value;    // tensor fill value
  int32_t      vals_size;     // values array size
} taco_tensor_t;
#endif
#if !_OPENMP
int omp_get_thread_num() { return 0; }
int omp_get_max_threads() { return 1; }
#endif
int cmp(const void *a, const void *b) {
  return *((const int*)a) - *((const int*)b);
}
int taco_gallop(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayStart] >= target || arrayStart >= arrayEnd) {
    return arrayStart;
  }
  int step = 1;
  int curr = arrayStart;
  while (curr + step < arrayEnd && array[curr + step] < target) {
    curr += step;
    step = step * 2;
  }

  step = step / 2;
  while (step > 0) {
    if (curr + step < arrayEnd && array[curr + step] < target) {
      curr += step;
    }
    step = step / 2;
  }
  return curr+1;
}
int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayStart] >= target) {
    return arrayStart;
  }
  int lowerBound = arrayStart; // always < target
  int upperBound = arrayEnd; // always >= target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return upperBound;
}
int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayEnd] <= target) {
    return arrayEnd;
  }
  int lowerBound = arrayStart; // always <= target
  int upperBound = arrayEnd; // always > target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return lowerBound;
}
taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,
                                  int32_t* dimensions, int32_t* mode_ordering,
                                  taco_mode_t* mode_types) {
  taco_tensor_t* t = (taco_tensor_t *) malloc(sizeof(taco_tensor_t));
  t->order         = order;
  t->dimensions    = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_ordering = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_types    = (taco_mode_t *) malloc(order * sizeof(taco_mode_t));
  t->indices       = (uint8_t ***) malloc(order * sizeof(uint8_t***));
  t->csize         = csize;
  for (int32_t i = 0; i < order; i++) {
    t->dimensions[i]    = dimensions[i];
    t->mode_ordering[i] = mode_ordering[i];
    t->mode_types[i]    = mode_types[i];
    switch (t->mode_types[i]) {
      case taco_mode_dense:
        t->indices[i] = (uint8_t **) malloc(1 * sizeof(uint8_t **));
        break;
      case taco_mode_sparse:
        t->indices[i] = (uint8_t **) malloc(2 * sizeof(uint8_t **));
        break;
    }
  }
  return t;
}
void deinit_taco_tensor_t(taco_tensor_t* t) {
  for (int i = 0; i < t->order; i++) {
    free(t->indices[i]);
  }
  free(t->indices);
  free(t->dimensions);
  free(t->mode_ordering);
  free(t->mode_types);
  free(t);
}
  float tblis_vector_dot_transfer(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_vector* A, const tblis_vector* B,
                      tblis_scalar* result){
   tblis_vector_dot(comm, cfg, A, B, result);
   return result->data.s; }

  void tblis_init_tensor_s_helper_row_major(tblis_tensor * t, int * dim, int num_dim, void * data){
    len_type * len = malloc(sizeof(len_type)*num_dim);
        if (!len){
          printf("error, len not valid!!!");
        }
    int stride_product = 1; 
    for (int i = 0; i < num_dim; i++){
        len[(len_type) i] = dim[i];
        stride_product *= dim[i];
    }
    stride_type * stride = malloc(sizeof(stride_type)*num_dim);
        if (!stride){
          printf("error, stride not valid!!!");
        }
   for (int i = 0; i < num_dim; i++){
        stride_product /= dim[i];
        stride[(stride_type) i] = stride_product;
    }
    tblis_init_tensor_s(t, num_dim, len, data, stride);
}
 void free_tblis_tensor(tblis_tensor * t){
     free(t->len);   free(t->stride);}
#endif

int assemble(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C) {

  float*  A_vals = (float*)(A->vals);

  A_vals = (float*)malloc(sizeof(float) * 8);

  A_vals = (float*)malloc(sizeof(float) * 8);

  A->vals = (uint8_t*)A_vals;
  return 0;
}

int compute(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C) {

  float*  A_vals = (float*)(A->vals);
  float*  B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int C3_dimension = (int)(C->dimensions[2]);
  float*  C_vals = (float*)(C->vals);

  for (int32_t i2436 = 0; i2436 < C1_dimension; i2436++) {
    for (int32_t i2437 = 0; i2437 < C2_dimension; i2437++) {
      int32_t i2437A = i2436 * 2 + i2437;
      int32_t i2437C = i2436 * 2 + i2437;
      for (int32_t i2438 = 0; i2438 < C3_dimension; i2438++) {
        int32_t i2438A = i2437A * 2 + i2438;
        int32_t i2438C = i2437C * 2 + i2438;
        A_vals[i2438A] = C_vals[i2438C];
      }
    }
  }
tblis_tensor var1;
tblis_tensor var2;
tblis_init_tensor_s_helper_row_major(&var1, B->dimensions, 3, A_vals);
tblis_init_tensor_s_helper_row_major(&var2, B->dimensions, 3, B_vals);
tblis_tensor_add(NULL, NULL, &var2, "ijk", &var1, "ijk");
  return 0;
}
#include "/home/manya227/temp/taco_tmp_2LwlCt/08knenw0t4as.h"
int _shim_assemble(void** parameterPack) {
  return assemble((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]));
}
int _shim_compute(void** parameterPack) {
  return compute((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]));
}
