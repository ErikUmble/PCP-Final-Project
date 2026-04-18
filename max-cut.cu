#include <stdint.h>
#include <cmath>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "max-cut.h"

extern "C" 
{
void cudaLandInit(uint64_t seed);
void cudaLandFastCut(int subiterations, int graph_bit_size, graph_var_t *graph, graph_var_t *state, graph_var_t *d_best_state, uint32_t *d_result);
void *cudaLandMalloc(size_t size);
void cudaLandFree(void *ptr);
}

curandState *d_rand_state;


// Main kernel to update towards best state and compute cut cost for each thread
// template <uint32_t graph_bit_size>
__global__ void fast_cut(int iterations, uint32_t graph_bit_size, graph_var_t *graph, graph_var_t *states, curandState *rand_state, graph_var_t *best_state, uint32_t *result) {
  // Get our portion of state
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t graph_int_size = graph_bit_size / GRAPH_VAR_BITSIZE;
  graph_var_t *local_state = states + idx * graph_int_size;

  for (int i = 0; i < iterations; i++) {
    // Randomly move towards/away from best state
    for (uint32_t bit = 0; bit < graph_bit_size; bit++) {
      float rand = curand_uniform(rand_state + idx);
      // 50% chance of moving towards best state
      // 30% chance of flipping
      // 20% chance of staying the same
      uint32_t byte = bit / GRAPH_VAR_BITSIZE;
      uint32_t offset = bit % GRAPH_VAR_BITSIZE;
      if (rand < 0.5f) {
        // Move towards best state
        uint32_t best_bit = (best_state[byte] >> offset) & 1;
        local_state[byte] &= ~(1ULL << offset); // Clear bit
        local_state[byte] |= (best_bit << offset); // Set to best bit
      } else if (rand < 0.80f) {
        // Flip bit
        local_state[byte] ^= (1ULL << offset);
      }
    }
  }

  // Sync threads
  __syncthreads();

  uint32_t cut = 0;
  for (uint32_t i = 0; i < graph_bit_size; i++) {
    uint32_t offset = i % GRAPH_VAR_BITSIZE;
    uint32_t byte = i / GRAPH_VAR_BITSIZE;
    uint32_t state_value = (local_state[byte] >> offset) & 1;
    if (state_value == 0) {
      for (uint32_t j = 0; j < graph_int_size; j++) {
        graph_var_t computed = local_state[j] & graph[i * graph_int_size + j];
        cut += __popcll(computed);
      }
    }
  }

  result[idx] = cut;
}

// A kernel to initialize the random states for each thread (curand_init must be called from a kernel)
__global__ void setup_kernel(uint64_t seed, curandState *state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}

void cudaLandInit(uint64_t seed) {
  // Initialize random states for each thread
  cudaMallocManaged(&d_rand_state, NUM_THREADS * sizeof(curandState));
  setup_kernel<<<NUM_THREADS / 256, 256>>>(seed, d_rand_state);
}

void cudaLandFastCut(int subiterations, int graph_bit_size, graph_var_t *graph, graph_var_t *state, graph_var_t *d_best_state, uint32_t *d_result) {
  // Find cut costs
  fast_cut<<<NUM_THREADS / 256, 256>>>(subiterations, graph_bit_size, graph, state, d_rand_state, d_best_state, d_result);
  cudaDeviceSynchronize();
}

void* cudaLandMalloc( size_t size )
{
  void *ptr;

  cudaMallocManaged( &ptr, size );

  return ptr;
}

void cudaLandFree( void *ptr )
{
  cudaFree( ptr );
}
