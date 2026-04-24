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
void cudaLandFastCut(int subiterations, uint32_t best_cut, uint32_t graph_bit_size, graph_var_t *graph, graph_var_t *state, graph_var_t *d_best_state, uint32_t *d_result);
void *cudaLandMalloc(size_t size);
void cudaLandFree(void *ptr);
}

curandState *d_rand_state;


// Main kernel to update towards best state and compute cut cost for each thread
template <uint32_t graph_bit_size>
__global__ void fast_cut(int iterations, uint32_t best_cut, graph_var_t *graph, graph_var_t *states, curandState *rand_state, graph_var_t *best_state, uint32_t *result) {
  // Get our portion of state
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t graph_int_size = graph_bit_size / GRAPH_VAR_BITSIZE;
  uint32_t our_best_cut = best_cut;
  graph_var_t *local_state = states + idx * graph_int_size;
  graph_var_t new_state[graph_int_size];

  // Copy local_state into new_state

  for (int i = 0; i < graph_int_size; i++) {
    new_state[i] = local_state[i];
  }

  for (int i = 0; i < iterations; i++) {
    // Randomly move towards/away from best state
    for (uint32_t bit = 0; bit < graph_bit_size; bit++) {
      float rand = curand_uniform(rand_state + idx);
      // 40% chance of moving towards best state
      // 30% chance of flipping
      // 30% chance of staying the same
      uint32_t byte = bit / GRAPH_VAR_BITSIZE;
      uint32_t offset = bit % GRAPH_VAR_BITSIZE;
      if (rand < 0.4f) {
        // Move towards best state
        uint32_t best_bit = (best_state[byte] >> offset) & 1;
        new_state[byte] &= ~(1ULL << offset); // Clear bit
        new_state[byte] |= (best_bit << offset); // Set to best bit
      } else if (rand < 0.70f) {
        // Flip bit
        new_state[byte] ^= (1ULL << offset);
      }
    }

    // Calculate cut

    uint32_t cut = 0;
    for (uint32_t i = 0; i < graph_bit_size; i++) {
      uint32_t offset = i % GRAPH_VAR_BITSIZE;
      uint32_t byte = i / GRAPH_VAR_BITSIZE;
      uint32_t state_value = (new_state[byte] >> offset) & 1;
      if (state_value == 0) {
	for (uint32_t j = 0; j < graph_int_size; j++) {
	  graph_var_t computed = new_state[j] & graph[i * graph_int_size + j];
	  cut += __popcll(computed);
	}
      }
    }

    if (cut > our_best_cut) {
      our_best_cut = cut;
      for (int i = 0; i < graph_int_size; i++) {
	local_state[i] = new_state[i];
      }
    }
  }

  result[idx] = our_best_cut;
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

void cudaLandFastCut(int subiterations, uint32_t best_cut, uint32_t graph_bit_size, graph_var_t *graph, graph_var_t *state, graph_var_t *d_best_state, uint32_t *d_result) {
  // Find cut costs
  // Javascript to generate the below
  /* s = ""
  for (let i = 6; i < 15; i += 0.5) {
    size=Math.round(2**i)
    s+=(`    case ${size}:\n`)
    s+=(`      fast_cut<${size}><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);\n`)
    s+=(`      break;\n`)
  }
  console.log(s)
  */

  switch (graph_bit_size) {
    case 64:
      fast_cut<64><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 91:
      fast_cut<91><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 128:
      fast_cut<128><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 181:
      fast_cut<181><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 256:
      fast_cut<256><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 362:
      fast_cut<362><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 512:
      fast_cut<512><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 724:
      fast_cut<724><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 1024:
      fast_cut<1024><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 1448:
      fast_cut<1448><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 2048:
      fast_cut<2048><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 2896:
      fast_cut<2896><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 4096:
      fast_cut<4096><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 5793:
      fast_cut<5793><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 8192:
      fast_cut<8192><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 11585:
      fast_cut<11585><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 16384:
      fast_cut<16384><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
      break;
    case 23170:
      break;
      fast_cut<23170><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, state, d_rand_state, d_best_state, d_result);
    default:
      printf("Error unsupported graph bit size: %d", graph_bit_size);
      exit(1);
  }
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
