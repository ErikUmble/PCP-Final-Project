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
void cudaLandInit(int gpu, uint64_t seed);
void cudaLandFastCut(int gpu, int subiterations, uint32_t best_cut, uint32_t graph_bit_size, graph_var_t *graph, graph_var_t *out_states, float_t *qstate, uint32_t *max_cuts);
void *cudaLandMalloc(size_t size);
void cudaLandFree(void *ptr);
}

curandState *d_rand_state;

// Main kernel to update towards best state and compute cut cost for each thread
template <uint32_t graph_bit_size>
__global__ void fast_cut(int iterations, uint32_t best_cut, graph_var_t *graph, graph_var_t *out_states, curandState *rand_state, float_t *qstate, uint32_t *max_cuts) {
  // Get our portion of state
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t graph_int_size = graph_bit_size / GRAPH_VAR_BITSIZE;
  uint32_t our_best_cut = 0;
  graph_var_t *local_out_state = out_states + idx * graph_int_size;
  graph_var_t new_state[graph_int_size];

  for (int i = 0; i < iterations; i++) {
    // "Observe" our qstate

    for (int i = 0; i < graph_bit_size; i++) {
      uint32_t offset = i % GRAPH_VAR_BITSIZE;
      uint32_t byte = i / GRAPH_VAR_BITSIZE;
      float rand = curand_uniform(rand_state + idx);
      if (rand < qstate[i]) {
        new_state[byte] |= (1ULL << offset); // Set bit
      } else {
        new_state[byte] &= ~(1ULL << offset); // Clear bit
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
        local_out_state[i] = new_state[i];
      }
    }
  }

  max_cuts[idx] = our_best_cut;
}

// A kernel to initialize the random states for each thread (curand_init must be called from a kernel)
__global__ void setup_kernel(uint64_t seed, curandState *state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}

void cudaLandInit(int gpu, uint64_t seed) {
  cudaSetDevice(gpu);
  // Initialize random states for each thread
  cudaMallocManaged(&d_rand_state, NUM_THREADS * sizeof(curandState));
  setup_kernel<<<NUM_THREADS / 256, 256>>>(seed, d_rand_state);
}

void cudaLandFastCut(int gpu, int subiterations, uint32_t best_cut, uint32_t graph_bit_size, graph_var_t *graph, graph_var_t *out_states, float_t *qstate, uint32_t *max_cuts) {
  cudaSetDevice(gpu);

  // Find cut costs
  // Javascript to generate the below
  /* s = ""
  for (let i = 6; i < 15; i += 0.5) {
    size=Math.round(2**i)
    size=Math.ceil(size/64)*64
    s+=(`    case ${size}:\n`)
    s+=(`      fast_cut<${size}><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);\n`)
    s+=(`      break;\n`)
  }
  console.log(s)
  */

  switch (graph_bit_size) {
    case 64:
      fast_cut<64><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 128:
      fast_cut<128><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 192:
      fast_cut<192><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 256:
      fast_cut<256><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 384:
      fast_cut<384><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 512:
      fast_cut<512><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 768:
      fast_cut<768><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 1024:
      fast_cut<1024><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 1472:
      fast_cut<1472><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 2048:
      fast_cut<2048><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 2944:
      fast_cut<2944><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 4096:
      fast_cut<4096><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 5824:
      fast_cut<5824><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 8192:
      fast_cut<8192><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 11648:
      fast_cut<11648><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 16384:
      fast_cut<16384><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    case 23232:
      fast_cut<23232><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
    //G1
    case 832:
      fast_cut<832><<<NUM_THREADS / 256, 256>>>(subiterations, best_cut, graph, out_states, d_rand_state, qstate, max_cuts);
      break;
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
