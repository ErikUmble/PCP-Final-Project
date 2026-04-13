#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define GRAPH_SIZE 64 * 512
#define GRAPH_VAR_BITSIZE 64
#define GRAPH_UINT64_SIZE (GRAPH_SIZE / 64)
#define NUM_THREADS 512

typedef uint64_t graph_var_t;

// graph_var_t graph[GRAPH_SIZE][GRAPH_UINT64_SIZE];
// graph_var_t states[NUM_THREADS][GRAPH_UINT64_SIZE];

// https://www.reddit.com/r/C_Programming/comments/ozew2u/comment/h7zijm8/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
uint64_t rnd64(uint64_t n)
{
    const uint64_t z = 0x9FB21C651E98DF25;

    n ^= ((n << 49) | (n >> 15)) ^ ((n << 24) | (n >> 40));
    n *= z;
    n ^= n >> 35;
    n *= z;
    n ^= n >> 28;

    return n;
}

uint64_t randomu64() {
  static uint64_t counter = 1;
  counter++;
  return rnd64(counter);
}

template <uint32_t graph_size>
__global__ void fast_cut(graph_var_t *graph, graph_var_t *state, uint32_t *result) {
  // get our portion of state
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  graph_var_t *local_state = state + idx * (GRAPH_VAR_BITSIZE / 8);

  uint32_t cut = 0;
  for (uint32_t i = 0; i < graph_size; i++) {
    uint32_t offset = i % GRAPH_VAR_BITSIZE;
    uint32_t byte = i / GRAPH_VAR_BITSIZE;
    uint32_t state_value = (local_state[byte] >> offset) & 1;
    if (state_value == 0) {
      for (uint32_t j = 0; j < GRAPH_UINT64_SIZE; j++) {
        graph_var_t computed = local_state[j] & graph[i * GRAPH_UINT64_SIZE + j];
        cut += __popcll(computed);
      }
    }
  }

  result[idx] = cut;
}

int main() {
  graph_var_t (*graph);
  graph_var_t (*state);
  cudaMallocManaged(&graph, GRAPH_SIZE * GRAPH_UINT64_SIZE * sizeof(graph_var_t));
  cudaMallocManaged(&state, NUM_THREADS * GRAPH_UINT64_SIZE * sizeof(graph_var_t));

  // Fill graph and state with random bits
  for (int i = 0; i < GRAPH_SIZE; i++) {
    for (int j = 0; j < GRAPH_UINT64_SIZE; j++) {
        graph[i * GRAPH_UINT64_SIZE + j] = randomu64();
    }
  }

  for (int i = 0; i < NUM_THREADS; i++) {
     for (int j = 0; j < GRAPH_UINT64_SIZE; j++) {
      state[i * GRAPH_UINT64_SIZE + j] = randomu64();
    }
  }

  printf("Graph and state initialized.\n");

  printf("Running kernel...\n");
  uint32_t *d_result;
  cudaMallocManaged(&d_result, NUM_THREADS * sizeof(uint32_t));

  fast_cut<GRAPH_SIZE><<<NUM_THREADS / 256, 256>>>(graph, state, d_result);
  cudaDeviceSynchronize();

  printf("Results:\n");
  for (int i = 0; i < NUM_THREADS; i++) {
    printf("Thread %d: Cut = %d\n", i, d_result[i]);
  }

  return 0;
}
