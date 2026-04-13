#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define GRAPH_SIZE 64 * 4
#define GRAPH_VAR_BITSIZE 64
#define GRAPH_UINT64_SIZE (GRAPH_SIZE / 64)
#define NUM_THREADS 2048

typedef uint64_t graph_var_t;

// == RANDOM NUMBERS ==

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

// Main kernel to update towards best state and compute cut cost for each thread
template <uint32_t graph_size>
__global__ void fast_cut(int iterations, graph_var_t *graph, graph_var_t *states, curandState *rand_state, graph_var_t *best_state, uint32_t *result) {
  // Get our portion of state
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  graph_var_t *local_state = states + idx * (GRAPH_VAR_BITSIZE / 8);

  for (int i = 0; i < iterations; i++) {
    // Randomly move towards/away from best state
    for (uint32_t bit = 0; bit < GRAPH_SIZE; bit++) {
      float rand = curand_uniform(rand_state + idx);
      // 60% chance of moving towards best state
      // 10% chance of becoming 1
      // 10% chance of becoming 0
      // 20% chance of staying the same
      uint32_t byte = bit / GRAPH_VAR_BITSIZE;
      uint32_t offset = bit % GRAPH_VAR_BITSIZE;
      if (rand < 0.6f) {
        // Move towards best state
        uint32_t best_bit = (best_state[byte] >> offset) & 1;
        if (best_bit == 1) {
          local_state[byte] |= (1ULL << offset);
        } else {
          local_state[byte] &= ~(1ULL << offset);
        }
      } else if (rand < 0.7f) {
        // Become 1
        local_state[byte] |= (1ULL << offset);
      } else if (rand < 0.8f) {
        // Become 0
        local_state[byte] &= ~(1ULL << offset);
      }
    }
  }

  // Sync threads
  __syncthreads();

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

// A kernel to initialize the random states for each thread (curand_init must be called from a kernel)
__global__ void setup_kernel(uint64_t seed, curandState *state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}

int main(int argc, char *argv[]) {
  // Get arguments (iterations, subiterations)
  if (argc != 3) {
    printf("Usage: %s <iterations> <subiterations>\n", argv[0]);
    return 1;
  }

  int iterations;
  int subiterations;

  iterations = atoi(argv[1]);
  subiterations = atoi(argv[2]);

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

  // Initialize random states for each thread
  curandState *d_rand_state;
  cudaMallocManaged(&d_rand_state, NUM_THREADS * sizeof(curandState));
  setup_kernel<<<NUM_THREADS / 256, 256>>>(randomu64(), d_rand_state);

  graph_var_t *d_best_state;
  cudaMallocManaged(&d_best_state, GRAPH_UINT64_SIZE * sizeof(graph_var_t));

  printf("Graph and state initialized.\n");

  printf("Running kernel...\n");
  uint32_t *d_result;
  cudaMallocManaged(&d_result, NUM_THREADS * sizeof(uint32_t));

  uint32_t max_cut = 0;
  uint32_t max_thread = 0;
  for (int i = 0; i < iterations; i++) {
    // Find cut costs
    fast_cut<GRAPH_SIZE><<<NUM_THREADS / 256, 256>>>(subiterations, graph, state, d_rand_state, d_best_state, d_result);
    cudaDeviceSynchronize();

    // Find max
    max_cut = 0;
    max_thread = 0;
    for (int j = 0; j < NUM_THREADS; j++) {
      if (d_result[j] > max_cut) {
        max_cut = d_result[j];
        max_thread = j;
      }
    }

    // Update best state
    for (int j = 0; j < GRAPH_UINT64_SIZE; j++) {
      d_best_state[j] = state[max_thread * (GRAPH_VAR_BITSIZE / 8) + j];
    }

    if (i % 200 == 0) {
      printf("Iteration %d: Max cut = %d (Thread %d)\n", i, max_cut, max_thread);
    }
  }

  printf("Max cut = %d (Thread %d)\n", max_cut, max_thread);

  return 0;
}
