#include <stdint.h>
#include <cmath>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// #define GRAPH_SIZE 64 * 128
#define GRAPH_VAR_BITSIZE 64
// #define GRAPH_UINT64_SIZE (GRAPH_SIZE / 64)
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

uint64_t rng_state = 1;

uint64_t randomu64() {
  rng_state++;
  return rnd64(rng_state);
}

// Main kernel to update towards best state and compute cut cost for each thread
// template <uint32_t graph_bit_size>
__global__ void fast_cut(int iterations, uint32_t graph_bit_size, graph_var_t *graph, graph_var_t *states, curandState *rand_state, graph_var_t *best_state, uint32_t *result // replace result with unsigned long long *global_max ) {
  // Get our portion of state
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  graph_var_t *local_state = states + idx * (GRAPH_VAR_BITSIZE / 8);

  for (int i = 0; i < iterations; i++) {
    // Randomly move towards/away from best state
    for (uint32_t bit = 0; bit < graph_bit_size; bit++) {
      float rand = curand_uniform(rand_state + idx);
      // 80% chance of moving towards best state
      // 5% chance of becoming 1
      // 5% chance of becoming 0
      // 10% chance of staying the same
      uint32_t byte = bit / GRAPH_VAR_BITSIZE;
      uint32_t offset = bit % GRAPH_VAR_BITSIZE;
      if (rand < 0.8f) {
        // Move towards best state
        uint32_t best_bit = (best_state[byte] >> offset) & 1;
        if (best_bit == 1) {
          local_state[byte] |= (1ULL << offset);
        } else {
          local_state[byte] &= ~(1ULL << offset);
        }
      } else if (rand < 0.85f) {
        // Become 1
        local_state[byte] |= (1ULL << offset);
      } else if (rand < 0.9f) {
        // Become 0
        local_state[byte] &= ~(1ULL << offset);
      }
    }
  }

  // Sync threads
  // possible change 1:
  //     is there a need for this considering that
  //     the treads each operate on the local state and the best
  //     state is only read upon initialization we may be able to
  //     get a bit of read/calc overlap if it is removed and since
  //     there are not read dependencies it should be safe
  __syncthreads();

  uint32_t graph_int_size = graph_bit_size / GRAPH_VAR_BITSIZE;

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

  // (possible change 2) Pack the 32-bit cut and 32-bit thread ID (idx) into a single 64-bit variable
  // (possible change 2) unsigned long long combined = ((unsigned long long)cut << 32) | (uint32_t)idx;
  // (possible change 2) atomicMax(global_max, combined);
  result[idx] = cut;
}

// A kernel to initialize the random states for each thread (curand_init must be called from a kernel)
__global__ void setup_kernel(uint64_t seed, curandState *state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}

int main(int argc, char *argv[]) {
  // Get arguments (iterations, subiterations)
  if (argc != 6) {
    printf("Usage: %s <seed> <iterations> <subiterations> <graph_size> <graph_file>\n", argv[0]);
    return 1;
  }

  int iterations;
  int subiterations;
  int graph_org_bit_size;
  char* graph_file;

  rng_state = atoi(argv[1]);
  rng_state = randomu64();
  iterations = atoi(argv[2]);
  subiterations = atoi(argv[3]);
  graph_org_bit_size = atoi(argv[4]);
  graph_file = argv[5];

  int graph_bit_size = ceil(((float)graph_org_bit_size / GRAPH_VAR_BITSIZE)) * GRAPH_VAR_BITSIZE;

  int graph_int_size = graph_bit_size / GRAPH_VAR_BITSIZE;

  graph_var_t (*graph);
  graph_var_t (*state);
  cudaMallocManaged(&graph, graph_bit_size * graph_int_size * sizeof(graph_var_t));
  cudaMallocManaged(&state, NUM_THREADS * graph_int_size * sizeof(graph_var_t));
  memset(graph, 0, graph_bit_size * graph_int_size * sizeof(graph_var_t));

  // Read in graph file (adjaceny matrix ascii 0s and 1s)
  FILE *fp = fopen(graph_file, "r");
  if (fp == NULL) {
    printf("Error opening graph file: %s\n", graph_file);
    return 1;
  }

  for (int i = 0; i < graph_org_bit_size; i++) {
    for (int j = 0; j < graph_org_bit_size; j++) {
      char c = fgetc(fp);
      if (c == '1') {
        // Set bit in graph
        int byte = j / GRAPH_VAR_BITSIZE;
        int offset = j % GRAPH_VAR_BITSIZE;
        graph[i * graph_int_size + byte] |= (1ULL << offset);
      }
    }
    // Consume newline
    fgetc(fp);
  }

  // Initialize the state randomly
  for (int i = 0; i < NUM_THREADS; i++) {
     for (int j = 0; j < graph_int_size; j++) {
      state[i * graph_int_size + j] = randomu64();
    }
  }

  // Initialize random states for each thread
  curandState *d_rand_state;
  cudaMallocManaged(&d_rand_state, NUM_THREADS * sizeof(curandState));
  setup_kernel<<<NUM_THREADS / 256, 256>>>(randomu64(), d_rand_state);

  graph_var_t *d_best_state;
  cudaMallocManaged(&d_best_state, graph_int_size * sizeof(graph_var_t));

  // Set best state to thread 0

  for (int i = 0; i < graph_int_size; i++) {
    d_best_state[i] = state[i];
  }

  printf("Graph and state initialized.\n");

  printf("Running kernel...\n");
  uint32_t *d_result;
  cudaMallocManaged(&d_result, NUM_THREADS * sizeof(uint32_t));

  // (possible change 2) global maximum bit
  // (possible change 2) unsigned long long *d_global_max;
  // (possible change 2) cudaMallocManaged(&d_global_max, sizeof(unsigned long long));

  uint32_t max_cut = 0;
  uint32_t max_thread = 0;
  for (int i = 0; i < iterations; i++) {
    // (possible change 2) result best thread
    // (possible change 2) *d_global_max = 0;

    // Find cut costs
    fast_cut<<<NUM_THREADS / 256, 256>>>(subiterations, graph_bit_size, graph, state, d_rand_state, d_best_state, d_result);
    cudaDeviceSynchronize();

    // Find max
    // possible change 2:
    //     should we try to implement parallel reduction to find max value of thread on gpu
    //     https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=atomicMax#atomicmax
    //     we would need to change the final parameter from uint32_t *result to unsigned long long *global_max instead of d_result
    //     to then use atomic max to find best thread. note if we scale the number of threads up we might have to deal with atomic
    //     collisions and need to either import CUB ArgMax or similar
    //     https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html#_CPPv4I00EN3cub12DeviceReduce6ArgMaxE11cudaError_tPvR6size_t14InputIteratorT15OutputIteratorTi12cudaStream_t
    // (possible change 2) max_cut = (uint32_t)(*d_global_max >> 32); // shift right 32
    // (possible change 2) max_thread = (uint32_t)(*d_global_max & 0xFFFFFFFF); // get thread id
    max_cut = 0;
    max_thread = 0;
    for (int j = 0; j < NUM_THREADS; j++) {
      if (d_result[j] > max_cut) {
        max_cut = d_result[j];
        max_thread = j;
      }
    }

    // Update best state
    for (int j = 0; j < graph_int_size; j++) {
      d_best_state[j] = state[max_thread * (GRAPH_VAR_BITSIZE / 8) + j];
    }

    if (i % (iterations/50) == 0) {
      printf("Iteration %d: Max cut = %lu (Thread %d)\n", i, max_cut, max_thread);
    }
  }

  printf("Max cut = %lu (Thread %d)\n", max_cut, max_thread);

  return 0;
}
