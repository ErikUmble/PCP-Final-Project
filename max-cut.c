#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<math.h>
#include<string.h>
#include<stdint.h>
#include<mpi.h>

#include "max-cut.h"

extern void cudaLandFastCut(int gpu, int subiterations, uint32_t best_cut, uint32_t graph_bit_size, graph_var_t *graph, graph_var_t *out_states, float_t *qstate, uint32_t *max_cuts);
extern void cudaLandInit(int gpu, uint64_t seed);
extern void* cudaLandMalloc( size_t size );
extern void cudaLandFree( void *ptr );

typedef unsigned long long ticks;

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

#define SEND_TAG 0

// IBM POWER9 System clock with 512MHZ resolution.
static __inline__ ticks getticks(void)
{
  unsigned int tbl, tbu0, tbu1;

  do {
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
    __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
  } while (tbu0 != tbu1);

  return (((unsigned long long)tbu0) << 32) | tbl;
}

int main(int argc, char *argv[])
{
  int seed, iterations, subiterations, graph_org_bit_size, communication_delay;
  char *graph_file;
  graph_var_t *graph;
  graph_var_t *out_state;


  // initialize MPI
  MPI_Init(&argc, &argv);

  int npes, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
		                      MPI_INFO_NULL, &local_comm);

  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);

  MPI_Barrier(MPI_COMM_WORLD);

  if (argc != 7) {
    // rank 0 reports errors with the command line arguments
    if (rank == 0) {
      printf("Usage: %s <seed> <iterations> <subiterations> <graph_size> <graph_file> <communication_delay>\n", argv[0]);
    }
    exit(1);
  }

  ticks start = getticks();

  sscanf(argv[1], "%u", &seed);
  sscanf(argv[2], "%u", &iterations);
  sscanf(argv[3], "%u", &subiterations);
  sscanf(argv[4], "%u", &graph_org_bit_size);
  graph_file = argv[5];
  sscanf(argv[6], "%u", &communication_delay);

  int device = local_rank; // The local rank should range from 0 to 3

  // Use a different random state offset for each process to ensure unique random sequences across them
  rng_state = seed + rank;
  rng_state += randomu64();

  int graph_bit_size = ceil(((float)graph_org_bit_size / GRAPH_VAR_BITSIZE)) * GRAPH_VAR_BITSIZE;
  int graph_int_size = graph_bit_size / GRAPH_VAR_BITSIZE;

  graph = cudaLandMalloc(graph_bit_size * graph_int_size * sizeof(graph_var_t));
  out_state = cudaLandMalloc(NUM_THREADS * graph_int_size * sizeof(graph_var_t));
  memset(graph, 0, graph_bit_size * graph_int_size * sizeof(graph_var_t));

  // Root reads in graph file (adjaceny matrix ascii 0s and 1s)
  if (rank == 0) {
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
  }

  // Distribute graph to all processes
  MPI_Bcast(graph, graph_bit_size * graph_int_size, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  float_t *qstate;
  qstate = cudaLandMalloc(graph_bit_size * sizeof(float_t));
  // Initialize the state to 50% for all vertices
  for (int j = 0; j < graph_bit_size; j++) {
    qstate[j] = 0.5;
  }

  cudaLandInit(device, randomu64());

  uint32_t *max_cuts;
  max_cuts = cudaLandMalloc(NUM_THREADS * sizeof(uint32_t));

  uint32_t max_cut = 0;
  uint32_t max_thread = 0;
  struct {
      int val;
      int rank;
  } local_best, global_best;
  for (int i = 0; i < iterations; i++) {
    // Find cut costs
    cudaLandFastCut(device, subiterations, max_cut, graph_bit_size, graph, out_state, qstate, max_cuts);

    // Find max
    uint32_t max_k_cuts[5];
    uint32_t max_k_threads[5];
    uint32_t iteration_max_cut = 0;
    for (int j = 0; j < NUM_THREADS; j++) {
      if (max_cuts[j] > iteration_max_cut) {
        uint32_t min = 100000000;
        for (int k = 0; k < 5; k++) {
          if (max_k_cuts[k] < min) {
            min = max_k_cuts[k];
            max_k_cuts[k] = max_cuts[j];
            max_k_threads[k] = j;
          }
        }
      }
    }

    // Update local qstate with best from top k threads
    // If the thread has a 1 in the state make it more likely to be a 1, if it has a 0 make it more likely to be a 0
    for (int k = 0; k < 5; k++) {
      int thread = max_k_threads[k];
      for (int j = 0; j < graph_int_size; j++) {
        graph_var_t bits = out_state[thread * graph_int_size + j];
        for (int b = 0; b < GRAPH_VAR_BITSIZE; b++) {
          int idx = j * GRAPH_VAR_BITSIZE + b;
          if (idx >= graph_org_bit_size) {
            break;
          }
          if (bits & (1ULL << b)) {
            qstate[idx] += 0.01;
            if (qstate[idx] > 0.99) {
              qstate[idx] = 0.99;
            }
          } else {
            qstate[idx] -= 0.01;
            if (qstate[idx] < 0.01) {
              qstate[idx] = 0.01;
            }
          }
        }
      }
    }

    // every communication_delay iterations, broadcast the best state to all processes to synchronize the best state across
    if (i % communication_delay == 0) {
      local_best.val = (int) max_cut;
      local_best.rank = rank;
      MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

      max_cut = global_best.val;

      if (rank == 0) {
        printf("Rank %d: Broadcasting best state from rank %d with cut %llu\n", rank, global_best.rank, (unsigned long long) max_cut);
      }

      // broadcast the best state from the best rank to all processes
      MPI_Bcast(qstate, graph_bit_size, MPI_FLOAT_T, global_best.rank, MPI_COMM_WORLD);
    }

    if (i % (iterations/50) == 0 && rank == 0) {
      printf("Iteration %d: Best cut so far = %llu\n", i, (unsigned long long) max_cut);
    }
  }

  if (rank == 0) {
    uint64_t states_checked = npes * iterations * NUM_THREADS * subiterations;
    uint64_t per_iter = npes * NUM_THREADS * subiterations;
    uint64_t per_global = npes * NUM_THREADS * subiterations * communication_delay;
    printf("Final max cut: %llu\n", (unsigned long long) max_cut);
    printf("Total checked: %llu\n", (unsigned long long) states_checked);
    printf("Per iteration: %llu\n", (unsigned long long) per_iter);
    printf("Per global sync: %llu\n", (unsigned long long) per_global);
    printf("(Thread %d)\n", max_thread);
    printf("Time: %f\n", (double)(getticks() - start) / 512e6);
  }

  // Cleanup
  cudaLandFree(graph);
  cudaLandFree(state);
  cudaLandFree(qstate);
  cudaLandFree(max_cuts);

  MPI_Finalize();

  return 0;
}

