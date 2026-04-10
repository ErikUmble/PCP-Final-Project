#include <stdint.h>
#include <stdio.h>

#define GRAPH_SIZE 64 * 512
#define GRAPH_VAR_BITSIZE 64
#define GRAPH_UINT64_SIZE (GRAPH_SIZE / 64)

typedef uint64_t graph_var_t;

graph_var_t graph[GRAPH_SIZE][GRAPH_UINT64_SIZE];
graph_var_t state[GRAPH_UINT64_SIZE];

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

uint8_t popcnt(uint64_t x) {
  // In CPU use popcnt instruction and in CUDA use __popcll()
#if defined(__CUDA_ARCH__)
  return __popcll(x);
#else
  return __builtin_popcountll(x);
#endif
}

uint64_t random() {
  static uint64_t counter = 1;
  counter++;
  return rnd64(counter);
}

int slow_but_correct_cut() {
  static uint8_t expanded_state[GRAPH_SIZE];
  // Expand state bits into an array of 0s and 1s
  for (int i = 0; i < GRAPH_UINT64_SIZE; i++) {
    for (int j = 0; j < GRAPH_VAR_BITSIZE; j++) {
      expanded_state[i * GRAPH_VAR_BITSIZE + j] = (state[i] >> j) & 1;
    }
  }

  static uint8_t expanded_graph[GRAPH_SIZE][GRAPH_SIZE];
  // Expand graph bits into a 2D array of 0s and 1s
  for (int i = 0; i < GRAPH_SIZE; i++) {
    for (int j = 0; j < GRAPH_SIZE; j++) {
      int bit_index = j % GRAPH_VAR_BITSIZE;
      int uint64_index = j / GRAPH_VAR_BITSIZE;
      expanded_graph[i][j] = (graph[i][uint64_index] >> bit_index) & 1;
    }
  }

  //
  // // Output the expanded state and graph
  // printf("Expanded state:\n");
  // for (int i = 0; i < GRAPH_SIZE; i++) {
  //   printf("%d", expanded_state[i]);
  // }
  // printf("\n");
  // printf("Expanded graph:\n");
  // for (int i = 0; i < GRAPH_SIZE; i++) {
  //   for (int j = 0; j < GRAPH_SIZE; j++) {
  //     printf("%d", expanded_graph[i][j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");
  // printf("Diagonal: ");
  // for (int i = 0; i < GRAPH_SIZE; i++) {
  //   printf("%d", expanded_graph[i][i]);
  // }
  // printf("\n");

  int cut = 0;
  for (int i = 0; i < GRAPH_SIZE; i++) {
    for (int j = 0; j < GRAPH_SIZE; j++) {
      if (expanded_state[i] == 0 && expanded_state[j] == 1) {
        cut += expanded_graph[i][j];
      }
    }
  }
  return cut;
}

int fast_cut() {
  uint32_t cut = 0;
  for (uint32_t i = 0; i < GRAPH_SIZE; i++) {
    uint32_t offset = i % GRAPH_VAR_BITSIZE;
    uint32_t byte = i / GRAPH_VAR_BITSIZE;
    uint32_t state_value = (state[byte] >> offset) & 1;
    if (state_value == 0) {
      for (uint32_t j = 0; j < GRAPH_UINT64_SIZE; j++) {
        graph_var_t computed = state[j] & graph[i][j];
        cut += popcnt(computed);
      }
    }
  }
  return cut;
}

int main() {
  // Fill graph and state with random bits
  for (int i = 0; i < GRAPH_SIZE; i++) {
    for (int j = 0; j < GRAPH_UINT64_SIZE; j++) {
      graph[i][j] = random();
    }
  }

  for (int i = 0; i < GRAPH_UINT64_SIZE; i++) {
    state[i] = random();
  }

  printf("Graph and state initialized.\n");
  // printf("State:\n");
  // for (int i = 0; i < GRAPH_UINT64_SIZE; i++) {
  //   printf("%016lx", state[i]);
  // }
  // printf("\n");
  // printf("Graph:\n");
  // for (int i = 0; i < GRAPH_SIZE; i++) {
  //   for (int j = 0; j < GRAPH_UINT64_SIZE; j++) {
  //     printf("%016lx ", graph[i][j]);
  //   }
  //   printf("\n");
  // }


  // Calculate max cut from graph and current state
  // int correct = slow_but_correct_cut();
  // printf("Correct cut: %d\n", correct);
  int fast;
  for (int i = 0; i < 50; i++) {
    fast = fast_cut();
  }
  printf("Fast cut: %d\n", fast);

  return 0;
}
