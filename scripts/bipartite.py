# Prints a random adjacency matrix for a bipartite graph with n vertices
# and m edges. Last line prints the max cut.

import random
import sys

def random_bipartite_graph(n, m):
    left = list(range(n // 2))
    right = list(range(n // 2, n))

    max_possible = len(left) * len(right)
    if m > max_possible:
        raise ValueError(f"Too many edges: max is {max_possible} for n={n}")

    all_edges = [(i, j) for i in left for j in right]
    chosen = random.sample(all_edges, m)

    graph = [[0 for _ in range(n)] for _ in range(n)]
    for i, j in chosen:
        graph[i][j] = 1
        graph[j][i] = 1

    return graph, m  # m == max cut

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python bipartite.py [seed] [n] [m]")
        sys.exit(1)

    seed = int(sys.argv[1])
    n = int(sys.argv[2])
    m = int(sys.argv[3])

    random.seed(seed)
    graph, max_cut = random_bipartite_graph(n, m)

    for row in graph:
        print("".join(str(x) for x in row))

