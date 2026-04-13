# Prints a random adjacency matrix for a graph with n vertices

import random
import sys

def random_graph(n):
    graph = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                graph[i][j] = 1
                graph[j][i] = 1
    return graph

# usage: python random-graph.py n

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python random-graph.py n")
        sys.exit(1)
    n = int(sys.argv[1])
    graph = random_graph(n)
    for row in graph:
        print("".join(str(x) for x in row))

