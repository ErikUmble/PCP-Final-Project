import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <input_file>")
    sys.exit(1)

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

first_line = lines[0].strip().split()
n = int(first_line[0])
m = int(first_line[1])

matrix = [[0 for _ in range(n)] for _ in range(n)]

for line in lines[1:]:
    parts = line.strip().split()
    h = int(parts[0]) - 1
    t = int(parts[1]) - 1
    c = int(parts[2])
    matrix[h][t] = c
    matrix[t][h] = c

for row in matrix:
    print(''.join(map(str, row)))