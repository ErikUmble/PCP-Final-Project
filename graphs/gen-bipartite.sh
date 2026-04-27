#!/bin/bash

# Generates the random graphs deterministiclly

SEED=0

pushd $(dirname $0)

for s in 8 9 10; do
    for e in 256 512 1024 2048 4096; do
      v=$(python3 -c "print(round(2**$s))");
      echo "Generating bipartite$s.$v.$e.txt";
      python3 ../scripts/bipartite.py $SEED $v $e > bipartite$s.$v.$e.txt
    done
done

