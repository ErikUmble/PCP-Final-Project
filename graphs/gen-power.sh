#!/bin/bash

# Generates the random graphs deterministiclly

SEED=0

pushd $(dirname $0)

for s in 8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5 12.0 12.5 13.0 13.5 14.0; do
  v=$(python3 -c "print(round(2**$s))");
  echo "Generating power$s.$v.txt";
  python3 ../scripts/random-graph.py $SEED $v > power$s.$v.txt;
done

