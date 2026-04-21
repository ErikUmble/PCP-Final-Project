# PCP-Final-Project

## Building

`make max-cut`

## Running

`sbatch -N [nodes] -t 5 --partition=el8-rpi ./batch.sh ./runs/[input].csv ./results/[output].csv`

Example: `sbatch -N 1 -t 5 --partition=el8-rpi ./batch.sh ./runs/1node.csv ./results/1node.csv`

Look at `runs/1node.csv` for an example of the file format.

Below is the csv header. Graph file should be relative to the project root directory (so like `graphs/[file]`)

```
nodes,ranks,seed,iterations,subiterations,graph_size,graph_file,communication_delay
```

> Note: Make sure to keep different node counts in different files!

