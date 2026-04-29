# PCP-Final-Project

## Building

We use a Makefile.

```bash
module load xl_r spectrum-mpi cuda
make max-cut
```

This will build the `./max-cut` executable.

## Running

You can run an experiment csv file using our batch script.

```bash
sbatch -N [nodes] -t 5 ./batch.sh ./runs/[input].csv ./results/[output].csv
# example
sbatch -N 1 -t 5 ./batch.sh ./runs/bigmaq.csv ./results/bigmaq.csv
```

Look at `runs/1node.csv` for an example of the file format.

Below is the csv header. Graph file should be relative to the project root directory (so like `graphs/[file]`)

```
nodes,ranks,seed,iterations,subiterations,graph_size,graph_file,communication_delay
```

> Note: Make sure to keep different node counts in different files!

## Getting results

Results are stored in `./results/[run-name].csv`. We also sometimes get the max cut over time by running this command.

```bash
cat logs/maxcut-seq-[jobid].out | grep "Iteration" | cut -d" " -f2,8 | tr ":" ","
```

Example output

```
0, 1338
2, 1340
4, 1343
6, 1344
8, 1346
10, 1349
12, 1367
14, 1384
16, 1390
...
```

## Specific runs

### Big Maq

This is a single node run with two graphs from bigmaq.

```bash
sbatch -N 1 -t 5 batch.sh ./runs/bigmaq.csv
```

The results will be stored in `./results/bigmaq.csv`

We also got the max cut over time using the command mentioned in the "Getting Results" section.

### Strong and weak scaling

Our strong and weak scaling runs are contained in the `1node.csv` `2node.csv` and `4node.csv` files. We ran them with the following commands.

```bash
sbatch -N 1 -t 13 batch.sh ./runs/1node.csv
sbatch -N 2 -t 5 batch.sh ./runs/2node.csv
sbatch -N 4 -t 3 batch.sh ./runs/4node.csv
```

The results are in the respective result folders.

### G1

We ran a single long test on the G1 graph.

```bash
sbatch -N 8 -t 20 batch.sh ./runs/gset-go-for-it.csv
```

The result is in the results folder and we used the command mentioned in the
"Getting Results" section to get the max cut over time. (We also committed our
`logs/[jobid].out` file as `results/gset-go-for-it.out`).

### Gset

We ran a couple of gset tests which are contained in the `runs/gset-benchmark.csv`. We were not able to run all of them due to the length of time and some were run in a seperate file `runs/gset-benchmark2.csv`.

```bash
# Only got through the 2000 node cases
sbatch -N 1 -t 45 batch.sh ./runs/gset-benchmark.csv
# Completed one 5000 node case
sbatch -N 1 -t 120 batch.sh ./runs/gset-benchmark2.csv
```

