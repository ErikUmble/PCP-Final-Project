#!/bin/bash
#SBATCH --job-name=kernel_2N
#SBATCH --partition=el8-rpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --output=2N_results.txt

run_test() {
    ranks=$1
    size=$2
    file=$3

    start=$(date +%s.%N)
    output=$(mpirun --bind-to core -np $ranks ./basic-kernel 1 400 1 $size $file | tail -n 1)
    end=$(date +%s.%N)

    runtime=$(echo "$end - $start" | bc)
    max_cut=$(echo "$output" | awk '{print $4}')

    printf "%d & %.3f & %s \\\\\n" "$ranks" "$runtime" "$max_cut"
}

# replace with files we want to use
run_test 8 100 "../data/g05_100.0.matrix.xxxx"

# replace with files we want to use
run_test 8 400 "../data/g05_400.0.matrix.XXXX"