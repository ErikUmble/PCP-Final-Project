#!/bin/bash
#SBATCH --job-name=kernel_1N
#SBATCH --partition=el8-rpi
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --output=1N_results.txt

echo "Compiling sources..."
nvcc basic-kernel.cu -o basic-kernel
nvcc kernel-check.c -o kernel-check

echo "Running kernel-check sanity test..."
./kernel-check
echo ""

# Function to run the kernel, measure time, and output LaTeX table format
run_test() {
    ranks=$1
    size=$2
    file=$3

    # Capture start time, run program, capture end time
    start=$(date +%s.%N)
    output=$(mpirun --bind-to core -np $ranks ./basic-kernel 1 400 1 $size $file | tail -n 1)
    end=$(date +%s.%N)

    # Calculate elapsed time
    runtime=$(echo "$end - $start" | bc)

    # Extract the actual max cut number from the string "Max cut = 1415 (Thread 1963)"
    max_cut=$(echo "$output" | awk '{print $4}')

    # Output format: Ranks & Time & MaxCut \\
    printf "%d & %.3f & %s \\\\\n" "$ranks" "$runtime" "$max_cut"
}

# replace with files we want to use (strong scaling)
run_test 2 100 "../data/g05_100.0.matrix.xxxx"
run_test 4 100 "../data/g05_100.0.matrix.xxxx"

# replace with files we want to use (weak scaling)
run_test 2 100 "../data/g05_100.0.matrix.xxxx"
run_test 4 200 "../data/g05_200.0.matrix.xxxx"