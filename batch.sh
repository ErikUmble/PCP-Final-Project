#!/bin/bash
#SBATCH --job-name=maxcut-seq
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch -N <max_nodes> $0 <runs.csv> <results.csv>"
    exit 1
fi

CSV="$1"
OUTCSV="$2"

mkdir -p logs

# absolute path to executable
EXEC=$(realpath ../max-cut)

# header
header=$(head -n 1 "$CSV")
echo "${header},time_result" > "$OUTCSV"

tail -n +2 "$CSV" | while IFS=, read -r \
    nodes ranks seed iterations subiterations graph_size graph_file communication_delay
do
    echo "Running: nodes=$nodes ranks=$ranks seed=$seed"

    logfile=$(mktemp)

    # subset host list from allocation
    scontrol show hostnames "$SLURM_NODELIST" | head -n "$nodes" > hostfile

    mpirun \
        -np "$ranks" \
        --hostfile hostfile \
        "$EXEC" \
        "$seed" \
        "$iterations" \
        "$subiterations" \
        "$graph_size" \
        "$graph_file" \
        "$communication_delay" \
        | tee "$logfile"

    time_result=$(grep "Time:" "$logfile" | tail -n1 | sed -E 's/.*Time:[[:space:]]*//')

    rm -f "$logfile" hostfile

    echo "${nodes},${ranks},${seed},${iterations},${subiterations},${graph_size},${graph_file},${communication_delay},${time_result}" >> "$OUTCSV"
done

