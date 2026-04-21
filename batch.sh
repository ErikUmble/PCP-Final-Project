#!/bin/bash
#SBATCH --job-name=maxcut-seq
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:4

set -euo pipefail
set -x

if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch -N <max_nodes> $0 <runs.csv> <results.csv>"
    exit 1
fi

module load xl_r spectrum-mpi cuda

CSV="$1"
OUTCSV="$2"

mkdir -p logs

# absolute path to executable
EXEC=$(realpath ./max-cut)

{
    # header
    read header
    echo "${header},time_result,cut_result" > "$OUTCSV"

    while IFS=, read -r \
        nodes ranks seed iterations subiterations graph_size graph_file communication_delay
    do
        echo "Running: nodes=$nodes ranks=$ranks seed=$seed"

        logfile=$(mktemp)

        mpirun \
            -np "$ranks" \
            "$EXEC" \
            "$seed" \
            "$iterations" \
            "$subiterations" \
            "$graph_size" \
            "$(dirname $EXEC)/$graph_file" \
            "$communication_delay" \
	    < /dev/null \
            | tee "$logfile" || true

        time_result=$(grep "Time:" "$logfile" | tail -n1 | sed -E 's/.*Time:[[:space:]]*//' || true)
        cut_result=$(grep "Final max cut:" "$logfile" | tail -n1 | sed -E 's/.*Final max cut:[[:space:]]*//' || true)

        rm -f "$logfile"

        echo "${nodes},${ranks},${seed},${iterations},${subiterations},${graph_size},${graph_file},${communication_delay},${time_result},${cut_result}" >> "$OUTCSV"
    done
} < "$CSV"

