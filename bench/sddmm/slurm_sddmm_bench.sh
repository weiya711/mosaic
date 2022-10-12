#!/bin/bash

#SBATCH -J saxpy_bench
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manya227@stanford.edu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=100
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --array=0-1

systems=("blas" "taco")

srun /home/manya227/taco-interoperable/build/bin/./taco-bench --benchmark_filter=bench_sddmm_${systems[$SLURM_ARRAY_TASK_ID]}  --benchmark_format=json --benchmark_out=/home/manya227/taco-interoperable/bench/sddmm/result/${systems[$SLURM_ARRAY_TASK_ID]}