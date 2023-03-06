#!/bin/bash

#SBATCH -J sddmm_var_sparsity_bench
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manya227@stanford.edu
#SBATCH --nodes=1
#SBATCH --mem MaxMemPerNode
#SBATCH --exclusive
#SBATCH --time=10:00:00

systems=("5_blas" "5_taco" "5_mkl" "5_gsl")

for i in "${systems[@]}"
do  
    /home/reviewer/mosaic/build/bin/./taco-bench --benchmark_filter=bench_blockedSparse4T_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/blockedSparse_4T_5/result/$i
done