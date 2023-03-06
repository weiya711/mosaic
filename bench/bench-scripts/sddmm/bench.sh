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

systems=("mkl" "gsl" "dot_blas" "taco" "blas" "tblis" "gemv_blas")

for i in "${systems[@]}"
do  
    /home/reviewer/mosaic/build/bin/./taco-bench --benchmark_filter=bench_sddmm_$i  --benchmark_format=json --benchmark_out=/home/reviewer/mosaic/bench/bench-scripts/sddmm/result/$i
done