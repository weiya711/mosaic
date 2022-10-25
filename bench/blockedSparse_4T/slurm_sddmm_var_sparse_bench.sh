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

systems=("blas" "taco")

for i in "${systems[@]}"
do  
    /home/manya227/taco-interoperable/build/bin/./taco-bench --benchmark_filter=bench_blockedSparse4T_$i  --benchmark_format=json --benchmark_out=/home/manya227/taco-interoperable/bench/blockedSparse_4T/result/$i
done