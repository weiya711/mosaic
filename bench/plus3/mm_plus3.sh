#!/bin/bash

systems=("gsl_tensor" "tblis" "taco")

for i in "${systems[@]}"
do
   /home/manya227/taco-interoperable/build/bin/./taco-bench --benchmark_filter=bench_plus3_$i  --benchmark_format=json --benchmark_out=/home/manya227/taco-interoperable/bench/plus3/result/$i
done