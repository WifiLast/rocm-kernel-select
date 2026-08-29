#!/bin/bash
A=/mnt/data/optimization/amd_tools/source/cmp_ext_turing/analyse/ablation
HIPCC=/opt/rocm/core-7.14/bin/hipcc
FLAGS="-O3 --offload-arch=gfx1100 -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -std=c++20 -fno-gpu-rdc -D__HIP_NO_HALF_OPERATORS__=1 -I/opt/rocm/core-7.14/include"
$HIPCC $FLAGS -c $A/driver.hip -o $A/driver.o || exit 1
while IFS='|' read -r name def; do
  [ -z "$name" ] && continue
  $HIPCC $FLAGS $def -c $A/k.hip -o $A/k.o 2>$A/err.txt || { printf "%-34s COMPILE FAIL\n" "$name"; head -5 $A/err.txt; continue; }
  $HIPCC $A/driver.o $A/k.o -o $A/bench --offload-arch=gfx1100 2>>$A/err.txt || { printf "%-34s LINK FAIL\n" "$name"; continue; }
  m=$($A/bench 2>/dev/null); for r in 1 2; do t=$($A/bench 2>/dev/null); m=$(python3 -c "print(min($m,$t))"); done; printf "%-34s %s ms (min of 3)\n" "$name" "$m"
done < $A/cfgs
