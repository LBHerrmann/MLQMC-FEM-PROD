#!/bin/bash
#BSUB -W 24:00
#BSUB -n 720
#BSUB -J U,m=1-10
#BSUB -o lsf/%J.out

#set number of processors
NP=${1:-$(</dev/stdin)}

for r in `seq 0 19 `; do # loop over r = 0, 1, ..., 20
# loop over number of points (N=2^m)
for Lmax in `seq 2 12`; do
    mpirun -np $NP python source/run_P1_Matern_preWav_MLQMC_MPI_precomputed.py --Lmax ${Lmax} --rep ${r} --alpha 4 --chi 0.9 --tau 1 --sigma0 0.1 --lambdaC 0.1
done
done
exit


