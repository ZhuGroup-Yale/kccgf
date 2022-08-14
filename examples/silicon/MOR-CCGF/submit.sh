#!/bin/bash
set -e

for kpt in `seq 0 1 0`; do
    for orb in `seq 0 1 25`; do

        mkdir orb$orb-kpt$kpt
        cp si_ccgf.py orb$orb-kpt$kpt
        cp si_ccgf orb$orb-kpt$kpt
        cd orb$orb-kpt$kpt
        echo "OMP_NUM_THREADS=8 mpirun -np 1 --bind-to none python -u si_ccgf.py $orb $kpt" >> si_ccgf
        sbatch si_ccgf

        cd ../

    done
done
