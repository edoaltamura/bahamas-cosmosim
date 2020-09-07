#!/usr/bin/env bash

OMP_NUM_THREADS=12
NUMBA_NUM_THREADS=$OMP_NUM_THREADS

export OMP_NUM_THREADS
export NUMBA_NUM_THREADS

#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 -u ./offset_type.py > ./main.log &
#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 ./offset_type.py
#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 ./map_particles.py --gas --dark_matter --stars
/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 ./map_density.py --gas --stars