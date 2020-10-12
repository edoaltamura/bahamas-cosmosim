#!/usr/bin/env bash

#Activate the virtual environment with Swiftsimio. Run:
# source /local/scratch/altamura/rksz-venv/bin/activate.csh

OMP_NUM_THREADS=12
NUMBA_NUM_THREADS=12

export OMP_NUM_THREADS
export NUMBA_NUM_THREADS

#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 -u ./offset_type.py > ./main.log &
#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 ./offset_type.py
#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 ./map_particles.py --gas --dark_matter --stars
#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 -u ./map_density.py --gas --dark_matter --stars > ./main.log &

/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 ./test_read.py
#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 -u ./mapping.py > ./main.log &

# Copy the outputs locally
# scp mizar:/local/scratch/altamura/bahamas/maps/* .