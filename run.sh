#!/usr/bin/env bash

#Activate the virtual environment with Swiftsimio
venv_path=$(pip -V)
if [[ ! $venv_path == *"rksz"* ]]; then
  echo "Activating project-specific Python virtual environment.."
  source /local/scratch/altamura/rksz-venv/bin/activate.csh
fi

OMP_NUM_THREADS=12
NUMBA_NUM_THREADS=12

export OMP_NUM_THREADS
export NUMBA_NUM_THREADS

#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 -u ./offset_type.py > ./main.log &
#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 ./offset_type.py
#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 ./map_particles.py --gas --dark_matter --stars
/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n $OMP_NUM_THREADS python3 -u ./map_density.py --gas --stars > ./main.log &