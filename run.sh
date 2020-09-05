#!/usr/bin/env bash

#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n 12 python3 -u ./offset_type.py > ./main.log &
#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n 12 python3 ./offset_type.py
/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n 12 python3 ./map_particles.py