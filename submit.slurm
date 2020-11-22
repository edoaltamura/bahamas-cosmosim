#!/bin/bash

#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH -J bahamas_proc
#SBATCH -o log_bahamas.out
#SBATCH -e log_bahamas.err
#SBATCH -p CLUSTER
#SBATCH -A altamura
#SBATCH -t 10:00:00

module purge
module load intel_comp
module load rocks-openmpi
module load hdf5_1.10.4
module load python37

source /home/altamura/nas/rksz-venv/bin/activate

mpirun -np $SLURM_NTASKS python3 flight.py

deactivate

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode