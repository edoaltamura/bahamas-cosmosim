#!/bin/bash

#SBATCH --nodes=17
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --mem-per-cpu=1000
# SBATCH --exclusive
#SBATCH -J bahamas_proc
#SBATCH -o log_bahamas.out
#SBATCH -e log_bahamas.err
#SBATCH -t 10:00:00

module purge
module load rocks-openmpi
module load python37

source /home/altamura/nas/rksz-venv/bin/activate

export OMPI_MCA_btl_base_warn_component_unused=0
mpirun -np $SLURM_NTASKS python3 test_read.py

deactivate

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode