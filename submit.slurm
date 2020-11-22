#!/bin/bash

#SBATCH --nodes=10
#SBATCH --ntasks-per-node=15
#SBATCH --mem=20000
#SBATCH --exclusive
#SBATCH -J bahamas_proc
#SBATCH -o log_bahamas.out
#SBATCH -e log_bahamas.err
#SBATCH -t 10:00:00

module purge
module load rocks-openmpi
module load python37

source /home/altamura/nas/rksz-venv/bin/activate

export OMPI_MCA_btl_base_warn_component_unused=0
mpirun -np $SLURM_NTASKS python3 bahamas-cosmosim/mapping.py

deactivate

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode