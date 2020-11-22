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
module load mpi/openmpi-x86_64
module load intelcompilers2017
module load hdf5_1.10.4
module load python37

source /home/altamura/nas/rksz-venv/bin/activate

export OMPI_MCA_btl_base_warn_component_unused=0
mpirun -env I_MPI_PIN=1 -env I_MPI_PIN_PROCESSOR_LIST=allcores -np $SLURM_NTASKS python3 bahamas-cosmosim/mapping.py

deactivate

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode