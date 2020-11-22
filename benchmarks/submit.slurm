#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20000
#SBATCH --exclusive
#SBATCH -J matmul_benchmark
#SBATCH -o log_benchmark.out
#SBATCH -e log_benchmark.err
#SBATCH -t 1:00:00

module purge
module load mpi/openmpi-x86_64
module load intelcompilers2017
module load hdf5_1.10.4
module load python37

source /home/altamura/nas/rksz-venv/bin/activate

export OMPI_MCA_btl_base_warn_component_unused=0
mpirun -np $SLURM_NTASKS python3 mpi_benchmark.py

deactivate

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode