#!/bin/bash

#SBATCH --nodes=10
#SBATCH --ntasks-per-node=15
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH -J bahamas_vr
#SBATCH -o log_bahamas.out
#SBATCH -e log_bahamas.err
#SBATCH -t 10:00:00

module purge
module load mpi/openmpi-x86_64
module load gsl2.5
module load hdf5_1.10.4
module load gcc7.3.0

mpirun -np $SLURM_NTASKS /home/altamura/VELOCIraptor-STF/stf -I 2 \
  -i /share/nas_virgo/Cosmo-OWLS/AGN_TUNED_nu0_L400N1024_Planck/snapshot_032/snap_032 \
  -o /home/altamura/nas/bahamas-stf/catalogue_032 \
  -C /home/altamura/nas/bahamas-stf/vrconfig_3dfofbound_subhalos_SO_hydro.cfg


echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode