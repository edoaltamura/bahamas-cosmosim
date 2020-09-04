# Generates the index matrix for the Subfind particledata arrays, in replacement of the
# /FOF/GroupLengthType and /FOF/GroupOffsetType arrays.

import numpy as np
import h5py as h5
from scipy.sparse import csr_matrix
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

from read import find_files, split, commune, get_header, pprint
from metadata import Metadata

simulation_type = 'dmo'
output_directory = '/local/scratch/altamura/bahamas_metadata'
# redshift = 'z003p000'

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]

for redshift in Metadata.data.REDSHIFTS:

    redshift_idx = Metadata.data.REDSHIFTS[redshift]
    files = find_files(simulation_type, redshift)
    header = get_header(files)

    GroupNumber = {}    # Input data structure
    metadata = {}       # Output data structure

    with h5.File(files[2], 'r') as h5file:

        # Create a HYDRO/DMO switch
        if "/PartType0" in h5file:
            part_types = [0, 1, 4]
        else:
            part_types = [1]

        # Loop over particle types (hydro/dmo sensitive)
        for part_type in part_types:

            # Read in GroupNumber info
            N_particles = header.data.subfind_particles.NumPart_ThisFile[part_type]
            start, end = split(N_particles)
            GroupNumber[f'PartType{part_type}'] = np.empty(0, dtype=np.int)
            GroupNumber[f'PartType{part_type}'] = np.append(
                GroupNumber[f'PartType{part_type}'],
                np.abs(h5file[f'PartType{part_type}/GroupNumber'][start:end])
            )

            # Generate the metadata in parallel through MPI
            pprint(f"Computing CSR indexing matrix...")
            metadata[f'PartType{part_type}'] = {}
            metadata[f'PartType{part_type}']['csrmatrix'] = get_indices_sparse(GroupNumber[f'PartType{part_type}']) + start

            # Merge data across cores handling interface
            metadata[f'PartType{part_type}']['csrmatrix'] = commune(metadata[f'PartType{part_type}']['csrmatrix'])
            pprint(metadata[f'PartType{part_type}']['csrmatrix'], metadata[f'PartType{part_type}']['csrmatrix'].shape)
            pprint(metadata[f'PartType{part_type}']['csrmatrix'][0], len(metadata[f'PartType{part_type}']['csrmatrix'][0]))


    comm.Barrier()
    if rank == 0:

        # Write output to hdf5 file
        with h5.File(f'{output_directory}/{simulation_type}_{redshift_idx}.hdf5', 'a') as h5file:
            CSRMatrix = h5file.create_group('CSRMatrix')
            # Loop over particle types (hydro/dmo sensitive)
            for part_type in part_types:
                CSRMatrix.create_dataset(
                    f'PartType{part_type}',
                    dtype=np.int,
                    shape=metadata[f'PartType{part_type}']['csrmatrix'].shape,
                    data=metadata[f'PartType{part_type}']['csrmatrix']
                )
