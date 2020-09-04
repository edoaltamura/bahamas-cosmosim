# Generates the equivalent of the /FOF/GroupLengthType and /FOF/GroupOffsetType
# arrays. The original Subfind results are not usable.
# This code assumes that the all group numbers of the particles within the same cluster
# are perfectly continuous, i.e. they're all clustered together in the large particledata
# array. If there are jumps, these arrays will not be useful and a sparse indexing matrix
# must be used.

import numpy as np
import h5py as h5
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

from read import find_files, split, commune, get_header, pprint
from metadata import Metadata

simulation_type = 'hydro'
output_directory = '/local/scratch/altamura/bahamas_metadata'
# redshift = 'z003p000'

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
            unique, unique_indices, unique_counts = np.unique(
                GroupNumber[f'PartType{part_type}'],
                return_index=True,
                return_counts=True
            )

            # Initialise and allocate metadata entries in each rank
            metadata[f'PartType{part_type}'] = {}
            metadata[f'PartType{part_type}']['unique'] = np.empty(0, dtype=np.int)
            metadata[f'PartType{part_type}']['offset'] = np.empty(0, dtype=np.int)
            metadata[f'PartType{part_type}']['length'] = np.empty(0, dtype=np.int)
            metadata[f'PartType{part_type}']['unique'] = np.append(
                metadata[f'PartType{part_type}']['unique'],
                unique
            )
            metadata[f'PartType{part_type}']['offset'] = np.append(
                metadata[f'PartType{part_type}']['offset'],
                unique_indices + start
            )
            metadata[f'PartType{part_type}']['length'] = np.append(
                metadata[f'PartType{part_type}']['length'],
                unique_counts
            )

            # Merge data across cores handling interface
            metadata[f'PartType{part_type}']['unique'] = commune(metadata[f'PartType{part_type}']['unique'])
            metadata[f'PartType{part_type}']['offset'] = commune(metadata[f'PartType{part_type}']['offset'])
            metadata[f'PartType{part_type}']['length'] = commune(metadata[f'PartType{part_type}']['length'])

            # Detect duplicates at the boundaries between ranks
            master_unique, master_unique_indices, master_unique_counts = np.unique(
                metadata[f'PartType{part_type}']['unique'],
                return_index=True,
                return_counts=True
            )
            gather = np.cumsum(np.insert(master_unique_counts, 0, 0))[:-1]
            metadata[f'PartType{part_type}']['length'] = np.add.reduceat(metadata[f'PartType{part_type}']['length'], gather)
            metadata[f'PartType{part_type}']['offset'] = metadata[f'PartType{part_type}']['offset'][master_unique_indices]
            metadata[f'PartType{part_type}']['unique'] = metadata[f'PartType{part_type}']['unique'][master_unique_indices]
            assert ~np.all(master_unique - metadata[f'PartType{part_type}']['unique'])

            # Sort the elements in the array from cluster 0 upwards
            sort_key = np.argsort(metadata[f'PartType{part_type}']['unique'])
            metadata[f'PartType{part_type}']['length'] = metadata[f'PartType{part_type}']['length'][sort_key]
            metadata[f'PartType{part_type}']['offset'] = metadata[f'PartType{part_type}']['offset'][sort_key]
            metadata[f'PartType{part_type}']['unique'] = metadata[f'PartType{part_type}']['unique'][sort_key]

            # pprint(f'PartType{part_type} unique', len(metadata[f'PartType{part_type}']['unique']), metadata[f'PartType{part_type}']['unique'])
            # pprint(f'PartType{part_type} length', len(metadata[f'PartType{part_type}']['length']), metadata[f'PartType{part_type}']['length'])
            # pprint(f'PartType{part_type} offset', len(metadata[f'PartType{part_type}']['offset']), metadata[f'PartType{part_type}']['offset'])

    comm.Barrier()
    if rank == 0:
        # Convert into arrays similar to the Subfind ones
        null_array = np.zeros_like(metadata[f'PartType{part_type}']['offset'], dtype=np.int)

        GroupLengthType = np.vstack((
            metadata['PartType0']['length'],#null_array,
            metadata['PartType1']['length'],
            null_array,
            null_array,
            metadata['PartType4']['length'],#null_array,
            null_array
        )).T

        GroupOffsetType = np.vstack((
            metadata['PartType0']['offset'],#null_array,
            metadata['PartType1']['offset'],
            null_array,
            null_array,
            metadata['PartType4']['offset'],#null_array,
            null_array
        )).T

        # Write output to hdf5 file
        with h5.File(f'{output_directory}/{simulation_type}_{redshift_idx}.hdf5', 'w') as h5file:
            h5file.create_dataset('GroupLengthType', dtype=np.int, shape=(len(GroupLengthType), 6), data=GroupLengthType)
            h5file.create_dataset('GroupOffsetType', dtype=np.int, shape=(len(GroupOffsetType), 6), data=GroupOffsetType)
