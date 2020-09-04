# Generates the index matrix for the Subfind particledata arrays, in replacement of the
# /FOF/GroupLengthType and /FOF/GroupOffsetType arrays.

import numpy as np
import h5py as h5
from scipy.sparse import csr_matrix
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

from read import split, commune, get_header, pprint
from metadata import AttrDict


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def get_indices_sparse(data) -> AttrDict:
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]

def csr_index_matrix(files: tuple):
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
            pprint(f"[+] Computing CSR indexing matrix...")
            metadata[f'PartType{part_type}'] = {}
            metadata[f'PartType{part_type}']['csrmatrix'] = get_indices_sparse(GroupNumber[f'PartType{part_type}'])
            pprint(metadata[f'PartType{part_type}']['csrmatrix'][0], len(metadata[f'PartType{part_type}']['csrmatrix'][0]))

    # Construct the nested AttrDict instance
    csrm = AttrDict()
    csrm.data = metadata
    return csrm

def particle_index_from_csrm(fofgroup: AttrDict, particle_type: int, csrm: AttrDict) -> np.ndarray:

    N_particles = fofgroup.data.header.subfind_particles.NumPart_ThisFile[particle_type]
    start, end = split(N_particles)
    idx = fofgroup.data.clusterID
    particle_index = np.empty(0, dtype=np.int)
    particle_index = np.append(particle_index, csrm.data[f'PartType{particle_type}']['csrmatrix'][idx][0] + start)
    particle_index = commune(particle_index)
    return particle_index
