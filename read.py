import os
from typing import List, Dict
import numpy as np
import h5py as h5
from scipy.sparse import csr_matrix
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()


def pprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


from metadata import Metadata

from conversion import (
    comoving_density,
    comoving_length,
    comoving_velocity,
    comoving_mass,
    comoving_kinetic_energy,
    comoving_momentum,
    comoving_ang_momentum,
    density_units,
    velocity_units,
    length_units,
    mass_units,
    momentum_units,
    energy_units,
)


def split(nfiles):
    nfiles = int(nfiles)
    nf = int(nfiles / nproc)
    rmd = nfiles % nproc
    st = rank * nf
    fh = (rank + 1) * nf
    if rank < rmd:
        st += rank
        fh += (rank + 1)
    else:
        st += rmd
        fh += rmd
    return st, fh


def commune(data):
    tmp = np.zeros(nproc, dtype=np.int)
    tmp[rank] = len(data)
    cnts = np.zeros(nproc, dtype=np.int)
    comm.Allreduce([tmp, MPI.INT], [cnts, MPI.INT], op=MPI.SUM)
    del tmp
    dspl = np.zeros(nproc, dtype=np.int)
    i = 0
    for j in range(nproc):
        dspl[j] = i
        i += cnts[j]
    rslt = np.zeros(i, dtype=data.dtype)
    comm.Allgatherv([data, cnts[rank]], [rslt, cnts, dspl, MPI._typedict[data.dtype.char]])
    del data, cnts, dspl
    return rslt


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def find_files(simulation_type: str, redshift: str):

    # Import the Metadata preloaded class
    paths = Metadata.data.PATHS
    directory = paths.dir_hydro if simulation_type == "hydro" else paths.dir_dmo
    pprint(f"[+] Find {paths.simulation_name:s} {simulation_type:s} files {redshift:s}...")

    # Make filepath templates and substitute the redshift index
    template_st = directory + paths.subdir_subfind_groups + paths.filename_subfind_groups_st
    template_gt = directory + paths.subdir_subfind_groups + paths.filename_subfind_groups_gt
    template_pd = directory + paths.subdir_subfind_particledata + paths.filename_subfind_particledata
    template_sn = directory + paths.subdir_gadget_snapshots + paths.filename_gadget_snapshots

    template_st = template_st.replace('REDSHIFTIDX', Metadata.data.REDSHIFTS[redshift])
    template_gt = template_gt.replace('REDSHIFTIDX', Metadata.data.REDSHIFTS[redshift])
    template_pd = template_pd.replace('REDSHIFTIDX', Metadata.data.REDSHIFTS[redshift])
    template_sn = template_sn.replace('REDSHIFTIDX', Metadata.data.REDSHIFTS[redshift])

    # Find all files with different split indices and push into lists
    subfind_st = []
    subfind_gt = []
    subfind_pd = template_pd
    gadget_sn = []

    split_idx = 0
    while True:
        st = template_st.replace('SPLITIDX', str(split_idx))
        gt = template_gt.replace('SPLITIDX', str(split_idx))
        sn = template_sn.replace('SPLITIDX', str(split_idx))

        if os.path.isfile(st):
            subfind_st.append(st)
            st_isfile = True
        else:
            st_isfile = False
        if os.path.isfile(gt):
            subfind_gt.append(gt)
            gt_isfile = True
        else:
            gt_isfile = False
        if os.path.isfile(sn):
            gadget_sn.append(sn)
            sn_isfile = True
        else:
            sn_isfile = False

        # If none of these files is found break loop
        if not any([st_isfile, gt_isfile, sn_isfile]):
            break
        else:
            split_idx += 1

    return subfind_st, subfind_gt, subfind_pd, gadget_sn



def header(simulation_type: str, files: list):
    pprint(f"[+] Find header information...")
    header = {}
    with h5.File(files[2], 'r') as f:
        print(f['\Header'].attrs)
    return header


# def fof_mass_cut(files: list):
# 	st, fh = split(len(files))
# 	M500 = np.empty(0, dtype=np.float32)
# 	for x in range(st, fh, 1):
# 		with h5.File(files[x], 'r') as f:
# 			M500 = np.append(M500, f['FOF/Group_M_Crit500'][:])
#
# 	header = {}
# 	with h5.File(files[0][0], 'r') as f:
# 		header['Hub'] = f['Header'].attrs['HubbleParam']
# 		header['aexp'] = f['Header'].attrs['ExpansionFactor']
# 		header['zred'] = f['Header'].attrs['Redshift']
#
# 	M500 = comoving_mass(header, M500 * 1.0e10)
# 	M500_comm = commune(M500)
# 	del M500, st, fh, header
# 	idx = np.where(M500_comm > 1.0e13)[0]
# 	pprint(f"\t Found {len(idx)} clusters with M500 > 10^13 M_sun")
# 	return idx

def fof_groups(files: list):
    pprint(f"[+] Find groups information...")
    st, fh = split(len(files[0]))
    Mfof = np.empty(0, dtype=np.float32)
    M2500 = np.empty(0, dtype=np.float32)
    M500 = np.empty(0, dtype=np.float32)
    M200 = np.empty(0, dtype=np.float32)
    R2500 = np.empty(0, dtype=np.float32)
    R500 = np.empty(0, dtype=np.float32)
    R200 = np.empty(0, dtype=np.float32)
    COP = np.empty(0, dtype=np.float32)
    NSUB = np.empty(0, dtype=np.int)
    FSID = np.empty(0, dtype=np.int)
    SCOP = np.empty(0, dtype=np.float32)
    for x in range(st, fh, 1):
        with h5.File(files[0][x], 'r') as f:
            Mfof = np.append(Mfof, f['FOF/GroupMass'][:])
            M2500 = np.append(M2500, f['FOF/Group_M_Crit2500'][:])
            R2500 = np.append(R2500, f['FOF/Group_R_Crit2500'][:])
            M500 = np.append(M500, f['FOF/Group_M_Crit500'][:])
            R500 = np.append(R500, f['FOF/Group_R_Crit500'][:])
            M200 = np.append(M200, f['FOF/Group_M_Crit200'][:])
            R200 = np.append(R200, f['FOF/Group_R_Crit200'][:])
            COP = np.append(COP, f['FOF/GroupCentreOfPotential'][:])
            NSUB = np.append(NSUB, f['FOF/NumOfSubhalos'][:])
            FSID = np.append(FSID, f['FOF/FirstSubhaloID'][:])
            SCOP = np.append(SCOP, f['Subhalo/CentreOfPotential'][:])

    del st, fh
    st, fh = split(len(files[2]))
    group_offset_partype = np.empty(0, dtype=np.int)
    group_length_partype = np.empty(0, dtype=np.int)
    for x in range(st, fh, 1):
        with h5.File(files[0][x], 'r') as f:
            group_offset_partype = np.append(group_offset_partype, f['FOF/GroupOffsetType'][:])
            group_length_partype = np.append(group_length_partype, f['FOF/GroupLengthType'][:])

    header = {}
    with h5.File(files[0][0], 'r') as f:
        header['Hub'] = f['Header'].attrs['HubbleParam']
        header['aexp'] = f['Header'].attrs['ExpansionFactor']
        header['zred'] = f['Header'].attrs['Redshift']

    # Conversion
    Mfof = comoving_mass(header, Mfof * 1.0e10)
    M2500 = comoving_mass(header, M2500 * 1.0e10)
    M500 = comoving_mass(header, M500 * 1.0e10)
    M200 = comoving_mass(header, M200 * 1.0e10)
    R2500 = comoving_length(header, R2500)
    R500 = comoving_length(header, R500)
    R200 = comoving_length(header, R200)
    COP = comoving_length(header, COP)
    SCOP = comoving_length(header, SCOP)

    data = {}
    data['groupfiles'] = files[0]
    data['particlefiles'] = files[1]
    data['grouptabfiles'] = files[2]
    data['Mfof'] = commune(Mfof)
    data['M2500'] = commune(M2500)
    data['R2500'] = commune(R2500)
    data['M500'] = commune(M500)
    data['R500'] = commune(R500)
    data['M200'] = commune(M200)
    data['R200'] = commune(R200)
    data['COP'] = commune(COP.reshape(-1, 1)).reshape(-1, 3)
    data['NSUB'] = commune(NSUB)
    data['FSID'] = commune(FSID)
    data['SCOP'] = commune(SCOP.reshape(-1, 1)).reshape(-1, 3)
    data['group_offset_partype'] = commune(group_offset_partype.reshape(-1, 1)).reshape(-1, 6)
    data['group_length_partype'] = commune(group_length_partype.reshape(-1, 1)).reshape(-1, 6)
    data['idx'] = np.where(data['M500'] > 1.0e13)[0]

    return data


def fof_group(clusterID: int, fofgroups: Dict[str, np.ndarray] = None):
    # pprint(f"[+] Find group information for cluster {clusterID}")
    new_data = {}
    new_data['clusterID'] = clusterID
    new_data['idx'] = fofgroups['idx'][clusterID]
    new_data['Mfof'] = fofgroups['Mfof'][new_data['idx']]
    new_data['M2500'] = fofgroups['M2500'][new_data['idx']]
    new_data['R2500'] = fofgroups['R2500'][new_data['idx']]
    new_data['M500'] = fofgroups['M500'][new_data['idx']]
    new_data['R500'] = fofgroups['R500'][new_data['idx']]
    new_data['M200'] = fofgroups['M200'][new_data['idx']]
    new_data['R200'] = fofgroups['R200'][new_data['idx']]
    new_data['COP'] = fofgroups['COP'][new_data['idx']]
    new_data['NSUB'] = fofgroups['NSUB'][new_data['idx']]
    new_data['FSID'] = fofgroups['FSID'][new_data['idx']]
    new_data['SCOP'] = fofgroups['SCOP'][new_data['idx']]
    new_data['group_offset_partype'] = fofgroups['group_offset_partype'][new_data['idx']]
    new_data['group_length_partype'] = fofgroups['group_length_partype'][new_data['idx']]
    new_data['groupfiles'] = fofgroups['groupfiles']
    new_data['particlefiles'] = fofgroups['particlefiles']
    new_data['grouptabfiles'] = fofgroups['grouptabfiles']

    return new_data


def snap_groupnumbers(fofgroups: Dict[str, np.ndarray] = None):
    pgn = []
    with h5.File(fofgroups['particlefiles'], 'r') as h5file:
        for pt in ['0', '1', '4']:
            Nparticles = h5file['Header'].attrs['NumPart_ThisFile'][int(pt)]
            st, fh = split(Nparticles)
            pprint(f"[+] Collecting particleType {pt} GroupNumber...")
            groupnumber = h5file[f'/PartType{pt}/GroupNumber'][st:fh]

            # Clip out negative values and exceeding values
            groupnumber = np.clip(groupnumber, 0, fofgroups['idx'][-1] + 1)
            pprint(f"\t Computing CSR indexing matrix...")
            groupnumber_csrm = get_indices_sparse(groupnumber)
            del groupnumber_csrm[0], groupnumber_csrm[-1]
            pgn.append(groupnumber_csrm)
            del groupnumber

    return pgn


def cluster_partgroupnumbers(fofgroup: Dict[str, np.ndarray] = None, groupNumbers: List[np.ndarray] = None):
    # pprint(f"[+] Find particle groupnumbers for cluster {fofgroup['clusterID']}")
    pgn = []
    partTypes = ['0', '1', '4']
    with h5.File(fofgroup['particlefiles'], 'r') as h5file:
        for pt in partTypes:
            # Gather groupnumbers from cores
            Nparticles = h5file['Header'].attrs['NumPart_ThisFile'][int(pt)]
            st, fh = split(Nparticles)
            gn_cores = groupNumbers[partTypes.index(pt)][fofgroup['idx']][0] + st
            gn_comm = commune(gn_cores)
            pgn.append(gn_comm)
            # pprint(f"\t PartType {pt} found {len(gn_comm)} particles")
            del gn_cores, gn_comm
    return pgn


def snap_coordinates(fofgroups: Dict[str, np.ndarray] = None):
    coords_allpt = []
    with h5.File(fofgroups['particlefiles'], 'r') as h5file:
        for pt in ['0', '1', '4']:
            Nparticles = h5file['Header'].attrs['NumPart_ThisFile'][int(pt)]
            st, fh = split(Nparticles)
            pprint(f"[+] Collecting particleType {pt} coordinates...")
            coords_cores = h5file[f'/PartType{pt}/Coordinates'][st:fh]
            print(coords_cores[0])
            coords_pt = commune(coords_cores.reshape(-1, 1)).reshape(-1, 3)
            coords_allpt.append(coords_pt)
            del coords_pt

    return coords_allpt


def cluster_partapertures(fofgroup: Dict[str, np.ndarray] = None, coordinatesAll: List[np.ndarray] = None):
    # pprint(f"[+] Find particle groupnumbers for cluster {fofgroup['clusterID']}")
    block_all = []
    partTypes = ['0', '1', '4']
    with h5.File(fofgroup['particlefiles'], 'r') as h5file:
        for pt in partTypes:

            Nparticles = h5file['Header'].attrs['NumPart_ThisFile'][int(pt)]
            st, fh = split(Nparticles)
            coords = coordinatesAll[partTypes.index(pt)][st:fh]

            # Periodic boundary wrapping of particle coordinates
            boxsize = h5file['Header'].attrs['BoxSize']
            for coord_axis in range(3):
                # Right boundary
                if fofgroup['COP'][coord_axis] + 5 * fofgroup['R200'] > boxsize:
                    beyond_index = np.where(coords[:, coord_axis] < boxsize / 2)[0]
                    coords[beyond_index, coord_axis] += boxsize
                    del beyond_index
                # Left boundary
                elif fofgroup['COP'][coord_axis] - 5 * fofgroup['R200'] < 0.:
                    beyond_index = np.where(coords[:, coord_axis] > boxsize / 2)[0]
                    coords[beyond_index, coord_axis] -= boxsize
                    del beyond_index

            block_idx = np.where(
                (np.abs(coords[:, 0] - fofgroup['COP'][0]) < 5 * fofgroup['R200']) &
                (np.abs(coords[:, 1] - fofgroup['COP'][1]) < 5 * fofgroup['R200']) &
                (np.abs(coords[:, 2] - fofgroup['COP'][2]) < 5 * fofgroup['R200'])
            )[0] + st
            block_idx_comm = commune(block_idx)
            block_all.append(block_idx_comm)
            del block_idx_comm

    return block_all


def cluster_particles(fofgroup: Dict[str, np.ndarray] = None, groupNumbers: List[np.ndarray] = None):
    """

	:param fofgroup:
	:param groupNumbers:
	:return:
	"""
    # pprint(f"[+] Find particle information for cluster {fofgroup['clusterID']}")
    data_out = {}
    header = {}
    partTypes = ['0', '1', '4']
    with h5.File(fofgroup['particlefiles'], 'r') as h5file:

        header['Hub'] = h5file['Header'].attrs['HubbleParam']
        header['aexp'] = h5file['Header'].attrs['ExpansionFactor']
        header['zred'] = h5file['Header'].attrs['Redshift']

        for pt in partTypes:

            # Initialise particledata arrays
            pgn_core = np.empty(0, dtype=np.int)
            subgroup_number = np.empty(0, dtype=np.int)
            velocity = np.empty(0, dtype=np.float32)
            coordinates = np.empty(0, dtype=np.float32)
            mass = np.empty(0, dtype=np.float32)
            temperature = np.empty(0, dtype=np.float32)
            sphdensity = np.empty(0, dtype=np.float32)
            sphlength = np.empty(0, dtype=np.float32)

            # Let each CPU core import a portion of the pgn data
            pgn = groupNumbers[partTypes.index(pt)]
            st, fh = split(len(pgn))
            pgn_core = np.append(pgn_core, pgn[st:fh])
            del pgn

            # Filter particle data with collected groupNumber indexing
            subgroup_number = np.append(subgroup_number, h5file[f'/PartType{pt}/SubGroupNumber'][pgn_core])
            velocity = np.append(velocity, h5file[f'/PartType{pt}/Velocity'][pgn_core])
            coordinates = np.append(coordinates, h5file[f'/PartType{pt}/Coordinates'][pgn_core])
            if pt == '1':
                particle_mass_DM = h5file['Header'].attrs['MassTable'][1]
                mass = np.append(mass, np.ones(len(pgn_core), dtype=np.float32) * particle_mass_DM)
            else:
                mass = np.append(mass, h5file[f'/PartType{pt}/Mass'][pgn_core])
            if pt == '0':
                temperature = np.append(temperature, h5file[f'/PartType{pt}/Temperature'][pgn_core])
                sphdensity = np.append(sphdensity, h5file[f'/PartType{pt}/Density'][pgn_core])
                sphlength = np.append(sphlength, h5file[f'/PartType{pt}/SmoothingLength'][pgn_core])

            del pgn_core

            # Conversion from comoving units to physical units
            velocity = comoving_velocity(header, velocity)
            coordinates = comoving_length(header, coordinates)
            mass = comoving_mass(header, mass * 1.0e10)
            if pt == '0':
                den_conv = h5file[f'/PartType{pt}/Density'].attrs['CGSConversionFactor']
                sphdensity = comoving_density(header, sphdensity * den_conv)
                sphlength = comoving_length(header, sphlength)

            # Gather the imports across cores
            data_out[f'partType{pt}'] = {}
            data_out[f'partType{pt}']['subgroupnumber'] = commune(subgroup_number)
            data_out[f'partType{pt}']['velocity'] = commune(velocity.reshape(-1, 1)).reshape(-1, 3)
            data_out[f'partType{pt}']['coordinates'] = commune(coordinates.reshape(-1, 1)).reshape(-1, 3)
            data_out[f'partType{pt}']['mass'] = commune(mass)
            if pt == '0':
                data_out[f'partType{pt}']['temperature'] = commune(temperature)
                data_out[f'partType{pt}']['sphdensity'] = commune(sphdensity)
                data_out[f'partType{pt}']['sphlength'] = commune(sphlength)

            del subgroup_number, velocity, mass, coordinates, temperature, sphdensity, sphlength

            # Periodic boundary wrapping of particle coordinates
            coords = data_out[f'partType{pt}']['coordinates']
            boxsize = comoving_length(header, h5file['Header'].attrs['BoxSize'])
            for coord_axis in range(3):
                # Right boundary
                if fofgroup['COP'][coord_axis] + 5 * fofgroup['R200'] > boxsize:
                    beyond_index = np.where(coords[:, coord_axis] < boxsize / 2)[0]
                    coords[beyond_index, coord_axis] += boxsize
                    del beyond_index
                # Left boundary
                elif fofgroup['COP'][coord_axis] - 5 * fofgroup['R200'] < 0.:
                    beyond_index = np.where(coords[:, coord_axis] > boxsize / 2)[0]
                    coords[beyond_index, coord_axis] -= boxsize
                    del beyond_index

            data_out[f'partType{pt}']['coordinates'] = coords
            del coords, boxsize

    return data_out


def cluster_data(clusterID: int,
                 header: Dict[str, float] = None,
                 fofgroups: Dict[str, np.ndarray] = None,
                 groupNumbers: List[np.ndarray] = None,
                 coordinates: List[np.ndarray] = None):
    """

	:param clusterID:
	:param header:
	:param fofgroups:
	:param groupNumbers:
	:return:
	"""
    pprint(f"[+] Running cluster {clusterID}")
    group_data = fof_group(clusterID, fofgroups=fofgroups)
    # halo_partgn = cluster_partgroupnumbers(fofgroup=group_data, groupNumbers=groupNumbers)
    halo_partgn = cluster_partapertures(fofgroup=group_data, coordinatesAll=coordinates)
    part_data = cluster_particles(fofgroup=group_data, groupNumbers=halo_partgn)

    out = {}
    out['Header'] = {**header}
    out['FOF'] = {**group_data}
    for pt in ['0', '1', '4']:
        out[f'partType{pt}'] = {**part_data[f'partType{pt}']}
    return out


def glance_cluster(cluster_dict: dict, verbose: bool = False, indent: int = 1) -> None:
    """

	:param cluster_dict:
	:param verbose:
	:param indent:
	:return:
	"""
    if not verbose:
        for key, value in cluster_dict.items():
            if isinstance(value, dict):
                pprint('\t' * indent + str(key))
                glance_cluster(value, indent=indent + 1)
            elif (isinstance(value, np.ndarray) or isinstance(value, list)) and len(value) > 10:
                pprint('\t' * indent + str(key) + ' : ' + f"len({len(value):d})\t val({value[0]} ... {value[-1]})")
            else:
                pprint('\t' * indent + str(key) + ' : ' + str(value))

    if verbose:
        for key, value in cluster_dict.items():
            if isinstance(value, dict):
                pprint('\t' * indent + str(key))
                glance_cluster(value, indent=indent + 1)
            else:
                pprint('\t' * indent + str(key) + ' : ' + str(value))
