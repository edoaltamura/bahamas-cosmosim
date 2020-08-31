import os
from typing import List, Dict
import numpy as np
import h5py as h5
import unyt
from scipy.sparse import csr_matrix
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()


def pprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


from metadata import Metadata, AttrDict

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


def get_header(files: list) -> AttrDict:
    """
    Gathers the header information from the files into
    an instance of the AttrDict class, which allows the access
    of the header data as nested attributes.
    Example: header.data.subfind_particles.MassTable

    :param files: tuple(list, str, list, list)
        Expected to parse the output from the find_files function.
        Structure of the tuple: (subfind_st, subfind_gt, subfind_pd,
        gadget_sn).
    :return: AttrDict
        It returns at AttrDict instance with nested attributes.
        The object gathers header attributes for the three file
        types: subfind group, particle data and Gadget snapshots.
    """
    # Collect header information for the 3 file types
    with h5.File(files[0][0], 'r') as f:
        st_header = dict(f['Header'].attrs)
    with h5.File(files[2], 'r') as f:
        sp_header = dict(f['Header'].attrs)
    with h5.File(files[3][0], 'r') as f:
        sn_header = dict(f['Header'].attrs)

    master_header = {}
    master_header['subfind_groups'] = st_header
    master_header['subfind_particles'] = sp_header
    master_header['gadget_snaps'] = sn_header

    # Construct the nested AttrDict instance
    header = AttrDict()
    header.data = master_header
    return header


def fof_groups(files: list, header: AttrDict) -> AttrDict:

    # Conversion factors
    conv_mass = 1e10 / header.data.subfind_particles.HubbleParam
    conv_length = header.data.subfind_particles.ExpansionFactor / header.data.subfind_particles.HubbleParam
    conv_density = 1e10 * header.data.subfind_particles.HubbleParam ** 2 / header.data.subfind_particles.ExpansionFactor ** 3
    conv_velocity = np.sqrt(header.data.subfind_particles.ExpansionFactor)

    # Units
    unit_mass = unyt.Solar_Mass
    unit_length = unyt.Mpc
    unit_density = unyt.Solar_Mass / unyt.Mpc ** 3
    unit_velocity = unyt.km / unyt.s


    pprint(f"[+] Find groups information...")

    # Find eagle subfind tab hdf5 internal paths
    subfind_tab_data = {}
    subfind_tab_data['FOF'] = {}
    subfind_tab_data['FOF']['FirstSubhaloID'] = np.empty(0, dtype=np.int)
    subfind_tab_data['FOF']['GroupCentreOfPotential'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['GroupLength'] = np.empty(0, dtype=np.int)
    subfind_tab_data['FOF']['GroupMass'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['GroupOffset'] = np.empty(0, dtype=np.int)
    subfind_tab_data['FOF']['Group_M_Crit200'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['Group_M_Crit2500'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['Group_M_Crit500'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['Group_M_Mean200'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['Group_M_Mean2500'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['Group_M_Mean500'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['Group_M_TopHat200'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['Group_R_Crit200'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['Group_R_Crit2500'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['Group_R_Crit500'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['Group_R_Mean200'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['Group_R_Mean2500'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['Group_R_Mean500'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['Group_R_TopHat200'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['FOF']['NumOfSubhalos'] = np.empty(0, dtype=np.int)

    subfind_tab_data['Subhalo'] = {}
    subfind_tab_data['Subhalo']['CentreOfMass'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['Subhalo']['CentreOfPotential'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['Subhalo']['GasSpin'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['Subhalo']['GroupNumber'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['Subhalo']['HalfMassProjRad'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['Subhalo']['HalfMassRad'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['Subhalo']['IDMostBound'] = np.empty(0, dtype=np.int)
    subfind_tab_data['Subhalo']['SubLength'] = np.empty(0, dtype=np.int)
    subfind_tab_data['Subhalo']['SubOffset'] = np.empty(0, dtype=np.int)
    subfind_tab_data['Subhalo']['Velocity'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['Subhalo']['Vmax'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['Subhalo']['VmaxRadius'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['Subhalo']['StarsMass'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['Subhalo']['StarsSpin'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['Subhalo']['StarFormationRate'] = np.empty(0, dtype=np.float32)
    subfind_tab_data['Subhalo']['StellarVelDisp'] = np.empty(0, dtype=np.float32)

    # Find subfind group tab hdf5 internal paths
    group_tab_data = {}
    group_tab_data['FOF'] = {}
    group_tab_data['FOF']['CentreOfMass'] = np.empty(0, dtype=np.float32)
    group_tab_data['FOF']['GroupLength'] = np.empty(0, dtype=np.int)
    group_tab_data['FOF']['GroupLengthType'] = np.empty(0, dtype=np.float32)
    group_tab_data['FOF']['GroupMassType'] = np.empty(0, dtype=np.float32)
    group_tab_data['FOF']['GroupOffset'] = np.empty(0, dtype=np.int)
    group_tab_data['FOF']['GroupOffsetType'] = np.empty(0, dtype=np.int)
    group_tab_data['FOF']['Mass'] = np.empty(0, dtype=np.float32)

    st, fh = split(len(files[0]))
    for x in range(st, fh, 1):
        with h5.File(files[0][x], 'r') as f:
            subfind_tab_data['FOF']['FirstSubhaloID'] = np.append(subfind_tab_data['FOF']['FirstSubhaloID'], f['FOF/FirstSubhaloID'][:])
            subfind_tab_data['FOF']['GroupCentreOfPotential'] = np.append(subfind_tab_data['FOF']['GroupCentreOfPotential'], f['FOF/GroupCentreOfPotential'][:])
            subfind_tab_data['FOF']['GroupLength'] = np.append(subfind_tab_data['FOF']['GroupLength'], f['FOF/GroupLength'][:])
            subfind_tab_data['FOF']['GroupMass'] = np.append(subfind_tab_data['FOF']['GroupMass'], f['FOF/GroupMass'][:])
            subfind_tab_data['FOF']['GroupOffset'] = np.append(subfind_tab_data['FOF']['GroupOffset'], f['FOF/GroupOffset'][:])
            subfind_tab_data['FOF']['Group_M_Crit200'] = np.append(subfind_tab_data['FOF']['Group_M_Crit200'], f['FOF/Group_M_Crit200'][:])
            subfind_tab_data['FOF']['Group_M_Crit2500'] = np.append(subfind_tab_data['FOF']['Group_M_Crit2500'], f['FOF/Group_M_Crit2500'][:])
            subfind_tab_data['FOF']['Group_M_Crit500'] = np.append(subfind_tab_data['FOF']['Group_M_Crit500'], f['FOF/Group_M_Crit500'][:])
            subfind_tab_data['FOF']['Group_M_Mean200'] = np.append(subfind_tab_data['FOF']['Group_M_Mean200'], f['FOF/Group_M_Mean200'][:])
            subfind_tab_data['FOF']['Group_M_Mean2500'] = np.append(subfind_tab_data['FOF']['Group_M_Mean2500'], f['FOF/Group_M_Mean2500'][:])
            subfind_tab_data['FOF']['Group_M_Mean500'] = np.append(subfind_tab_data['FOF']['Group_M_Mean500'], f['FOF/Group_M_Mean500'][:])
            subfind_tab_data['FOF']['Group_M_TopHat200'] = np.append(subfind_tab_data['FOF']['Group_M_TopHat200'], f['FOF/Group_M_TopHat200'][:])
            subfind_tab_data['FOF']['Group_R_Crit200'] = np.append(subfind_tab_data['FOF']['Group_R_Crit200'], f['FOF/Group_R_Crit200'][:])
            subfind_tab_data['FOF']['Group_R_Crit2500'] = np.append(subfind_tab_data['FOF']['Group_R_Crit2500'], f['FOF/Group_R_Crit2500'][:])
            subfind_tab_data['FOF']['Group_R_Crit500'] = np.append(subfind_tab_data['FOF']['Group_R_Crit500'], f['FOF/Group_R_Crit500'][:])
            subfind_tab_data['FOF']['Group_R_Mean200'] = np.append(subfind_tab_data['FOF']['Group_R_Mean200'], f['FOF/Group_R_Mean200'][:])
            subfind_tab_data['FOF']['Group_R_Mean2500'] = np.append(subfind_tab_data['FOF']['Group_R_Mean2500'], f['FOF/Group_R_Mean2500'][:])
            subfind_tab_data['FOF']['Group_R_Mean500'] = np.append(subfind_tab_data['FOF']['Group_R_Mean500'], f['FOF/Group_R_Mean500'][:])
            subfind_tab_data['FOF']['Group_R_TopHat200'] = np.append(subfind_tab_data['FOF']['Group_R_TopHat200'], f['FOF/Group_R_TopHat200'][:])
            subfind_tab_data['FOF']['NumOfSubhalos'] = np.append(subfind_tab_data['FOF']['NumOfSubhalos'], f['FOF/NumOfSubhalos'][:])

            subfind_tab_data['Subhalo']['CentreOfMass'] = np.append(subfind_tab_data['Subhalo']['CentreOfMass'], f['Subhalo/CentreOfMass'][:])
            subfind_tab_data['Subhalo']['CentreOfPotential'] = np.append(subfind_tab_data['Subhalo']['CentreOfPotential'], f['Subhalo/CentreOfPotential'][:])
            subfind_tab_data['Subhalo']['GasSpin'] = np.append(subfind_tab_data['Subhalo']['GasSpin'], f['Subhalo/GasSpin'][:])
            subfind_tab_data['Subhalo']['GroupNumber'] = np.append(subfind_tab_data['Subhalo']['GroupNumber'], f['Subhalo/GroupNumber'][:])
            subfind_tab_data['Subhalo']['HalfMassProjRad'] = np.append(subfind_tab_data['Subhalo']['HalfMassProjRad'], f['Subhalo/HalfMassProjRad'][:])
            subfind_tab_data['Subhalo']['HalfMassRad'] = np.append(subfind_tab_data['Subhalo']['HalfMassRad'], f['Subhalo/HalfMassRad'][:])
            subfind_tab_data['Subhalo']['IDMostBound'] = np.append(subfind_tab_data['Subhalo']['IDMostBound'], f['Subhalo/IDMostBound'][:])
            subfind_tab_data['Subhalo']['SubLength'] = np.append(subfind_tab_data['Subhalo']['SubLength'], f['Subhalo/SubLength'][:])
            subfind_tab_data['Subhalo']['SubOffset'] = np.append(subfind_tab_data['Subhalo']['SubOffset'], f['Subhalo/SubOffset'][:])
            subfind_tab_data['Subhalo']['Velocity'] = np.append(subfind_tab_data['Subhalo']['Velocity'], f['Subhalo/Velocity'][:])
            subfind_tab_data['Subhalo']['Vmax'] = np.append(subfind_tab_data['Subhalo']['Vmax'], f['Subhalo/Vmax'][:])
            subfind_tab_data['Subhalo']['VmaxRadius'] = np.append(subfind_tab_data['Subhalo']['VmaxRadius'], f['Subhalo/VmaxRadius'][:])
            subfind_tab_data['Subhalo']['StarsMass'] = np.append(subfind_tab_data['Subhalo']['StarsMass'], f['Subhalo/Stars/Mass'][:])
            subfind_tab_data['Subhalo']['StarsSpin'] = np.append(subfind_tab_data['Subhalo']['StarsSpin'], f['Subhalo/Stars/Spin'][:])
            subfind_tab_data['Subhalo']['StarFormationRate'] = np.append(subfind_tab_data['Subhalo']['StarFormationRate'], f['Subhalo/StarFormationRate'][:])
            subfind_tab_data['Subhalo']['StellarVelDisp'] = np.append(subfind_tab_data['Subhalo']['StellarVelDisp'], f['Subhalo/StellarVelDisp'][:])

    st, fh = split(len(files[1]))
    for x in range(st, fh, 1):
        with h5.File(files[1][x], 'r') as f:
            group_tab_data['FOF']['CentreOfMass'] = np.append(group_tab_data['FOF']['CentreOfMass'], f['FOF/CentreOfMass'][:])
            group_tab_data['FOF']['GroupLength'] = np.append(group_tab_data['FOF']['GroupLength'], f['FOF/GroupLength'][:])
            group_tab_data['FOF']['GroupLengthType'] = np.append(group_tab_data['FOF']['GroupLengthType'], f['FOF/GroupLengthType'][:])
            group_tab_data['FOF']['GroupMassType'] = np.append(group_tab_data['FOF']['GroupMassType'], f['FOF/GroupMassType'][:])
            group_tab_data['FOF']['GroupOffset'] = np.append(group_tab_data['FOF']['GroupOffset'], f['FOF/GroupOffset'][:])
            group_tab_data['FOF']['GroupOffsetType'] = np.append(group_tab_data['FOF']['GroupOffsetType'], f['FOF/GroupOffsetType'][:])
            group_tab_data['FOF']['Mass'] = np.append(group_tab_data['FOF']['Mass'], f['FOF/Mass'][:])

    subfind_tab_data['FOF']['FirstSubhaloID'] = commune(subfind_tab_data['FOF']['FirstSubhaloID'])
    subfind_tab_data['FOF']['GroupCentreOfPotential'] = commune(subfind_tab_data['FOF']['GroupCentreOfPotential'].reshape(-1, 1)).reshape(-1, 3) * conv_length * unit_length
    subfind_tab_data['FOF']['GroupLength'] = commune(subfind_tab_data['FOF']['GroupLength'])
    subfind_tab_data['FOF']['GroupMass'] = commune(subfind_tab_data['FOF']['GroupMass']) * conv_mass * unit_mass
    subfind_tab_data['FOF']['GroupOffset'] = commune(subfind_tab_data['FOF']['GroupOffset'])
    subfind_tab_data['FOF']['Group_M_Crit200'] = commune(subfind_tab_data['FOF']['Group_M_Crit200']) * conv_mass * unit_mass
    subfind_tab_data['FOF']['Group_M_Crit2500'] = commune(subfind_tab_data['FOF']['Group_M_Crit2500']) * conv_mass * unit_mass
    subfind_tab_data['FOF']['Group_M_Crit500'] = commune(subfind_tab_data['FOF']['Group_M_Crit500']) * conv_mass * unit_mass
    subfind_tab_data['FOF']['Group_M_Mean200'] = commune(subfind_tab_data['FOF']['Group_M_Mean200']) * conv_mass * unit_mass
    subfind_tab_data['FOF']['Group_M_Mean2500'] = commune(subfind_tab_data['FOF']['Group_M_Mean2500']) * conv_mass * unit_mass
    subfind_tab_data['FOF']['Group_M_Mean500'] = commune(subfind_tab_data['FOF']['Group_M_Mean500']) * conv_mass * unit_mass
    subfind_tab_data['FOF']['Group_M_TopHat200'] = commune(subfind_tab_data['FOF']['Group_M_TopHat200']) * conv_mass * unit_mass
    subfind_tab_data['FOF']['Group_R_Crit200'] = commune(subfind_tab_data['FOF']['Group_R_Crit200']) * conv_length * unit_length
    subfind_tab_data['FOF']['Group_R_Crit2500'] = commune(subfind_tab_data['FOF']['Group_R_Crit2500']) * conv_length * unit_length
    subfind_tab_data['FOF']['Group_R_Crit500'] = commune(subfind_tab_data['FOF']['Group_R_Crit500']) * conv_length * unit_length
    subfind_tab_data['FOF']['Group_R_Mean200'] = commune(subfind_tab_data['FOF']['Group_R_Mean200']) * conv_length * unit_length
    subfind_tab_data['FOF']['Group_R_Mean2500'] = commune(subfind_tab_data['FOF']['Group_R_Mean2500']) * conv_length * unit_length
    subfind_tab_data['FOF']['Group_R_Mean500'] = commune(subfind_tab_data['FOF']['Group_R_Mean500']) * conv_length * unit_length
    subfind_tab_data['FOF']['Group_R_TopHat200'] = commune(subfind_tab_data['FOF']['Group_R_TopHat200']) * conv_length * unit_length
    subfind_tab_data['FOF']['NumOfSubhalos'] = commune(subfind_tab_data['FOF']['NumOfSubhalos'])
    subfind_tab_data['Subhalo']['CentreOfMass'] = commune(subfind_tab_data['Subhalo']['CentreOfMass'].reshape(-1, 1)).reshape(-1, 3) * conv_length * unit_length
    subfind_tab_data['Subhalo']['CentreOfPotential'] = commune(subfind_tab_data['Subhalo']['CentreOfPotential'].reshape(-1, 1)).reshape(-1, 3) * conv_length * unit_length
    subfind_tab_data['Subhalo']['GasSpin'] = commune(subfind_tab_data['Subhalo']['GasSpin'].reshape(-1, 1)).reshape(-1, 3)
    subfind_tab_data['Subhalo']['GroupNumber'] = commune(subfind_tab_data['Subhalo']['GroupNumber'])
    subfind_tab_data['Subhalo']['HalfMassProjRad'] = commune(subfind_tab_data['Subhalo']['HalfMassProjRad'].reshape(-1, 1)).reshape(-1, 6) * conv_length * unit_length
    subfind_tab_data['Subhalo']['HalfMassRad'] = commune(subfind_tab_data['Subhalo']['HalfMassRad'].reshape(-1, 1)).reshape(-1, 6) * conv_length * unit_length
    subfind_tab_data['Subhalo']['IDMostBound'] = commune(subfind_tab_data['Subhalo']['IDMostBound'])
    subfind_tab_data['Subhalo']['SubLength'] = commune(subfind_tab_data['Subhalo']['SubLength'])
    subfind_tab_data['Subhalo']['SubOffset'] = commune(subfind_tab_data['Subhalo']['SubOffset'])
    subfind_tab_data['Subhalo']['Velocity'] = commune(subfind_tab_data['Subhalo']['Velocity'].reshape(-1, 1)).reshape(-1, 3) * conv_velocity * unit_velocity
    subfind_tab_data['Subhalo']['Vmax'] = commune(subfind_tab_data['Subhalo']['Vmax']) * conv_velocity * unit_velocity
    subfind_tab_data['Subhalo']['VmaxRadius'] = commune(subfind_tab_data['Subhalo']['VmaxRadius']) * conv_length * unit_length
    subfind_tab_data['Subhalo']['StarsMass'] = commune(subfind_tab_data['Subhalo']['StarsMass']) * conv_mass * unit_mass
    subfind_tab_data['Subhalo']['StarsSpin'] = commune(subfind_tab_data['Subhalo']['StarsSpin'].reshape(-1, 1)).reshape(-1, 3)
    subfind_tab_data['Subhalo']['StarFormationRate'] = commune(subfind_tab_data['Subhalo']['StarFormationRate'])
    subfind_tab_data['Subhalo']['StellarVelDisp'] = commune(subfind_tab_data['Subhalo']['StellarVelDisp']) * conv_velocity * unit_velocity
    group_tab_data['FOF']['CentreOfMass'] = commune(group_tab_data['FOF']['CentreOfMass'].reshape(-1, 1)).reshape(-1, 3) * conv_length * unit_length
    group_tab_data['FOF']['GroupLength'] = commune(group_tab_data['FOF']['GroupLength'])
    group_tab_data['FOF']['GroupLengthType'] = commune(group_tab_data['FOF']['GroupLengthType'].reshape(-1, 1)).reshape(-1, 6)
    group_tab_data['FOF']['GroupMassType'] = commune(group_tab_data['FOF']['GroupMassType'].reshape(-1, 1)).reshape(-1, 6)
    group_tab_data['FOF']['GroupOffset'] = commune(group_tab_data['FOF']['GroupOffset'])
    group_tab_data['FOF']['GroupOffsetType'] = commune(group_tab_data['FOF']['GroupOffsetType'].reshape(-1, 1)).reshape(-1, 6)
    group_tab_data['FOF']['Mass'] = commune(group_tab_data['FOF']['Mass']) * conv_mass * unit_mass

    # Gather all data into a large dictionary
    data_dict = {}
    data_dict['files'] = files
    data_dict['header'] = header.data
    data_dict['subfind_tab'] = subfind_tab_data
    data_dict['group_tab'] = group_tab_data
    data_obj = AttrDict()
    data_obj.data = data_dict
    return data_obj


def fof_group(clusterID: int, fofgroups: AttrDict) -> AttrDict:
    # pprint(f"[+] Find group information for cluster {clusterID}")

    # Filter groups
    filter_idx = np.where(
        fofgroups.data.subfind_tab.FOF.Group_M_Crit500 > 1e13
    )[0][clusterID]

    # Create an AttrDict object and push the filtered data
    new_data = fofgroups
    for category in ['FOF', 'Subhalo']:
        for dataset in fofgroups.data['subfind_tab'][category]:
            new_data.data['subfind_tab'][category][dataset] = None
            new_data.data['subfind_tab'][category][dataset] = fofgroups.data['subfind_tab'][category][dataset][filter_idx]
    for dataset in fofgroups.data['group_tab']['FOF']:
        new_data.data['group_tab']['FOF'][dataset] = None
        new_data.data['group_tab']['FOF'][dataset] = fofgroups.data['group_tab']['FOF'][dataset][filter_idx]

    return new_data

def fof_particles(fofgroups: AttrDict) -> AttrDict:
    with h5.File(fofgroups.data.files[2], 'r') as h5file:
        for pt in ['0', '1', '4']:
            start = fofgroups.data.group_tab.FOF.GroupOffsetType[int(pt)]
            end = start + fofgroups.data.group_tab.FOF.GroupLengthType[int(pt)]
            groupnumber = h5file[f'/PartType{pt}/GroupNumber'][start:end]
            pprint(groupnumber.size)

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
