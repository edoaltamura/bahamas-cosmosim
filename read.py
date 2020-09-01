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
    conv_starFormationRate = 1e10 * header.data.subfind_particles.HubbleParam ** 2 / header.data.subfind_particles.ExpansionFactor ** 3

    # Units
    unit_mass = unyt.Solar_Mass
    unit_length = unyt.Mpc
    unit_density = unyt.Solar_Mass / unyt.Mpc ** 3
    unit_velocity = unyt.km / unyt.s
    unit_starFormationRate = unyt.Solar_Mass / (unyt.year * unyt.Mpc ** 3)


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
    subfind_tab_data['Subhalo']['GroupNumber'] = np.empty(0, dtype=np.int)
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
    subfind_tab_data['Subhalo']['StarFormationRate'] = commune(subfind_tab_data['Subhalo']['StarFormationRate']) * conv_starFormationRate * unit_starFormationRate
    subfind_tab_data['Subhalo']['StellarVelDisp'] = commune(subfind_tab_data['Subhalo']['StellarVelDisp']) * conv_velocity * unit_velocity
    group_tab_data['FOF']['CentreOfMass'] = commune(group_tab_data['FOF']['CentreOfMass'].reshape(-1, 1)).reshape(-1, 3) * conv_length * unit_length
    group_tab_data['FOF']['GroupLength'] = commune(group_tab_data['FOF']['GroupLength'])
    group_tab_data['FOF']['GroupLengthType'] = commune(group_tab_data['FOF']['GroupLengthType'].reshape(-1, 1)).reshape(-1, 6)
    group_tab_data['FOF']['GroupMassType'] = commune(group_tab_data['FOF']['GroupMassType'].reshape(-1, 1)).reshape(-1, 6) * conv_mass * unit_mass
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

    # Edit the AttrDict object and push the filtered data
    for category in ['FOF', 'Subhalo']:
        for dataset in fofgroups.data['subfind_tab'][category]:
            fofgroups.data['subfind_tab'][category][dataset] = fofgroups.data['subfind_tab'][category][dataset][filter_idx]
    for dataset in fofgroups.data['group_tab']['FOF']:
        fofgroups.data['group_tab']['FOF'][dataset] = fofgroups.data['group_tab']['FOF'][dataset][filter_idx]

    return fofgroups

def fof_particles(fofgroup: AttrDict) -> AttrDict:

    with h5.File(fofgroup.data.files[2], 'r') as h5file:

        # Create a HYDRO/DMO switch
        is_hydro = "/PartType0" in h5file

        if is_hydro:
            for pt in ['0', '1', '4']:
                offset = int(fofgroup.data.group_tab.FOF.GroupOffsetType[int(pt)])
                length = int(fofgroup.data.group_tab.FOF.GroupLengthType[int(pt)])
                start, end = split(length)
                start += offset
                end += offset

                groupnumber = h5file[f'/PartType{pt}/SubGroupNumber'][start:end]
                groupnumber = commune(groupnumber)
                pprint(f"GroupNumber{pt}", offset, length, groupnumber)
                coords = h5file[f'/PartType{pt}/Coordinates'][start:end]
                coords = commune(coords.reshape(-1, 1)).reshape(-1, 3)
                pprint(f"Coordinates{pt}", offset, length, coords)

        else:
            pass


