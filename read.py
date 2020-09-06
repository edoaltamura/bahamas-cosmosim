import os
import numpy as np
import h5py as h5
import unyt
from scipy.sparse import csr_matrix
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

from metadata import Metadata, AttrDict


def pprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def class_wrap(input: dict) -> AttrDict:
    data_obj = AttrDict()
    data_obj.data = input
    return data_obj


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

    # Handle custom metadata structures
    template_meta = paths.custom_metadata.replace('SIMTYPE', simulation_type)
    custom_metadata = template_meta.replace('REDSHIFTIDX', Metadata.data.REDSHIFTS[redshift])

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

    return subfind_st, subfind_gt, subfind_pd, gadget_sn, custom_metadata


def get_header(files: list) -> dict:
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
    return master_header


def fof_groups(files: list) -> dict:

    # Introduce header handle
    master_header = get_header(files)
    header = master_header['subfind_particles']

    # Conversion factors
    conv_mass = 1e10 / header['HubbleParam']
    conv_length = header['ExpansionFactor'] / header['HubbleParam']
    conv_density = 1e10 * header['HubbleParam'] ** 2 / header['ExpansionFactor'] ** 3
    conv_velocity = np.sqrt(header['ExpansionFactor'])
    conv_starFormationRate = 1e10 * header['HubbleParam'] ** 2 / header['ExpansionFactor'] ** 3

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
    group_tab_data['FOF']['GroupLengthType'] = np.empty(0, dtype=np.int)
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
            group_tab_data['FOF']['GroupMassType'] = np.append(group_tab_data['FOF']['GroupMassType'], f['FOF/GroupMassType'][:])
            group_tab_data['FOF']['GroupOffset'] = np.append(group_tab_data['FOF']['GroupOffset'], f['FOF/GroupOffset'][:])
            group_tab_data['FOF']['Mass'] = np.append(group_tab_data['FOF']['Mass'], f['FOF/Mass'][:])
            group_tab_data['FOF']['GroupLengthType'] = np.append(group_tab_data['FOF']['GroupLengthType'], f['FOF/GroupLengthType'][:])
            group_tab_data['FOF']['GroupOffsetType'] = np.append(group_tab_data['FOF']['GroupOffsetType'], f['FOF/GroupOffsetType'][:])

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

    # Edit the AttrDict object and push the filtered data
    filter_idx = np.where(
        subfind_tab_data['FOF']['Group_M_Mean200'] > 1e13
    )[0]

    for category in ['FOF', 'Subhalo']:
        for dataset in subfind_tab_data[category]:
            subfind_tab_data[category][dataset] = subfind_tab_data[category][dataset][filter_idx]
    for dataset in group_tab_data['FOF']:
        group_tab_data['FOF'][dataset] = group_tab_data['FOF'][dataset][filter_idx]

    # Gather all data into a large dictionary
    data_dict = {}
    data_dict['files'] = files
    data_dict['header'] = master_header
    data_dict['subfind_tab'] = subfind_tab_data
    data_dict['group_tab'] = group_tab_data
    data_dict['mass_DMpart'] = header['MassTable'][1] * conv_mass * unit_mass

    return data_dict


def fof_group(clusterID: int, fofgroups: dict) -> dict:
    # pprint(f"[+] Find group information for cluster {clusterID}")

    # Filter groups
    fofgroups['subfind_tab']['FOF']['FirstSubhaloID'] = fofgroups['subfind_tab']['FOF']['FirstSubhaloID'][clusterID]
    fofgroups['subfind_tab']['FOF']['GroupCentreOfPotential'] = fofgroups['subfind_tab']['FOF']['GroupCentreOfPotential'][clusterID]
    fofgroups['subfind_tab']['FOF']['GroupLength'] = fofgroups['subfind_tab']['FOF']['GroupLength'][clusterID]
    fofgroups['subfind_tab']['FOF']['GroupMass'] = fofgroups['subfind_tab']['FOF']['GroupMass'][clusterID]
    fofgroups['subfind_tab']['FOF']['GroupOffset'] = fofgroups['subfind_tab']['FOF']['GroupOffset'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_M_Crit200'] = fofgroups['subfind_tab']['FOF']['Group_M_Crit200'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_M_Crit2500'] = fofgroups['subfind_tab']['FOF']['Group_M_Crit2500'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_M_Crit500'] = fofgroups['subfind_tab']['FOF']['Group_M_Crit500'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_M_Mean200'] = fofgroups['subfind_tab']['FOF']['Group_M_Mean200'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_M_Mean2500'] = fofgroups['subfind_tab']['FOF']['Group_M_Mean2500'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_M_Mean500'] = fofgroups['subfind_tab']['FOF']['Group_M_Mean500'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_M_TopHat200'] = fofgroups['subfind_tab']['FOF']['Group_M_TopHat200'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_R_Crit200'] = fofgroups['subfind_tab']['FOF']['Group_R_Crit200'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_R_Crit2500'] = fofgroups['subfind_tab']['FOF']['Group_R_Crit2500'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_R_Crit500'] = fofgroups['subfind_tab']['FOF']['Group_R_Crit500'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_R_Mean200'] = fofgroups['subfind_tab']['FOF']['Group_R_Mean200'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_R_Mean2500'] = fofgroups['subfind_tab']['FOF']['Group_R_Mean2500'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_R_Mean500'] = fofgroups['subfind_tab']['FOF']['Group_R_Mean500'][clusterID]
    fofgroups['subfind_tab']['FOF']['Group_R_TopHat200'] = fofgroups['subfind_tab']['FOF']['Group_R_TopHat200'][clusterID]
    fofgroups['subfind_tab']['FOF']['NumOfSubhalos'] = fofgroups['subfind_tab']['FOF']['NumOfSubhalos'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['CentreOfMass'] = fofgroups['subfind_tab']['Subhalo']['CentreOfMass'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['CentreOfPotential'] = fofgroups['subfind_tab']['Subhalo']['CentreOfPotential'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['GasSpin'] = fofgroups['subfind_tab']['Subhalo']['GasSpin'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['GroupNumber'] = fofgroups['subfind_tab']['Subhalo']['GroupNumber'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['HalfMassProjRad'] = fofgroups['subfind_tab']['Subhalo']['HalfMassProjRad'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['HalfMassRad'] = fofgroups['subfind_tab']['Subhalo']['HalfMassRad'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['IDMostBound'] = fofgroups['subfind_tab']['Subhalo']['IDMostBound'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['SubLength'] = fofgroups['subfind_tab']['Subhalo']['SubLength'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['SubOffset'] = fofgroups['subfind_tab']['Subhalo']['SubOffset'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['Velocity'] = fofgroups['subfind_tab']['Subhalo']['Velocity'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['Vmax'] = fofgroups['subfind_tab']['Subhalo']['Vmax'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['VmaxRadius'] = fofgroups['subfind_tab']['Subhalo']['VmaxRadius'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['StarsMass'] = fofgroups['subfind_tab']['Subhalo']['StarsMass'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['StarsSpin'] = fofgroups['subfind_tab']['Subhalo']['StarsSpin'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['StarFormationRate'] = fofgroups['subfind_tab']['Subhalo']['StarFormationRate'][clusterID]
    fofgroups['subfind_tab']['Subhalo']['StellarVelDisp'] = fofgroups['subfind_tab']['Subhalo']['StellarVelDisp'][clusterID]
    fofgroups['group_tab']['FOF']['CentreOfMass'] = fofgroups['group_tab']['FOF']['CentreOfMass'][clusterID]
    fofgroups['group_tab']['FOF']['GroupLength'] = fofgroups['group_tab']['FOF']['GroupLength'][clusterID]
    fofgroups['group_tab']['FOF']['GroupLengthType'] = fofgroups['group_tab']['FOF']['GroupLengthType'][clusterID]
    fofgroups['group_tab']['FOF']['GroupMassType'] = fofgroups['group_tab']['FOF']['GroupMassType'][clusterID]
    fofgroups['group_tab']['FOF']['GroupOffset'] = fofgroups['group_tab']['FOF']['GroupOffset'][clusterID]
    fofgroups['group_tab']['FOF']['GroupOffsetType'] = fofgroups['group_tab']['FOF']['GroupOffsetType'][clusterID]
    fofgroups['group_tab']['FOF']['Mass'] = fofgroups['group_tab']['FOF']['Mass'][clusterID]

    # Gather all data into a large dictionary
    data_dict = {}
    data_dict['clusterID'] = clusterID
    data_dict['files'] = fofgroups['files']
    data_dict['header'] = fofgroups['header']
    data_dict['subfind_tab'] = fofgroups['subfind_tab']
    data_dict['group_tab'] = fofgroups['group_tab']
    data_dict['mass_DMpart'] = fofgroups['mass_DMpart']

    return data_dict


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def csr_index_matrix(fofgroups: dict) -> dict:

    max_group_id = len(fofgroups['subfind_tab']['FOF']['Group_M_Crit200'])

    GroupNumber = {}    # Input data structure
    csrmatrix = {}       # Output data structure

    with h5.File(fofgroups['files'][2], 'r') as h5file:

        # Create a HYDRO/DMO switch
        if "/PartType0" in h5file:
            part_types = [0, 1, 4]
        else:
            part_types = [1]

        # Loop over particle types (hydro/dmo sensitive)
        counter = 1
        for part_type in part_types:

            # Read in GroupNumber info
            N_particles = fofgroups['header']['subfind_particles']['NumPart_ThisFile'][part_type]
            start, end = split(N_particles)
            GroupNumber[f'PartType{part_type}'] = np.empty(0, dtype=np.int)
            GroupNumber[f'PartType{part_type}'] = np.append(
                GroupNumber[f'PartType{part_type}'],
                np.abs(h5file[f'PartType{part_type}/GroupNumber'][start:end])
            )
            GroupNumber[f'PartType{part_type}'] = np.clip(GroupNumber[f'PartType{part_type}'], 0, max_group_id + 1)

            # Generate the metadata in parallel through MPI
            pprint(f"[+] ({counter}/{len(part_types)}) Computing CSR indexing matrix...")
            csrmatrix[f'PartType{part_type}'] = get_indices_sparse(GroupNumber[f'PartType{part_type}'])
            del csrmatrix[f'PartType{part_type}'][0]
            del csrmatrix[f'PartType{part_type}'][-1]
            counter += 1

    return csrmatrix


def particle_index_from_csrm(fofgroup: dict, particle_type: int, csrm: dict) -> np.ndarray:

    N_particles = fofgroup['header']['subfind_particles']['NumPart_ThisFile'][particle_type]
    start, _ = split(N_particles)
    idx = fofgroup['clusterID']
    particle_index = csrm[f'PartType{particle_type}'][idx][0] + start
    # particle_index = commune(particle_index)
    return particle_index


def fof_particles(fofgroup: dict, csrm: dict) -> dict:
    # pprint(f"[+] Find particle information...")

    # Conversion factors
    conv_mass = 1e10 / fofgroup['header']['subfind_particles']['HubbleParam']
    conv_length = fofgroup['header']['subfind_particles']['ExpansionFactor'] / fofgroup['header']['subfind_particles']['HubbleParam']
    conv_density = 1e10 * fofgroup['header']['subfind_particles']['HubbleParam'] ** 2 / fofgroup['header']['subfind_particles']['ExpansionFactor'] ** 3
    conv_velocity = np.sqrt(fofgroup['header']['subfind_particles']['ExpansionFactor'])
    conv_starFormationRate = 1e10 * fofgroup['header']['subfind_particles']['HubbleParam'] ** 2 / fofgroup['header']['subfind_particles']['ExpansionFactor'] ** 3
    conv_time = 3.08568e+19

    # Units
    unit_mass = unyt.Solar_Mass
    unit_length = unyt.Mpc
    unit_density = unyt.Solar_Mass / unyt.Mpc ** 3
    unit_velocity = unyt.km / unyt.s
    unit_starFormationRate = unyt.Solar_Mass / (unyt.year * unyt.Mpc ** 3)

    subfind_particle_data = {}

    with h5.File(fofgroup['files'][2], 'r') as h5file:

        # Create a HYDRO/DMO switch
        is_hydro = "/PartType0" in h5file

        if is_hydro:
            subfind_particle_data[f'PartType0'] = {}
            subfind_particle_data[f'PartType1'] = {}
            subfind_particle_data[f'PartType4'] = {}

            particle_idx0 = particle_index_from_csrm(fofgroup, 0, csrm)
            particle_idx1 = particle_index_from_csrm(fofgroup, 1, csrm)
            particle_idx4 = particle_index_from_csrm(fofgroup, 4, csrm)

            # Initialise empty arrays on all cores
            subfind_particle_data['PartType0']['Coordinates'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType0']['Density'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType0']['GroupNumber'] = np.empty(0, dtype=np.int)
            # subfind_particle_data['PartType0']['HostHalo_TVir_Mass'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType0']['InternalEnergy'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType0']['Mass'] = np.empty(0, dtype=np.float)
            # subfind_particle_data['PartType0']['Metallicity'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType0']['OnEquationOfState'] = np.empty(0, dtype=np.float)
            # subfind_particle_data['PartType0']['ParticleIDs'] = np.empty(0, dtype=np.int)
            # subfind_particle_data['PartType0']['SmoothedMetallicity'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType0']['SmoothingLength'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType0']['StarFormationRate'] = np.empty(0, dtype=np.float)
            # subfind_particle_data['PartType0']['SubGroupNumber'] = np.empty(0, dtype=np.int)
            subfind_particle_data['PartType0']['Temperature'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType0']['Velocity'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType1']['Coordinates'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType1']['GroupNumber'] = np.empty(0, dtype=np.int)
            # subfind_particle_data['PartType1']['ParticleIDs'] = np.empty(0, dtype=np.int)
            # subfind_particle_data['PartType1']['SubGroupNumber'] = np.empty(0, dtype=np.int)
            subfind_particle_data['PartType1']['Velocity'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType4']['Coordinates'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType4']['Density'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType4']['GroupNumber'] = np.empty(0, dtype=np.int)
            # subfind_particle_data['PartType4']['HostHalo_TVir'] = np.empty(0, dtype=np.float)
            # subfind_particle_data['PartType4']['HostHalo_TVir_Mass'] = np.empty(0, dtype=np.float)
            # subfind_particle_data['PartType4']['InitialMass'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType4']['Mass'] = np.empty(0, dtype=np.float)
            # subfind_particle_data['PartType4']['Metallicity'] = np.empty(0, dtype=np.float)
            # subfind_particle_data['PartType4']['ParticleIDs'] = np.empty(0, dtype=np.int)
            subfind_particle_data['PartType4']['SmoothingLength'] = np.empty(0, dtype=np.float)
            subfind_particle_data['PartType4']['StellarFormationTime'] = np.empty(0, dtype=np.float)
            # subfind_particle_data['PartType4']['SubGroupNumber'] = np.empty(0, dtype=np.int)
            subfind_particle_data['PartType4']['Velocity'] = np.empty(0, dtype=np.float)

            # Fill arrays in every core with a chunk of the data
            subfind_particle_data['PartType0']['Coordinates'] = np.append(subfind_particle_data['PartType0']['Coordinates'], h5file['PartType0/Coordinates'][particle_idx0])
            subfind_particle_data['PartType0']['Density'] = np.append(subfind_particle_data['PartType0']['Density'], h5file['PartType0/Density'][particle_idx0])
            subfind_particle_data['PartType0']['GroupNumber'] = np.append(subfind_particle_data['PartType0']['GroupNumber'], h5file['PartType0/GroupNumber'][particle_idx0])
            # subfind_particle_data['PartType0']['HostHalo_TVir_Mass'] = np.append(subfind_particle_data['PartType0']['HostHalo_TVir_Mass'], h5file['PartType0/HostHalo_TVir_Mass'][particle_idx0])
            subfind_particle_data['PartType0']['InternalEnergy'] = np.append(subfind_particle_data['PartType0']['InternalEnergy'], h5file['PartType0/InternalEnergy'][particle_idx0])
            subfind_particle_data['PartType0']['Mass'] = np.append(subfind_particle_data['PartType0']['Mass'], h5file['PartType0/Mass'][particle_idx0])
            # subfind_particle_data['PartType0']['Metallicity'] = np.append(subfind_particle_data['PartType0']['Metallicity'], h5file['PartType0/Metallicity'][particle_idx0])
            subfind_particle_data['PartType0']['OnEquationOfState'] = np.append(subfind_particle_data['PartType0']['OnEquationOfState'], h5file['PartType0/OnEquationOfState'][particle_idx0])
            # subfind_particle_data['PartType0']['ParticleIDs'] = np.append(subfind_particle_data['PartType0']['ParticleIDs'], h5file['PartType0/ParticleIDs'][particle_idx0])
            # subfind_particle_data['PartType0']['SmoothedMetallicity'] = np.append(subfind_particle_data['PartType0']['SmoothedMetallicity'], h5file['PartType0/SmoothedMetallicity'][particle_idx0])
            subfind_particle_data['PartType0']['SmoothingLength'] = np.append(subfind_particle_data['PartType0']['SmoothingLength'], h5file['PartType0/SmoothingLength'][particle_idx0])
            subfind_particle_data['PartType0']['StarFormationRate'] = np.append(subfind_particle_data['PartType0']['StarFormationRate'], h5file['PartType0/StarFormationRate'][particle_idx0])
            # subfind_particle_data['PartType0']['SubGroupNumber'] = np.append(subfind_particle_data['PartType0']['SubGroupNumber'], h5file['PartType0/SubGroupNumber'][particle_idx0])
            subfind_particle_data['PartType0']['Temperature'] = np.append(subfind_particle_data['PartType0']['Temperature'], h5file['PartType0/Temperature'][particle_idx0])
            subfind_particle_data['PartType0']['Velocity'] = np.append(subfind_particle_data['PartType0']['Velocity'], h5file['PartType0/Velocity'][particle_idx0])
            subfind_particle_data['PartType1']['Coordinates'] = np.append(subfind_particle_data['PartType1']['Coordinates'], h5file['PartType1/Coordinates'][particle_idx1])
            subfind_particle_data['PartType1']['GroupNumber'] = np.append(subfind_particle_data['PartType1']['GroupNumber'], h5file['PartType1/GroupNumber'][particle_idx1])
            # subfind_particle_data['PartType1']['ParticleIDs'] = np.append(subfind_particle_data['PartType1']['ParticleIDs'], h5file['PartType1/ParticleIDs'][particle_idx1])
            # subfind_particle_data['PartType1']['SubGroupNumber'] = np.append(subfind_particle_data['PartType1']['SubGroupNumber'], h5file['PartType1/SubGroupNumber'][particle_idx1])
            subfind_particle_data['PartType1']['Velocity'] = np.append(subfind_particle_data['PartType1']['Velocity'], h5file['PartType1/Velocity'][particle_idx1])
            subfind_particle_data['PartType4']['Coordinates'] = np.append(subfind_particle_data['PartType4']['Coordinates'], h5file['PartType4/Coordinates'][particle_idx4])
            subfind_particle_data['PartType4']['Density'] = np.append(subfind_particle_data['PartType4']['Density'], h5file['PartType4/Density'][particle_idx4])
            subfind_particle_data['PartType4']['GroupNumber'] = np.append(subfind_particle_data['PartType4']['GroupNumber'], h5file['PartType4/GroupNumber'][particle_idx4])
            # subfind_particle_data['PartType4']['HostHalo_TVir'] = np.append(subfind_particle_data['PartType4']['HostHalo_TVir'], h5file['PartType4/HostHalo_TVir'][particle_idx4])
            # subfind_particle_data['PartType4']['HostHalo_TVir_Mass'] = np.append(subfind_particle_data['PartType4']['HostHalo_TVir_Mass'], h5file['PartType4/HostHalo_TVir_Mass'][particle_idx4])
            # subfind_particle_data['PartType4']['InitialMass'] = np.append(subfind_particle_data['PartType4']['InitialMass'], h5file['PartType4/InitialMass'][particle_idx4])
            subfind_particle_data['PartType4']['Mass'] = np.append(subfind_particle_data['PartType4']['Mass'], h5file['PartType4/Mass'][particle_idx4])
            # subfind_particle_data['PartType4']['Metallicity'] = np.append(subfind_particle_data['PartType4']['Metallicity'], h5file['PartType4/Metallicity'][particle_idx4])
            # subfind_particle_data['PartType4']['ParticleIDs'] = np.append(subfind_particle_data['PartType4']['ParticleIDs'], h5file['PartType4/ParticleIDs'][particle_idx4])
            subfind_particle_data['PartType4']['SmoothingLength'] = np.append(subfind_particle_data['PartType4']['SmoothingLength'], h5file['PartType4/SmoothingLength'][particle_idx4])
            subfind_particle_data['PartType4']['StellarFormationTime'] = np.append(subfind_particle_data['PartType4']['StellarFormationTime'], h5file['PartType4/StellarFormationTime'][particle_idx4])
            # subfind_particle_data['PartType4']['SubGroupNumber'] = np.append(subfind_particle_data['PartType4']['SubGroupNumber'], h5file['PartType4/SubGroupNumber'][particle_idx4])
            subfind_particle_data['PartType4']['Velocity'] = np.append(subfind_particle_data['PartType4']['Velocity'], h5file['PartType4/Velocity'][particle_idx4])

            # Gather all data from cores into the same array and assign units
            subfind_particle_data['PartType0']['Coordinates'] = commune(subfind_particle_data['PartType0']['Coordinates'].reshape(-1, 1)).reshape(-1, 3) * conv_length * unit_length
            subfind_particle_data['PartType0']['Density'] = commune(subfind_particle_data['PartType0']['Density']) * conv_density * unit_density
            subfind_particle_data['PartType0']['GroupNumber'] = commune(subfind_particle_data['PartType0']['GroupNumber'])
            # subfind_particle_data['PartType0']['HostHalo_TVir_Mass'] = commune(subfind_particle_data['PartType0']['HostHalo_TVir_Mass']) * conv_mass * unit_mass
            subfind_particle_data['PartType0']['InternalEnergy'] = commune(subfind_particle_data['PartType0']['InternalEnergy'])
            subfind_particle_data['PartType0']['Mass'] = commune(subfind_particle_data['PartType0']['Mass']) * conv_mass * unit_mass
            # subfind_particle_data['PartType0']['Metallicity'] = commune(subfind_particle_data['PartType0']['Metallicity'])
            subfind_particle_data['PartType0']['OnEquationOfState'] = commune(subfind_particle_data['PartType0']['OnEquationOfState'])
            # subfind_particle_data['PartType0']['ParticleIDs'] = commune(subfind_particle_data['PartType0']['ParticleIDs'])
            # subfind_particle_data['PartType0']['SmoothedMetallicity'] = commune(subfind_particle_data['PartType0']['SmoothedMetallicity'])
            subfind_particle_data['PartType0']['SmoothingLength'] = commune(subfind_particle_data['PartType0']['SmoothingLength']) * conv_length * unit_length
            subfind_particle_data['PartType0']['StarFormationRate'] = commune(subfind_particle_data['PartType0']['StarFormationRate']) * conv_starFormationRate * unit_starFormationRate
            # subfind_particle_data['PartType0']['SubGroupNumber'] = commune(subfind_particle_data['PartType0']['SubGroupNumber'])
            subfind_particle_data['PartType0']['Temperature'] = commune(subfind_particle_data['PartType0']['Temperature']) * unyt.K
            subfind_particle_data['PartType0']['Velocity'] = commune(subfind_particle_data['PartType0']['Velocity'].reshape(-1, 1)).reshape(-1, 3) * conv_velocity * unit_velocity
            subfind_particle_data['PartType1']['Coordinates'] = commune(subfind_particle_data['PartType1']['Coordinates'].reshape(-1, 1)).reshape(-1, 3) * conv_length * unit_length
            subfind_particle_data['PartType1']['GroupNumber'] = commune(subfind_particle_data['PartType1']['GroupNumber'])
            # subfind_particle_data['PartType1']['ParticleIDs'] = commune(subfind_particle_data['PartType1']['ParticleIDs'])
            # subfind_particle_data['PartType1']['SubGroupNumber'] = commune(subfind_particle_data['PartType1']['SubGroupNumber'])
            subfind_particle_data['PartType1']['Velocity'] = commune(subfind_particle_data['PartType1']['Velocity'].reshape(-1, 1)).reshape(-1, 3) * conv_velocity * unit_velocity
            subfind_particle_data['PartType4']['Coordinates'] = commune(subfind_particle_data['PartType4']['Coordinates'].reshape(-1, 1)).reshape(-1, 3) * conv_length * unit_length
            subfind_particle_data['PartType4']['Density'] = commune(subfind_particle_data['PartType4']['Density']) * conv_density * unit_density
            subfind_particle_data['PartType4']['GroupNumber'] = commune(subfind_particle_data['PartType4']['GroupNumber'])
            # subfind_particle_data['PartType4']['HostHalo_TVir'] = commune(subfind_particle_data['PartType4']['HostHalo_TVir']) * unyt.K
            # subfind_particle_data['PartType4']['HostHalo_TVir_Mass'] = commune(subfind_particle_data['PartType4']['HostHalo_TVir_Mass']) * conv_mass * unit_mass
            # subfind_particle_data['PartType4']['InitialMass'] = commune(subfind_particle_data['PartType4']['InitialMass']) * conv_mass * unit_mass
            subfind_particle_data['PartType4']['Mass'] = commune(subfind_particle_data['PartType4']['Mass']) * conv_mass * unit_mass
            # subfind_particle_data['PartType4']['Metallicity'] = commune(subfind_particle_data['PartType4']['Metallicity'])
            # subfind_particle_data['PartType4']['ParticleIDs'] = commune(subfind_particle_data['PartType4']['ParticleIDs'])
            subfind_particle_data['PartType4']['SmoothingLength'] = commune(subfind_particle_data['PartType4']['SmoothingLength']) * conv_length * unit_length
            subfind_particle_data['PartType4']['StellarFormationTime'] = commune(subfind_particle_data['PartType4']['StellarFormationTime']) * (conv_time * unyt.s).to('Gyr')
            # subfind_particle_data['PartType4']['SubGroupNumber'] = commune(subfind_particle_data['PartType4']['SubGroupNumber'])
            subfind_particle_data['PartType4']['Velocity'] = commune(subfind_particle_data['PartType4']['Velocity'].reshape(-1, 1)).reshape(-1, 3) * conv_velocity * unit_velocity

        else:

            subfind_particle_data[f'PartType1'] = {}
            particle_idx1 = particle_index_from_csrm(fofgroup, 1, csrm)

            # Initialise empty arrays on all cores
            subfind_particle_data['PartType1']['Coordinates'] = np.empty(0, dtype=np.float32)
            subfind_particle_data['PartType1']['GroupNumber'] = np.empty(0, dtype=np.int)
            subfind_particle_data['PartType1']['ParticleIDs'] = np.empty(0, dtype=np.int)
            subfind_particle_data['PartType1']['SubGroupNumber'] = np.empty(0, dtype=np.int)
            subfind_particle_data['PartType1']['Velocity'] = np.empty(0, dtype=np.float32)

            # Fill arrays in every core with a chunk of the data
            subfind_particle_data['PartType1']['Coordinates'] = h5file['PartType1/Coordinates'][particle_idx1]
            subfind_particle_data['PartType1']['GroupNumber'] = h5file['PartType1/GroupNumber'][particle_idx1]
            subfind_particle_data['PartType1']['ParticleIDs'] = h5file['PartType1/ParticleIDs'][particle_idx1]
            subfind_particle_data['PartType1']['SubGroupNumber'] = h5file['PartType1/SubGroupNumber'][particle_idx1]
            subfind_particle_data['PartType1']['Velocity'] = h5file['PartType1/Velocity'][particle_idx1]

            # Gather all data from cores into the same array and assign units
            subfind_particle_data['PartType1']['Coordinates'] = commune(subfind_particle_data['PartType1']['Coordinates'].reshape(-1, 1)).reshape(-1,3) * conv_length * unit_length
            subfind_particle_data['PartType1']['GroupNumber'] = commune(subfind_particle_data['PartType1']['GroupNumber'])
            subfind_particle_data['PartType1']['ParticleIDs'] = commune(subfind_particle_data['PartType1']['ParticleIDs'])
            subfind_particle_data['PartType1']['SubGroupNumber'] = commune(subfind_particle_data['PartType1']['SubGroupNumber'])
            subfind_particle_data['PartType1']['Velocity'] = commune(subfind_particle_data['PartType1']['Velocity'].reshape(-1, 1)).reshape(-1,3) * conv_velocity * unit_velocity

        for pt in subfind_particle_data:

            if pt.startswith('PartType'):

                # Periodic boundary wrapping of particle coordinates
                coords = subfind_particle_data[pt]['Coordinates']
                boxsize = fofgroup['header']['subfind_particles']['BoxSize'] * conv_length * unit_length
                cop = fofgroup['subfind_tab']['FOF']['GroupCentreOfPotential']
                r200 = fofgroup['subfind_tab']['FOF']['Group_R_Crit200']
                for coord_axis in range(3):
                    # Right boundary
                    if cop[coord_axis] + 10 * r200 > boxsize:
                        beyond_index = np.where(coords[:, coord_axis] < boxsize / 2)[0]
                        coords[beyond_index, coord_axis] += boxsize
                    # Left boundary
                    elif cop[coord_axis] - 10 * r200 < 0.:
                        beyond_index = np.where(coords[:, coord_axis] > boxsize / 2)[0]
                        coords[beyond_index, coord_axis] -= boxsize

                subfind_particle_data[pt]['Coordinates'] = coords

        # Gather all data into a large dictionary
        data_dict = {}
        data_dict['files'] = fofgroup['files']
        data_dict['header'] = fofgroup['header']
        data_dict['clusterID'] = fofgroup['clusterID']
        data_dict['subfind_tab'] = fofgroup['subfind_tab']
        data_dict['group_tab'] = fofgroup['group_tab']
        data_dict['subfind_particles'] = subfind_particle_data
        data_dict['mass_DMpart'] = fofgroup['mass_DMpart']

        return data_dict



