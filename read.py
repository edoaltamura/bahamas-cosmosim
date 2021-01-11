import os
import numpy as np
import h5py as h5
import unyt
from scipy.sparse import csr_matrix
from copy import deepcopy
from mpi4py import MPI
from warnings import warn

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

from metadata import Metadata, AttrDict


def pprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def wwarn(*args, **kwargs):
    if rank == 0:
        warn(*args, **kwargs)


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
    conv_mass = 1.e10 / header['HubbleParam']
    conv_length = header['ExpansionFactor'] / header['HubbleParam']
    conv_density = 1.e10 * header['HubbleParam'] ** 2 / header['ExpansionFactor'] ** 3
    conv_velocity = np.sqrt(header['ExpansionFactor'])
    conv_starFormationRate = 1.e10 * header['HubbleParam'] ** 2 / header['ExpansionFactor'] ** 3

    # Units
    unit_mass = unyt.Solar_Mass
    unit_length = unyt.Mpc
    unit_density = unyt.Solar_Mass / unyt.Mpc ** 3
    unit_velocity = unyt.km / unyt.s
    unit_starFormationRate = unyt.Solar_Mass / (unyt.year * unyt.Mpc ** 3)

    pprint(f"[+] Find groups information...")

    fof_fields = [
        'FirstSubhaloID', 'GroupCentreOfPotential', 'GroupLength', 'GroupMass', 'GroupOffset',
        'Group_M_Crit200', 'Group_M_Crit2500', 'Group_M_Crit500', 'Group_M_Mean200', 'Group_M_Mean2500',
        'Group_M_Mean500', 'Group_M_TopHat200', 'Group_R_Crit200', 'Group_R_Crit2500', 'Group_R_Crit500',
        'Group_R_Mean200', 'Group_R_Mean2500', 'Group_R_Mean500', 'Group_R_TopHat200', 'NumOfSubhalos'
    ]
    subhalo_fields = [
        'CentreOfMass', 'CentreOfPotential', 'GasSpin', 'GroupNumber', 'HalfMassProjRad', 'HalfMassRad',
        'IDMostBound', 'SubLength', 'SubOffset', 'Velocity', 'Vmax', 'VmaxRadius'
    ]

    # Find eagle subfind tab hdf5 internal paths
    subfind_tab_data = {}
    subfind_tab_data['FOF'] = {}
    for fof_field in fof_fields:
        subfind_tab_data['FOF'][fof_field] = np.empty(0)

    subfind_tab_data['Subhalo'] = {}
    for subhalo_field in subhalo_fields:
        subfind_tab_data['Subhalo'][subhalo_field] = np.empty(0)

    # Find subfind group tab hdf5 internal paths
    group_tab_data = {}
    group_tab_data['FOF'] = {}
    group_tab_fields = [
        'CentreOfMass',
        'GroupLength',
        'GroupLengthType',
        'GroupMassType',
        'GroupOffset',
        'GroupOffsetType',
        'Mass',
    ]
    for group_tab_field in group_tab_fields:
        group_tab_data['FOF'][group_tab_field] = np.empty(0)

    st, fh = split(len(files[0]))
    for x in range(st, fh, 1):

        with h5.File(files[0][x], 'r') as f:

            for fof_field in fof_fields:
                field_data_handle = f[f'FOF/{fof_field}']
                subfind_tab_data['FOF'][fof_field] = np.append(
                    subfind_tab_data['FOF'][fof_field],
                    field_data_handle[:].flatten()
                )

                # Convert FOF fields to the corresponding data type
                subfind_tab_data['FOF'][fof_field] = subfind_tab_data['FOF'][fof_field].astype(
                    field_data_handle.dtype.char
                )

            for subhalo_field in subhalo_fields:
                field_data_handle = f[f'Subhalo/{subhalo_field}']
                subfind_tab_data['Subhalo'][subhalo_field] = np.append(
                    subfind_tab_data['Subhalo'][subhalo_field],
                    field_data_handle[:].flatten()
                )

                # Convert Subhalo fields to the corresponding data type
                subfind_tab_data['Subhalo'][subhalo_field] = subfind_tab_data['Subhalo'][subhalo_field].astype(
                    field_data_handle.dtype.char
                )

    st, fh = split(len(files[1]))
    for x in range(st, fh, 1):

        # Operate on the group data file
        with h5.File(files[1][x], 'r') as f:

            for group_tab_field in group_tab_fields:
                field_data_handle = f[f'FOF/{group_tab_field}']
                group_tab_data['FOF'][group_tab_field] = np.append(
                    group_tab_data['FOF'][group_tab_field],
                    field_data_handle[:].flatten()
                )

                # Convert group data fields to the corresponding data type
                group_tab_data['FOF'][group_tab_field] = group_tab_data['FOF'][group_tab_field].astype(
                    field_data_handle.dtype.char
                )

    # Make a deep copy of the dictionary to MPI-gather data
    _subfind_tab_data = subfind_tab_data.copy()
    _group_tab_data = group_tab_data.copy()

    for fof_field in fof_fields:
        # pprint(subfind_tab_data['FOF'][fof_field])
        filtered_data = subfind_tab_data['FOF'][fof_field]
        _subfind_tab_data['FOF'][fof_field] = commune(filtered_data)

    for subhalo_field in subhalo_fields:
        filtered_data = subfind_tab_data['Subhalo'][subhalo_field]
        _subfind_tab_data['Subhalo'][subhalo_field] = commune(filtered_data)

    for group_tab_field in group_tab_fields:
        filtered_data = group_tab_data['FOF'][group_tab_field]
        _group_tab_data['FOF'][group_tab_field] = commune(filtered_data)

    # Give units to the datasets: FOF Subfind data
    _subfind_tab_data['FOF']['GroupCentreOfPotential'] *= conv_length * unit_length
    _subfind_tab_data['FOF']['GroupMass'] *= conv_mass * unit_mass
    _subfind_tab_data['FOF']['Group_M_Crit200'] *= conv_mass * unit_mass
    _subfind_tab_data['FOF']['Group_M_Crit2500'] *= conv_mass * unit_mass
    _subfind_tab_data['FOF']['Group_M_Crit500'] *= conv_mass * unit_mass
    _subfind_tab_data['FOF']['Group_M_Mean200'] *= conv_mass * unit_mass
    _subfind_tab_data['FOF']['Group_M_Mean2500'] *= conv_mass * unit_mass
    _subfind_tab_data['FOF']['Group_M_Mean500'] *= conv_mass * unit_mass
    _subfind_tab_data['FOF']['Group_M_TopHat200'] *= conv_mass * unit_mass
    _subfind_tab_data['FOF']['Group_R_Crit200'] *= conv_length * unit_length
    _subfind_tab_data['FOF']['Group_R_Crit2500'] *= conv_length * unit_length
    _subfind_tab_data['FOF']['Group_R_Crit500'] *= conv_length * unit_length
    _subfind_tab_data['FOF']['Group_R_Mean200'] *= conv_length * unit_length
    _subfind_tab_data['FOF']['Group_R_Mean2500'] *= conv_length * unit_length
    _subfind_tab_data['FOF']['Group_R_Mean500'] *= conv_length * unit_length
    _subfind_tab_data['FOF']['Group_R_TopHat200'] *= conv_length * unit_length

    # Give units to the datasets: subhalo Subfind data
    _subfind_tab_data['Subhalo']['CentreOfMass'] *= conv_length * unit_length
    _subfind_tab_data['Subhalo']['CentreOfPotential'] *= conv_length * unit_length
    _subfind_tab_data['Subhalo']['HalfMassProjRad'] *= conv_length * unit_length
    _subfind_tab_data['Subhalo']['HalfMassRad'] *= conv_length * unit_length
    _subfind_tab_data['Subhalo']['Velocity'] *= conv_velocity * unit_velocity
    _subfind_tab_data['Subhalo']['Vmax'] *= conv_velocity * unit_velocity
    _subfind_tab_data['Subhalo']['VmaxRadius'] *= conv_length * unit_length

    # Give units to the datasets: FOF group-tab data
    _group_tab_data['FOF']['CentreOfMass'] *= conv_length * unit_length
    _group_tab_data['FOF']['GroupMassType'] *= conv_mass * unit_mass
    _group_tab_data['FOF']['Mass'] *= conv_mass * unit_mass

    # From the FOF dataset, only need to reshape the CoP
    _subfind_tab_data['FOF']['GroupCentreOfPotential'] = \
        _subfind_tab_data['FOF']['GroupCentreOfPotential'].reshape(-1, 3)

    # Reshape datasets from subhalo fields
    for key in [
        'CentreOfMass',
        'CentreOfPotential',
        'GasSpin',
        'Velocity',
    ]:
        _subfind_tab_data['Subhalo'][key] = _subfind_tab_data['Subhalo'][key].reshape(-1, 3)

    # The HalfMassProjRad and HalfMassRad fields have special shape
    _subfind_tab_data['Subhalo']['HalfMassProjRad'] = _subfind_tab_data['Subhalo']['HalfMassProjRad'].reshape(-1, 6)
    _subfind_tab_data['Subhalo']['HalfMassRad'] = _subfind_tab_data['Subhalo']['HalfMassRad'].reshape(-1, 6)

    # Reshape group_tab_data fields that were flattened over MPI
    _group_tab_data['FOF']['CentreOfMass'] = _group_tab_data['FOF']['CentreOfMass'].reshape(-1, 3)
    _group_tab_data['FOF']['GroupLengthType'] = _group_tab_data['FOF']['GroupLengthType'].reshape(-1, 6)
    _group_tab_data['FOF']['GroupMassType'] = _group_tab_data['FOF']['GroupMassType'].reshape(-1, 6)
    _group_tab_data['FOF']['GroupOffsetType'] = _group_tab_data['FOF']['GroupOffsetType'].reshape(-1, 6)

    # Edit the AttrDict object and push the filtered data
    filter_idx = np.where(
        _subfind_tab_data['FOF']['Group_M_Crit500'] > 1.e13
    )[0]

    for category in ['FOF', 'Subhalo']:
        for dataset in _subfind_tab_data[category]:
            _subfind_tab_data[category][dataset] = _subfind_tab_data[category][dataset][filter_idx]

    for dataset in _group_tab_data['FOF']:
        _group_tab_data['FOF'][dataset] = _group_tab_data['FOF'][dataset][filter_idx]

    # Gather all data into a large dictionary
    data_dict = {}
    data_dict['files'] = files
    data_dict['header'] = master_header
    data_dict['subfind_tab'] = _subfind_tab_data
    data_dict['group_tab'] = _group_tab_data
    data_dict['mass_DMpart'] = header['MassTable'][1] * conv_mass * unit_mass

    return data_dict


def fof_group(clusterID: int, fofgroups: dict) -> dict:
    # pprint(f"[+] Find group information for cluster {clusterID}")
    _fofgroups = fofgroups.copy()

    # Filter groups
    for dataset_category in ['subfind_tab', 'group_tab']:
        for object_type in _fofgroups[dataset_category].keys():
            for data_field in _fofgroups[dataset_category][object_type].keys():
                _fofgroups[dataset_category][object_type][data_field] = \
                    fofgroups[dataset_category][object_type][data_field][clusterID]

    # Gather all data into a large dictionary
    data_dict = {}
    data_dict['clusterID'] = clusterID
    data_dict['files'] = _fofgroups['files']
    data_dict['header'] = _fofgroups['header']
    data_dict['subfind_tab'] = _fofgroups['subfind_tab']
    data_dict['group_tab'] = _fofgroups['group_tab']
    data_dict['mass_DMpart'] = _fofgroups['mass_DMpart']

    return data_dict


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def csr_index_matrix(fofgroups: dict) -> dict:
    max_group_id = len(fofgroups['subfind_tab']['FOF']['Group_M_Crit200'])

    GroupNumber = {}  # Input data structure
    csrmatrix = {}  # Output data structure

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
    particle_index = commune(particle_index)
    return particle_index


def fof_particles(fofgroup: dict, csrm: dict) -> dict:
    # pprint(f"[+] Find particle information...")

    # Conversion factors
    conv_mass = 1.e10 / fofgroup['header']['subfind_particles']['HubbleParam']

    conv_length = fofgroup['header']['subfind_particles']['ExpansionFactor'] / \
                  fofgroup['header']['subfind_particles']['HubbleParam']

    conv_density = 1.e10 * fofgroup['header']['subfind_particles']['HubbleParam'] ** 2 / \
                   fofgroup['header']['subfind_particles']['ExpansionFactor'] ** 3

    conv_velocity = np.sqrt(fofgroup['header']['subfind_particles']['ExpansionFactor'])

    conv_starFormationRate = 1.e10 * fofgroup['header']['subfind_particles']['HubbleParam'] ** 2 / \
                             fofgroup['header']['subfind_particles']['ExpansionFactor'] ** 3

    conv_time = 3.08568e19

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

        gas_fields = [
            'Coordinates',
            'Density',
            'GroupNumber',
            'InternalEnergy',
            'Mass',
            'Metallicity',
            'OnEquationOfState',
            'SmoothedMetallicity',
            'SmoothingLength',
            'StarFormationRate',
            'SubGroupNumber',
            'Temperature',
            'Velocity',
        ]
        dm_fields = [
            'Coordinates',
            'GroupNumber',
            'SubGroupNumber',
            'Velocity',
        ]
        stars_fields = [
            'Mass',
            'Metallicity',
            'SmoothingLength',
            'StellarFormationTime',
            'GroupNumber',
            'SubGroupNumber',
            'Velocity',
        ]

        if is_hydro:

            subfind_particle_data[f'PartType0'] = {}
            subfind_particle_data[f'PartType1'] = {}
            subfind_particle_data[f'PartType4'] = {}

            particle_idx0 = particle_index_from_csrm(fofgroup, 0, csrm)
            particle_idx1 = particle_index_from_csrm(fofgroup, 1, csrm)
            particle_idx4 = particle_index_from_csrm(fofgroup, 4, csrm)

            # pprint('particle_idx0', particle_idx0)
            # pprint('particle_idx1', particle_idx1)
            # pprint('particle_idx4', particle_idx4)

            for field in gas_fields:
                subfind_particle_data['PartType0'][field] = np.empty(0)
                field_data_handle = h5file[f'PartType0/{field}']
                subfind_particle_data['PartType0'][field] = np.append(
                    subfind_particle_data['PartType0'][field],
                    field_data_handle[particle_idx0].flatten()
                )

                # Convert group data fields to the corresponding data type
                subfind_particle_data['PartType0'][field] = subfind_particle_data['PartType0'][field].astype(
                    str(field_data_handle.dtype)
                )

            for field in dm_fields:
                subfind_particle_data['PartType1'][field] = np.empty(0)
                field_data_handle = h5file[f'PartType1/{field}']
                subfind_particle_data['PartType1'][field] = np.append(
                    subfind_particle_data['PartType1'][field],
                    field_data_handle[particle_idx1].flatten()
                )

                # Convert group data fields to the corresponding data type
                subfind_particle_data['PartType1'][field] = subfind_particle_data['PartType1'][field].astype(
                    str(field_data_handle.dtype)
                )

            for field in stars_fields:
                subfind_particle_data['PartType4'][field] = np.empty(0)
                field_data_handle = h5file[f'PartType4/{field}']
                subfind_particle_data['PartType4'][field] = np.append(
                    subfind_particle_data['PartType4'][field],
                    field_data_handle[particle_idx4].flatten()
                )

                # Convert group data fields to the corresponding data type
                subfind_particle_data['PartType4'][field] = subfind_particle_data['PartType4'][field].astype(
                    str(field_data_handle.dtype)
                )

            # Reshape coordinates and velocities
            for particle_type in ['PartType0', 'PartType1', 'PartType4']:
                subfind_particle_data[particle_type]['Coordinates'] = \
                    subfind_particle_data[particle_type]['Coordinates'].reshape(-1, 3)
                subfind_particle_data[particle_type]['Velocity'] = \
                    subfind_particle_data[particle_type]['Velocity'].reshape(-1, 3)

            subfind_particle_data['PartType0']['Coordinates'] *= conv_length * unit_length
            subfind_particle_data['PartType0']['Density'] *= conv_density * unit_density
            subfind_particle_data['PartType0']['Mass'] *= conv_mass * unit_mass
            subfind_particle_data['PartType0']['SmoothingLength'] *= conv_length * unit_length
            subfind_particle_data['PartType0']['StarFormationRate'] *= conv_starFormationRate * unit_starFormationRate
            subfind_particle_data['PartType0']['Temperature'] *= unyt.K
            subfind_particle_data['PartType0']['Velocity'] *= conv_velocity * unit_velocity

            subfind_particle_data['PartType1']['Coordinates'] *= conv_length * unit_length
            subfind_particle_data['PartType1']['Velocity'] *= conv_velocity * unit_velocity

            subfind_particle_data['PartType4']['Coordinates'] *= conv_length * unit_length
            subfind_particle_data['PartType4']['Density'] *= conv_density * unit_density
            subfind_particle_data['PartType4']['Mass'] *= conv_mass * unit_mass
            subfind_particle_data['PartType4']['SmoothingLength'] *= conv_length * unit_length
            subfind_particle_data['PartType4']['StellarFormationTime'] *= (conv_time * unyt.s).to('Gyr')
            subfind_particle_data['PartType4']['Velocity'] *= conv_velocity * unit_velocity

        else:

            subfind_particle_data[f'PartType1'] = {}
            particle_idx1 = particle_index_from_csrm(fofgroup, 1, csrm)

            for field in dm_fields:
                subfind_particle_data['PartType1'][field] = np.empty(0)
                field_data_handle = h5file[f'PartType1/{field}']
                subfind_particle_data['PartType1'][field] = np.append(
                    subfind_particle_data['PartType1'][field],
                    field_data_handle[particle_idx1].flatten()
                )

                # Convert group data fields to the corresponding data type
                subfind_particle_data['PartType1'][field] = subfind_particle_data['PartType1'][field].astype(
                    str(field_data_handle.dtype)
                )

            subfind_particle_data['PartType1']['Coordinates'] = \
                subfind_particle_data['PartType1']['Coordinates'].reshape(-1, 3)
            subfind_particle_data['PartType1']['Velocity'] = \
                subfind_particle_data['PartType1']['Velocity'].reshape(-1, 3)

            subfind_particle_data['PartType1']['Coordinates'] *= conv_length * unit_length
            subfind_particle_data['PartType1']['Velocity'] *= conv_velocity * unit_velocity

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
        data_dict['boxsize'] = boxsize

        return data_dict
    
    
def snapshot_data(fofgroup: dict) -> dict:
    pprint(f"[+] Find snapshot data...")

    header = fofgroup['header']['gadget_snaps']

    # Conversion factors
    conv_mass = 1.e10 / header['HubbleParam']
    conv_length = header['ExpansionFactor'] / header['HubbleParam']
    conv_density = 1.e10 * header['HubbleParam'] ** 2 / header['ExpansionFactor'] ** 3
    conv_velocity = np.sqrt(header['ExpansionFactor'])
    conv_starFormationRate = 1.e10 * header['HubbleParam'] ** 2 / header['ExpansionFactor'] ** 3
    conv_time = 3.08568e19

    # Units
    unit_mass = unyt.Solar_Mass
    unit_length = unyt.Mpc
    unit_density = unyt.Solar_Mass / unyt.Mpc ** 3
    unit_velocity = unyt.km / unyt.s
    unit_starFormationRate = unyt.Solar_Mass / (unyt.year * unyt.Mpc ** 3)

    snap_data = {}
    pprint(f"Scattering {len(fofgroup['files'][3])} files")
    st, fh = split(len(fofgroup['files'][3]))
    for x in range(st, fh, 1):

        # Operate on the group data file
        with h5.File(fofgroup['files'][3][x], 'r') as h5file:

            # Create a HYDRO/DMO switch
            is_hydro = "/PartType0" in h5file

            gas_fields = [
                'Coordinates',
                'Density',
                'InternalEnergy',
                'Mass',
                'Metallicity',
                'OnEquationOfState',
                'SmoothedMetallicity',
                'SmoothingLength',
                'StarFormationRate',
                'Temperature',
                'Velocity',
            ]
            dm_fields = [
                'Coordinates',
                'Velocity',
            ]
            stars_fields = [
                'Mass',
                'Metallicity',
                'SmoothingLength',
                'StellarFormationTime',
                'Velocity',
            ]

            if is_hydro:
                pprint('Detected hydro')

                snap_data[f'PartType0'] = {}
                snap_data[f'PartType1'] = {}
                snap_data[f'PartType4'] = {}

                for field in gas_fields:
                    pprint(f'PartType0/{field}')
                    snap_data['PartType0'][field] = np.empty(0)
                    field_data_handle = h5file[f'PartType0/{field}']
                    snap_data['PartType0'][field] = np.append(
                        snap_data['PartType0'][field],
                        field_data_handle[:].flatten()
                    )

                    # Convert group data fields to the corresponding data type
                    snap_data['PartType0'][field] = snap_data['PartType0'][field].astype(
                        str(field_data_handle.dtype)
                    )

                for field in dm_fields:
                    pprint(f'PartType1/{field}')
                    snap_data['PartType1'][field] = np.empty(0)
                    field_data_handle = h5file[f'PartType1/{field}']
                    snap_data['PartType1'][field] = np.append(
                        snap_data['PartType1'][field],
                        field_data_handle[:].flatten()
                    )

                    # Convert group data fields to the corresponding data type
                    snap_data['PartType1'][field] = snap_data['PartType1'][field].astype(
                        str(field_data_handle.dtype)
                    )

                for field in stars_fields:
                    pprint(f'PartType4/{field}')
                    snap_data['PartType4'][field] = np.empty(0)
                    field_data_handle = h5file[f'PartType4/{field}']
                    snap_data['PartType4'][field] = np.append(
                        snap_data['PartType4'][field],
                        field_data_handle[:].flatten()
                    )

                    # Convert group data fields to the corresponding data type
                    snap_data['PartType4'][field] = snap_data['PartType4'][field].astype(
                        str(field_data_handle.dtype)
                    )


                for field_group, particle_type in zip(
                        [gas_fields, dm_fields, stars_fields],
                        ['PartType0', 'PartType1', 'PartType4']
                ):
                    for field in field_group:
                        pprint(f"Comuning data {particle_type} {field}")
                        snap_data[particle_type][field] = \
                            commune(snap_data[particle_type][field])

                # Reshape coordinates and velocities
                for particle_type in ['PartType0', 'PartType1', 'PartType4']:
                    snap_data[particle_type]['Coordinates'] = \
                        snap_data[particle_type]['Coordinates'].reshape(-1, 3)
                    snap_data[particle_type]['Velocity'] = \
                        snap_data[particle_type]['Velocity'].reshape(-1, 3)

                snap_data['PartType0']['Coordinates'] *= conv_length * unit_length
                snap_data['PartType0']['Density'] *= conv_density * unit_density
                snap_data['PartType0']['Mass'] *= conv_mass * unit_mass
                snap_data['PartType0']['SmoothingLength'] *= conv_length * unit_length
                snap_data['PartType0']['StarFormationRate'] *= conv_starFormationRate * unit_starFormationRate
                snap_data['PartType0']['Temperature'] *= unyt.K
                snap_data['PartType0']['Velocity'] *= conv_velocity * unit_velocity

                snap_data['PartType1']['Coordinates'] *= conv_length * unit_length
                snap_data['PartType1']['Velocity'] *= conv_velocity * unit_velocity

                snap_data['PartType4']['Coordinates'] *= conv_length * unit_length
                snap_data['PartType4']['Density'] *= conv_density * unit_density
                snap_data['PartType4']['Mass'] *= conv_mass * unit_mass
                snap_data['PartType4']['SmoothingLength'] *= conv_length * unit_length
                snap_data['PartType4']['StellarFormationTime'] *= (conv_time * unyt.s).to('Gyr')
                snap_data['PartType4']['Velocity'] *= conv_velocity * unit_velocity

            else:

                snap_data[f'PartType1'] = {}

                for field in dm_fields:
                    snap_data['PartType1'][field] = np.empty(0)
                    field_data_handle = h5file[f'PartType1/{field}']
                    snap_data['PartType1'][field] = np.append(
                        snap_data['PartType1'][field],
                        field_data_handle[...].flatten()
                    )

                    # Convert group data fields to the corresponding data type
                    snap_data['PartType1'][field] = snap_data['PartType1'][field].astype(
                        str(field_data_handle.dtype)
                    )

                snap_data['PartType1']['Coordinates'] = \
                    snap_data['PartType1']['Coordinates'].reshape(-1, 3)
                snap_data['PartType1']['Velocity'] = \
                    snap_data['PartType1']['Velocity'].reshape(-1, 3)

                snap_data['PartType1']['Coordinates'] *= conv_length * unit_length
                snap_data['PartType1']['Velocity'] *= conv_velocity * unit_velocity

        # Gather all data into a large dictionary
        data_dict = {}
        data_dict['files'] = fofgroup['files']
        data_dict['header'] = fofgroup['header']
        data_dict['snaps'] = snap_data

        return data_dict
