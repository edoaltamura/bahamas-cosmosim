import read

simulation_type = 'hydro'
redshift = 'z003p000'
cluster_id = 1

# -------------------------------------------------------------------- #

files = read.find_files(simulation_type, redshift)
halo_catalogue = read.fof_groups(files)
csrm = read.csr_index_matrix(halo_catalogue)
fof = read.fof_group(cluster_id, halo_catalogue)
cluster_dict = read.fof_particles(fof, csrm)
# cluster_dict = read.snapshot_data(files)

read.pprint("Dark matter particle mass:", cluster_dict['mass_DMpart'])

for key in cluster_dict['subfind_tab']['FOF']:
    read.pprint(f"subfind_tab.FOF.{key:<30s}", cluster_dict['subfind_tab']['FOF'][key])

for key in cluster_dict['subfind_tab']['Subhalo']:
    read.pprint(f"subfind_tab.Subhalo.{key:<30s}", cluster_dict['subfind_tab']['Subhalo'][key])

for key in cluster_dict['group_tab']['FOF']:
    read.pprint(f"group_tab.FOF.{key:<30s}", cluster_dict['group_tab']['FOF'][key])

for pt in ['0', '1', '4']:
    for key in cluster_dict['subfind_particles'][f'PartType{pt}']:
        read.pprint(f"PartType{pt:s}.{key:<30s}", cluster_dict['subfind_particles'][f'PartType{pt}'][key])

for pt in ['0', '1', '4']:
    for key in cluster_dict['snaps'][f'PartType{pt}']:
        read.pprint(f"PartType{pt:s}.{key:<30s}", cluster_dict['snaps'][f'PartType{pt}'][key])