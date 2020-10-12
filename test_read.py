import read

simulation_type = 'hydro'
redshift = 'z003p000'
cluster_id = 1

# -------------------------------------------------------------------- #

files = read.find_files(simulation_type, redshift)
fofs = read.fof_groups(files)
csrm = read.csr_index_matrix(fofs)
fof = read.fof_group(cluster_id, fofs)
particle_data = read.fof_particles(fof, csrm)

read.pprint("Dark matter particle mass:", particle_data['mass_DMpart'])

for key in particle_data['group_data']['subfind_tab']['FOF']:
    read.pprint(f"subfind_tab.FOF.{key:<30s}", particle_data['group_data']['subfind_tab']['FOF'][key])

for key in particle_data['group_data']['subfind_tab']['Subhalo']:
    read.pprint(f"subfind_tab.Subhalo.{key:<30s}", particle_data['group_data']['subfind_tab']['Subhalo'][key])

for key in particle_data['group_data']['group_tab']['FOF']:
    read.pprint(f"group_tab.FOF.{key:<30s}", particle_data['group_data']['group_tab']['FOF'][key])


for pt in ['0', '1', '4']:
    for key in particle_data['subfind_particles'][f'PartType{pt}']:
        read.pprint(f"PartType{pt:s}.{key:<30s}", particle_data['subfind_particles'][f'PartType{pt}'][key])
