import read

simulation_type = 'hydro'
redshift = 'z003p000'
cluster_id = 0

# -------------------------------------------------------------------- #
# Examples:
# fofs.data.subfind_tab.FOF.Group_M_Crit500
# fofs.data.group_tab.FOF.CentreOfMass
# fofs.data.mass_DMpart
# group_data.data.subfind_tab.FOF.Group_M_Crit500
# particle_data.data.subfind_particles.PartType0.Coordinates
# -------------------------------------------------------------------- #

files = read.find_files(simulation_type,redshift)
header = read.get_header(files)
fofs = read.fof_groups(files, header)
csrm = read.csr_index_matrix(files, fofs)
group_data = read.fof_group(cluster_id, fofs).data

read.pprint("Dark matter particle mass:", fofs.data.mass_DMpart)

for key in group_data.subfind_tab.FOF:
    read.pprint(f"subfind_tab.FOF.{key:<30s}", group_data.subfind_tab.FOF[key])

for key in group_data.subfind_tab.Subhalo:
    read.pprint(f"subfind_tab.Subhalo.{key:<30s}", group_data.subfind_tab.Subhalo[key])

for key in group_data.group_tab.FOF:
    read.pprint(f"group_tab.FOF.{key:<30s}", group_data.group_tab.FOF[key])

particle_data = read.fof_particles(group_data, csrm).data.subfind_particles
for pt in ['0', '1', '4']:
    for key in particle_data[f'PartType{pt}']:
        read.pprint(f"PartType{pt:s}.{key:<30s}", particle_data[f'PartType{pt}'][key])
