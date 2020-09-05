import read

simulation_type = 'hydro'
redshift = 'z003p000'
cluster_id = 0

# -------------------------------------------------------------------- #

files = read.find_files(simulation_type,redshift)
header = read.get_header(files)
fofs = read.fof_groups(files, header)
csrm = read.csr_index_matrix(files)
fof = read.fof_group(cluster_id, fofs)

read.pprint("Dark matter particle mass:", fofs.data.mass_DMpart)
read.pprint(csrm.data.PartType0.csrmatrix[:4])

# for key in fofs.data.subfind_tab.FOF:
#     read.pprint(f"subfind_tab.FOF.{key:<30s} {len(fofs.data.subfind_tab.FOF[key])}", fofs.data.subfind_tab.FOF[key][:2])
#
# for key in fofs.data.subfind_tab.Subhalo:
#     read.pprint(f"subfind_tab.Subhalo.{key:<30s} {len(fofs.data.subfind_tab.Subhalo[key])}", fofs.data.subfind_tab.Subhalo[key][:2])
#
# for key in fofs.data.group_tab.FOF:
#     read.pprint(f"group_tab.FOF.{key:<30s} {len(fofs.data.group_tab.FOF[key])}", fofs.data.group_tab.FOF[key][:2])

for key in fof.data.subfind_tab.FOF:
    read.pprint(f"subfind_tab.FOF.{key:<30s}", fof.data.subfind_tab.FOF[key])

for key in fof.data.subfind_tab.Subhalo:
    read.pprint(f"subfind_tab.Subhalo.{key:<30s}", fof.data.subfind_tab.Subhalo[key])

for key in fof.data.group_tab.FOF:
    read.pprint(f"group_tab.FOF.{key:<30s}", fof.data.group_tab.FOF[key])

particles = read.fof_particles(fof, csrm)
for pt in ['0', '1', '4']:
    for key in particles.data.subfind_particles[f'PartType{pt}']:
        read.pprint(f"PartType{pt:s}.{key:<30s}", particles.data.subfind_particles[f'PartType{pt}'][key])
    # read.pprint(f"\nPartType{pt:s}.GroupNumber\n", particles.data.subfind_particles[f'PartType{pt}']['GroupNumber'])
    # read.pprint(f"\nPartType{pt:s}.Coordinates\n", particles.data.subfind_particles[f'PartType{pt}']['Coordinates'])
