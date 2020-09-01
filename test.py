import read

files = read.find_files('hydro', 'z003p000')
header = read.get_header(files)
fofs = read.fof_groups(files, header)
read.pprint(header.data.subfind_particles.MassTable)

for key in fofs.data.subfind_tab.FOF:
    read.pprint(f"{key:<30s} {len(fofs.data.subfind_tab.FOF[key])}", fofs.data.subfind_tab.FOF[key][:2])

for key in fofs.data.subfind_tab.Subhalo:
    read.pprint(f"{key:<30s} {len(fofs.data.subfind_tab.Subhalo[key])}", fofs.data.subfind_tab.Subhalo[key][:2])

for key in fofs.data.group_tab.FOF:
    read.pprint(f"{key:<30s} {len(fofs.data.group_tab.FOF[key])}", fofs.data.group_tab.FOF[key][:2])

fof = read.fof_group(0, fofs)
for key in fof.data.subfind_tab.FOF:
    read.pprint(f"{key:<30s}", fof.data.subfind_tab.FOF[key])

for key in fof.data.subfind_tab.Subhalo:
    read.pprint(f"{key:<30s}", fof.data.subfind_tab.Subhalo[key])

for key in fof.data.group_tab.FOF:
    read.pprint(f"{key:<30s}", fof.data.group_tab.FOF[key])

particles = read.fof_particles(fof)
for pt in ['0', '1', '4']:
    for key in particles.data.subfind_particles[f'PartType{pt}']:
        read.pprint(f"PartType{pt:s}\t{key:<30s}", particles.data.subfind_particles[f'PartType{pt}'][key])
