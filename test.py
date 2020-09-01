import read

files = read.find_files('hydro', 'z003p000')
header = read.get_header(files)
fof = read.fof_groups(files, header)
read.pprint(header.data.subfind_particles.MassTable)

for key in fof.data.subfind_tab.FOF:
    read.pprint(f"{key:<30s} {len(fof.data.subfind_tab.FOF[key])}", fof.data.subfind_tab.FOF[key][:2])

for key in fof.data.subfind_tab.Subhalo:
    read.pprint(f"{key:<30s} {len(fof.data.subfind_tab.Subhalo[key])}", fof.data.subfind_tab.Subhalo[key][:2])

for key in fof.data.group_tab.FOF:
    read.pprint(f"{key:<30s} {len(fof.data.group_tab.FOF[key])}", fof.data.group_tab.FOF[key][:2])

foff = read.fof_group(0, fof)
for key in foff.data.subfind_tab.FOF:
    read.pprint(f"{key:<30s}", foff.data.subfind_tab.FOF[key])

for key in foff.data.subfind_tab.Subhalo:
    read.pprint(f"{key:<30s}", foff.data.subfind_tab.Subhalo[key])

for key in foff.data.group_tab.FOF:
    read.pprint(f"{key:<30s}", foff.data.group_tab.FOF[key])

particles = read.fof_particles(foff)
for pt in ['0', '1', '4']:
    for key in particles.data.subfind_particles[f'PartType{pt}']:
        read.pprint(f"PartType{pt:s}\t{key:<30s}", particles.data.subfind_particles[f'PartType{pt}'][key])
