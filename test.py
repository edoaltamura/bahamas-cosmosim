import read

files = read.find_files('hydro', 'z000p000')
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
    read.pprint(f"{key:<30s} {len(foff.data.subfind_tab.FOF[key])}", foff.data.subfind_tab.FOF[key])

for key in foff.data.subfind_tab.Subhalo:
    read.pprint(f"{key:<30s} {len(foff.data.subfind_tab.Subhalo[key])}", foff.data.subfind_tab.Subhalo[key])

for key in foff.data.group_tab.FOF:
    read.pprint(f"{key:<30s} {len(foff.data.group_tab.FOF[key])}", foff.data.group_tab.FOF[key])

