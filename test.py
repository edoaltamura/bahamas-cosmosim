import read

files = read.find_files('hydro', 'z000p000')
header = read.get_header(files)
fof = read.fof_groups(files)
read.pprint(header.data.subfind_particles.MassTable)
read.pprint(len(fof.data.subfind_tab.FOF.Group_M_Crit200), fof.data.subfind_tab.FOF.Group_M_Crit200)
read.pprint(len(fof.data.subfind_tab.FOF.GroupOffset), fof.data.subfind_tab.FOF.GroupOffset)