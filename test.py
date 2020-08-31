import read

files = read.find_files('hydro', 'z000p000')
header = read.get_header(files)
fof = read.fof_groups(files)
print(header.data.subfind_particles.MassTable)
print(fof.data.subfind_tab_data.FOF)