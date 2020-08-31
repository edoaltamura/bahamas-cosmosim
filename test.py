import read

files = read.find_files('hydro', 'z000p000')
header = read.header(files)
print(header.MassTable)