import read

files = read.find_files('hydro', 'z000p000')
print(read.header('hydro', files))