import read

files = read.find_files('hydro', 'z000p000')
print(files[0][0], files[0][-1])
print(files[1][0], files[1][-1])
print(files[3][0], files[3][-1])