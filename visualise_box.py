import numpy as np
import copy
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from swiftsimio.visualisation.projection import scatter_parallel as scatter
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths

import read

simulation_type = 'hydro'
redshift = 'z003p000'
resolution = 1024

# -------------------------------------------------------------------- #

files = read.find_files(simulation_type, redshift)
# halo_catalogue = read.fof_groups(files)
snapshot = read.snapshot_data(files)['snaps']

x_max = np.max(snapshot['PartType0']['Coordinates'][:, 0])
x_min = np.min(snapshot['PartType0']['Coordinates'][:, 0])
y_max = np.max(snapshot['PartType0']['Coordinates'][:, 1])
y_min = np.min(snapshot['PartType0']['Coordinates'][:, 1])
x_range = x_max - x_min
y_range = y_max - y_min
x = (snapshot['PartType0']['Coordinates'][:, 0] - x_min) / x_range
y = (snapshot['PartType0']['Coordinates'][:, 1] - y_min) / y_range
h = snapshot['PartType0']['SmoothingLength'] / x_range

# Gather and handle coordinates to be processed
x = np.asarray(x.value, dtype=np.float64)
y = np.asarray(y.value, dtype=np.float64)
m = np.asarray(snapshot['PartType0']['Density'].value, dtype=np.float32)
h = np.asarray(h.value, dtype=np.float32)
read.pprint(f'Computing map ({resolution} x {resolution})')
smoothed_map = scatter(x=x, y=y, m=m, h=h, res=resolution).T
smoothed_map = np.ma.masked_where(np.abs(smoothed_map) < 1.e-9, smoothed_map)

fig, axes = plt.subplots()
cmap = copy.copy(plt.get_cmap('twilight'))
cmap.set_under('black')
axes.axis("off")
axes.set_aspect("equal")
axes.imshow(
    smoothed_map.T,
    norm=LogNorm(),
    cmap=cmap,
    origin="lower",
    extent=[x_min, x_max, y_min, y_max]
)
plt.show()
