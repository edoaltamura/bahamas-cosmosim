# Classes and methods for generating property reports gievn cluster
# data objects from the read.py package.

import numpy as np
import unyt
import yaml
from typing import Tuple
from swiftsimio.visualisation.projection import scatter_parallel as scatter
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

import read

ksz_const = - unyt.thompson_cross_section / 1.16 / unyt.speed_of_light / unyt.proton_mass
tsz_const = unyt.thompson_cross_section * unyt.boltzmann_constant / 1.16 / \
            unyt.speed_of_light ** 2 / unyt.proton_mass / unyt.electron_mass

class Properties:

    def __init__(self, cluster_data) -> None:

        self.data = cluster_data

        # Default parameters
        self.display_live = False
        self.output_to_file = True
        self.plot_limits_scale = 'R500crit'
        self.plot_limits = [-5., 5., -5., 5.]
        self.resolution = 512
        self.hot_gas_temperature_threshold = 1.e5

        self.basename = None
        self.subdir = None


    def set_dm_particles(self) -> None:

        boxsize = self.data.boxsize
        coord = self.data.subfind_particles[f'PartType1']['Coordinates']
        for i in [0, 1, 2]:
            if np.min(coord[:, i]) < 0:
                coord[:, i] += boxsize / 2
            elif np.max(coord[:, i]) > boxsize:
                coord[:, i] -= boxsize / 2
        smoothing_lengths = generate_smoothing_lengths(
            coord,
            boxsize,
            kernel_gamma=1.8,
            neighbours=57,
            speedup_fac=3,
            dimension=3,
        )

        self.data.subfind_particles['PartType1']['SmoothingLength'] = smoothing_lengths
        masses = np.ones_like(coord[:, 0].value, dtype=np.float32) * self.data.mass_DMpart
        self.data.subfind_particles['PartType1']['Mass'] = masses
        return