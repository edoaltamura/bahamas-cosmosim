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
from metadata import AttrDict

ksz_const = - unyt.thompson_cross_section / 1.16 / unyt.speed_of_light / unyt.proton_mass
tsz_const = unyt.thompson_cross_section * unyt.boltzmann_constant / 1.16 / \
            unyt.speed_of_light ** 2 / unyt.proton_mass / unyt.electron_mass


class Mapping:

    def __init__(self, cluster_data: AttrDict) -> None:

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

        self.set_dm_particles()
        self.set_hot_gas()

    # def __parameter_parser(self, param_file: str) -> None:
    #
    #     params = yaml.load(open(param_file))
    #
    #     # Handle default parameters
    #     if 'Default' in params:
    #         assert len(params['Default'].keys()) > 0
    #         if 'output_to_file' in params['Default']:
    #             if params['Default']['output_to_file']:
    #                 assert 'basename' in params['Default'], 'File basename not specified in the parameter file.'
    #                 assert 'subdir' in params['Default'], 'Output directory not specified in the parameter file.'
    #
    #             if params['Default']['display_live']:
    #                 read.wwarn('If you are going to save many figures, you may want to turn off `display_live`.')
    #
    #         # Push parameters onto class
    #         for param in params['Default']:
    #             setattr(self, param, params[param])
    #     else:
    #         read.wwarn('Default parameters do not appear to be specified in the parameter file.')
    #
    #     # Handle particle parameters {partType: allowed_maps}
    #     maps_register = {
    #         'Gas': [
    #             'mass',
    #             'density',
    #             'particle_dot',
    #             'particle_number',
    #             'mass_weighted_temperature',
    #             'metallicity',
    #             'tSZ',
    #             'kSZ',
    #             'rkSZ',
    #         ],
    #         'Dark_matter': [
    #             'mass',
    #             'particle_dot',
    #             'particle_number',
    #         ],
    #         'Stars': [
    #             'mass',
    #             'density',
    #             'particle_dot',
    #             'particle_number',
    #             'metallicity',
    #         ],
    #     }
    #
    #     for particle_species in maps_register:
    #
    #         if particle_species in params and len(params[particle_species].items()) > 0:
    #
    #             # Gather all map info into one dictionary
    #             part_type_info = dict()
    #
    #             # Check allowed map_types
    #             if 'map_type' in params[particle_species]:
    #
    #                 for map_type_call in params[particle_species]['map_type']:
    #                     assert map_type_call in maps_register[particle_species], \
    #                         f"Map type {map_type_call} is not allowed for {particle_species} maps."
    #
    #                 part_type_info['map_type'] = params[particle_species]['map_type']
    #
    #             else:
    #                 read.wwarn(f"`map_type` not detected for {particle_species} particles.")
    #
    #             setattr(self, particle_species, part_type_info)
    #
    #     return
    #
    # @staticmethod
    # def _rotation_align_with_vector(
    #         coordinates: unyt.array, rotation_center: unyt.array, vector: unyt.array, axis: str
    # ) -> unyt.array:
    #
    #     out_units = coordinates.units
    #     _coordinates = coordinates.value
    #     _rotation_center = rotation_center.value
    #     _vector = vector.value
    #
    #     # Normalise vector for more reliable handling
    #     _vector /= np.linalg.norm(_vector)
    #
    #     # Get the de-rotation matrix:
    #     # axis='z' is the default and corresponds to face-on (looking down z-axis)
    #     # axis='y' corresponds to edge-on (maximum rotational signal)
    #     rotation_matrix = rotation_matrix_from_vector(_vector, axis=axis)
    #
    #     if _rotation_center is not None:
    #         _coordinates[:, 0] -= _rotation_center[0]
    #         _coordinates[:, 1] -= _rotation_center[1]
    #         _coordinates[:, 2] -= _rotation_center[2]
    #         x, y, z = np.matmul(rotation_matrix, _coordinates.T)
    #         x += _rotation_center[0]
    #         y += _rotation_center[1]
    #         z += _rotation_center[2]
    #
    #     else:
    #         x, y, z = _coordinates.T
    #
    #     rotated = np.vstack((x, y, z)).T
    #
    #     return rotated * out_units
    #
    # def get_tilt(self, tilt: str = 'z') -> Tuple[unyt.unyt_array, str]:
    #
    #     if len(tilt) == 1:
    #         vec = np.array([0., 0., 1.]) * unyt.dimensionless
    #         if tilt == 'z':
    #             ax = 'y'
    #         elif tilt == 'y':
    #             ax = 'x'
    #         elif tilt == 'x':
    #             ax = 'z'
    #     else:
    #         vec = self.angular_momentum_hot_gas
    #         if tilt == 'faceon':
    #             ax = 'y'
    #         elif tilt == 'edgeon':
    #             ax = 'z'
    #
    #     return vec, ax

    def rotate_coordinates(self, particle_type: int, tilt: str = 'z') -> unyt.array:

        cop = self.data.subfind_tab.FOF.GroupCentreOfPotential
        coord = self.data.subfind_particles[f'PartType{particle_type}']['Coordinates']
        coord[:, 0] -= cop[0]
        coord[:, 1] -= cop[1]
        coord[:, 2] -= cop[2]
        x, y, z = coord.value.T

        if tilt == 'y':
            new_coord = np.vstack((x, z, y)).T
        elif tilt == 'z':
            new_coord = np.vstack((x, y, z)).T
        elif tilt == 'x':
            new_coord = np.vstack((z, y, x)).T
        elif tilt == 'faceon':
            face_on_rotation_matrix = rotation_matrix_from_vector(self.angular_momentum_hot_gas.value)
            new_coord = np.einsum('ijk,ik->ij', face_on_rotation_matrix, coord.value)
        elif tilt == 'edgeon':
            edge_on_rotation_matrix = rotation_matrix_from_vector(self.angular_momentum_hot_gas.value, axis='y')
            new_coord = np.einsum('ijk,ik->ij', edge_on_rotation_matrix, coord.value)

        new_coord *= coord.units
        # new_coord[:, 0] += cop[0]
        # new_coord[:, 1] += cop[1]
        # new_coord[:, 2] += cop[2]
        return new_coord

    def rotate_velocities(self, particle_type: int, tilt: str = 'z', boost: unyt.unyt_array = None) -> unyt.array:

        velocities = self.data.subfind_particles[f'PartType{particle_type}']['Velocity']
        if boost is not None:
            velocities[:, 0] -= boost[0]
            velocities[:, 1] -= boost[1]
            velocities[:, 2] -= boost[2]

        vx, vy, vz = velocities.value.T

        if tilt == 'y':
            new_vel = np.vstack((vx, vz, vy)).T
        elif tilt == 'z':
            new_vel = np.vstack((vx, -vy, vz)).T
        elif tilt == 'x':
            new_vel = np.vstack((-vz, -vy, vx)).T
        elif tilt == 'faceon':
            face_on_rotation_matrix = rotation_matrix_from_vector(self.angular_momentum_hot_gas.value)
            new_vel = np.einsum('ijk,ik->ij', face_on_rotation_matrix, velocities.value)
        elif tilt == 'edgeon':
            edge_on_rotation_matrix = rotation_matrix_from_vector(self.angular_momentum_hot_gas.value, axis='y')
            new_vel = np.einsum('ijk,ik->ij', edge_on_rotation_matrix, velocities.value)

        return new_vel * velocities.units

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

    def set_hot_gas(self) -> None:

        temperature = self.data.subfind_particles['PartType0']['Temperature']
        temperature_cut = np.where(temperature.value > self.hot_gas_temperature_threshold)[0]
        coord = self.data.subfind_particles['PartType0']['Coordinates'][temperature_cut]
        vel = self.data.subfind_particles['PartType0']['Velocity'][temperature_cut]
        masses = self.data.subfind_particles['PartType0']['Mass'][temperature_cut]

        peculiar_velocity = np.sum(vel * masses[:, None], axis=0) / np.sum(masses)
        setattr(self, 'peculiar_velocity_hot_gas', peculiar_velocity)
        read.pprint(self.peculiar_velocity_hot_gas)

        angular_momentum = np.sum(np.cross(coord, vel * masses[:, None]), axis=0)
        setattr(self, 'angular_momentum_hot_gas', angular_momentum * coord.units * vel.units * masses.units)
        read.pprint(self.angular_momentum_hot_gas)

        return

    def make_map(self, particle_type: int, weights: unyt.array, tilt: str = 'z') -> np.ndarray:

        cop = self.data.subfind_tab.FOF.GroupCentreOfPotential
        R500c = self.data.subfind_tab.FOF.Group_R_Crit500
        coord = self.data.subfind_particles[f'PartType{particle_type}']['Coordinates']
        coord_rot = self.rotate_coordinates(particle_type, tilt=tilt)
        smoothing_lengths = self.data.subfind_particles[f'PartType{particle_type}']['SmoothingLength']
        aperture = unyt.unyt_quantity(5 * R500c / np.sqrt(3), coord.units)

        if particle_type == 0:
            temperature = self.data.subfind_particles['PartType0']['Temperature']
            spatial_filter = np.where(
                (np.abs(coord_rot[:, 0]) < aperture) &
                (np.abs(coord_rot[:, 1]) < aperture) &
                (np.abs(coord_rot[:, 2]) < aperture) &
                (temperature.value > self.hot_gas_temperature_threshold)
            )[0]
        else:
            spatial_filter = np.where(
                (np.abs(coord_rot[:, 0]) < aperture) &
                (np.abs(coord_rot[:, 1]) < aperture) &
                (np.abs(coord_rot[:, 2]) < aperture)
            )[0]

        read.pprint(coord_rot, cop, spatial_filter, sep='\n')
        x_max = np.max(coord_rot[spatial_filter, 0])
        x_min = np.min(coord_rot[spatial_filter, 0])
        y_max = np.max(coord_rot[spatial_filter, 1])
        y_min = np.min(coord_rot[spatial_filter, 1])
        x_range = x_max - x_min
        y_range = y_max - y_min
        x = (coord_rot[spatial_filter, 0] - x_min) / x_range
        y = (coord_rot[spatial_filter, 1] - y_min) / y_range
        h = smoothing_lengths[spatial_filter] / (2 * aperture)

        # Gather and handle coordinates to be processed
        x = np.asarray(x.value, dtype=np.float64)
        y = np.asarray(y.value, dtype=np.float64)
        m = np.asarray(weights[spatial_filter].value, dtype=np.float32)
        h = np.asarray(h.value, dtype=np.float32)
        smoothed_map = scatter(x=x, y=y, m=m, h=h, res=self.resolution).T
        smoothed_map = np.ma.masked_where(np.abs(smoothed_map) < 1.e-9, smoothed_map)

        surface_element = x_range * y_range / self.resolution ** 2
        setattr(Mapping, f'surface_element{particle_type}', surface_element)

        return smoothed_map  # * weights.units / coord.units ** 2

    def map_particle_number(self, particle_type: int, tilt: str = 'z') -> np.ndarray:
        masses = self.data.subfind_particles[f'PartType{particle_type}']['Mass']
        weights = np.ones_like(masses.value, dtype=np.float32) * unyt.dimensionless
        del masses
        return self.make_map(particle_type, weights, tilt=tilt)

    def map_mass(self, particle_type: int, tilt: str = 'z') -> np.ndarray:
        weights = self.data.subfind_particles[f'PartType{particle_type}']['Mass']
        return self.make_map(particle_type, weights, tilt=tilt)

    def map_density(self, particle_type: int, tilt: str = 'z') -> np.ndarray:
        if particle_type != 1:
            weights = self.data.subfind_particles[f'PartType{particle_type}']['Mass']
            return self.make_map(particle_type, weights, tilt=tilt)
        else:
            read.wwarn('Density map not defined for dark_matter particles.')

    def map_metallicity(self, particle_type: int, tilt: str = 'z') -> np.ndarray:
        if particle_type != 1:
            weights = self.data.subfind_particles[f'PartType{particle_type}']['Metallicity']
            return self.make_map(particle_type, weights, tilt=tilt)
        else:
            read.wwarn('Metallicity map not defined for dark_matter particles.')

    def map_mass_weighted_temperature(self, particle_type: int, tilt: str = 'z') -> np.ndarray:
        if particle_type == 0:
            mass_weighted_temps = self.data.subfind_particles[f'PartType{particle_type}']['Mass'].T * \
                                  self.data.subfind_particles[f'PartType{particle_type}']['Temperature']
            mass_weighted_temps_map = self.make_map(particle_type, mass_weighted_temps, tilt=tilt)
            mass = self.data.subfind_particles[f'PartType{particle_type}']['Mass']
            mass_map = self.make_map(particle_type, mass, tilt=tilt)
            return mass_weighted_temps_map / mass_map
        else:
            read.wwarn('Mass-weighted-temperature map only defined for gas particles.')

    def map_tSZ(self, particle_type: int, tilt: str = 'z') -> np.ndarray:
        if particle_type == 0:

            mass_weighted_temps = self.data.subfind_particles[f'PartType{particle_type}']['Mass'].T * \
                                  self.data.subfind_particles[f'PartType{particle_type}']['Temperature']
            weights = mass_weighted_temps * tsz_const / unyt.unyt_quantity(1., unyt.Mpc)
            return self.make_map(particle_type, weights, tilt=tilt)
        else:
            read.wwarn('Thermal SZ map only defined for gas particles.')

    def map_kSZ(self, particle_type: int, tilt: str = 'z') -> np.ndarray:

        if particle_type == 0:
            radial_velocities = self.rotate_velocities(particle_type, tilt=tilt)[:, 2]
            mass_weighted_temps = self.data.subfind_particles[f'PartType{particle_type}']['Mass'].T * radial_velocities
            weights = mass_weighted_temps * ksz_const / unyt.unyt_quantity(1., unyt.Mpc)
            return self.make_map(particle_type, weights, tilt=tilt)
        else:
            read.wwarn('Kinetic SZ map only defined for gas particles.')

    def map_rkSZ(self, particle_type: int, tilt: str = 'z') -> np.ndarray:

        if particle_type == 0:
            # Derotate velocities and subtract bulk motion (work in cluster's frame)
            # The boost velocity is subtracted as bulk motion
            radial_velocities = self.rotate_velocities(
                particle_type,
                tilt=tilt,
                boost=self.peculiar_velocity_hot_gas
            )[:, 2]
            mass_weighted_temps = self.data.subfind_particles[f'PartType{particle_type}']['Mass'].T * radial_velocities
            weights = mass_weighted_temps * ksz_const / unyt.unyt_quantity(1., unyt.Mpc)
            return self.make_map(particle_type, weights, tilt=tilt)
        else:
            read.wwarn('Rotational-kinetic SZ map only defined for gas particles.')

    def map_particle_dot(self, particle_type: int, tilt: str = 'z') -> np.ndarray:
        cop = self.data.subfind_tab.FOF.GroupCentreOfPotential
        R500c = self.data.subfind_tab.FOF.Group_R_Crit500
        coord = self.data.subfind_particles[f'PartType{particle_type}']['Coordinates']
        coord_rot = self.rotate_coordinates(particle_type, tilt=tilt)
        aperture = unyt.unyt_quantity(5 * R500c / np.sqrt(3), coord.units)
        spatial_filter = np.where(
            (np.abs(coord_rot[:, 0] - cop[0]) < aperture) &
            (np.abs(coord_rot[:, 1] - cop[1]) < aperture) &
            (np.abs(coord_rot[:, 2] - cop[2]) < aperture)
        )[0]
        x = coord_rot[spatial_filter, 0] - cop[0]
        y = coord_rot[spatial_filter, 1] - cop[1]
        return x.value, y.value

    def view_all(self):

        fig, axarr = plt.subplots(nrows=5, ncols=15, sharex=False, sharey=False, figsize=(30, 10))

        viewpoints = ['z', 'y', 'x', 'faceon', 'edgeon']

        for i_plot, viewpoint in enumerate(viewpoints):

            read.pprint(f"Rendering veiwpoint {i_plot + 1:d}/{len(viewpoints):d}: {viewpoint:s}.")

            for ax in axarr[i_plot, :]:
                ax.set_aspect('equal')
                ax.axis("off")

            axarr[i_plot, 0].imshow(
                self.map_mass(0, tilt=viewpoint),
                norm=LogNorm(),
                cmap="PuOr",
                origin="lower",
            )
            axarr[i_plot, 0].text(.5, .9, 'Gas mass', horizontalalignment='center',
                                  transform=axarr[i_plot, 0].transAxes)

            axarr[i_plot, 1].imshow(
                self.map_density(0, tilt=viewpoint),
                norm=LogNorm(),
                cmap="inferno",
                origin="lower",
            )
            axarr[i_plot, 1].text(.5, .9, 'Gas density', horizontalalignment='center',
                                  transform=axarr[i_plot, 1].transAxes)

            coord_x, coord_y = self.map_particle_dot(0, tilt=viewpoint)
            axarr[i_plot, 2].plot(coord_x, coord_y, ',', c="C0", alpha=1)
            axarr[i_plot, 2].set_xlim([np.min(coord_x), np.max(coord_x)])
            axarr[i_plot, 2].set_ylim([np.min(coord_y), np.max(coord_y)])
            axarr[i_plot, 2].text(.5, .9, 'Gas dot', horizontalalignment='center', transform=axarr[i_plot, 2].transAxes)

            axarr[i_plot, 3].imshow(
                self.map_particle_number(0, tilt=viewpoint),
                norm=LogNorm(),
                cmap="cividis",
                origin="lower",
            )
            axarr[i_plot, 3].text(.5, .9, 'Gas particle number', horizontalalignment='center',
                                  transform=axarr[i_plot, 3].transAxes)

            axarr[i_plot, 4].imshow(
                self.map_mass_weighted_temperature(0, tilt=viewpoint),
                norm=LogNorm(),
                cmap="inferno",
                origin="lower",
            )
            axarr[i_plot, 4].text(.5, .9, 'Gas temperature', horizontalalignment='center',
                                  transform=axarr[i_plot, 4].transAxes)

            axarr[i_plot, 5].imshow(
                self.map_tSZ(0, tilt=viewpoint),
                norm=LogNorm(),
                cmap="inferno",
                origin="lower",
            )
            axarr[i_plot, 5].text(.5, .9, 'Gas tSZ', horizontalalignment='center', transform=axarr[i_plot, 5].transAxes)

            ksz = self.map_kSZ(0, tilt=viewpoint)
            vlim = np.abs(ksz).max()
            read.pprint(vlim)
            axarr[i_plot, 6].imshow(
                ksz,
                norm=SymLogNorm(linthresh=1e-5, linscale=0.5, vmin=-vlim, vmax=vlim, base=10),
                cmap="PuOr",
                origin="lower",
            )
            axarr[i_plot, 6].text(.5, .9, 'Gas kSZ', horizontalalignment='center', transform=axarr[i_plot, 6].transAxes)

            rksz = self.map_rkSZ(0, tilt=viewpoint)
            vlim = np.abs(rksz).max()
            read.pprint(vlim)
            axarr[i_plot, 7].imshow(
                rksz,
                norm=SymLogNorm(linthresh=1e-5, linscale=0.5, vmin=-vlim, vmax=vlim, base=10),
                cmap="PuOr",
                origin="lower",
            )
            axarr[i_plot, 7].text(.5, .9, 'Gas rkSZ', horizontalalignment='center',
                                  transform=axarr[i_plot, 7].transAxes)

            axarr[i_plot, 8].imshow(
                self.map_mass(1, tilt=viewpoint),
                norm=LogNorm(),
                cmap="PuOr",
                origin="lower",
            )
            axarr[i_plot, 8].text(.5, .9, 'DM mass', horizontalalignment='center', transform=axarr[i_plot, 8].transAxes)

            coord_x, coord_y = self.map_particle_dot(1, tilt=viewpoint)
            axarr[i_plot, 9].plot(coord_x, coord_y, ',', c="C0", alpha=1)
            axarr[i_plot, 9].set_xlim([np.min(coord_x), np.max(coord_x)])
            axarr[i_plot, 9].set_ylim([np.min(coord_y), np.max(coord_y)])
            axarr[i_plot, 9].text(.5, .9, 'DM dot', horizontalalignment='center', transform=axarr[i_plot, 9].transAxes)

            axarr[i_plot, 10].imshow(
                self.map_particle_number(1, tilt=viewpoint),
                norm=LogNorm(),
                cmap="cividis",
                origin="lower",
            )
            axarr[i_plot, 10].text(.5, .9, 'DM particle number', horizontalalignment='center',
                                   transform=axarr[i_plot, 10].transAxes)

            axarr[i_plot, 11].imshow(
                self.map_mass(4, tilt=viewpoint),
                norm=LogNorm(),
                cmap="inferno",
                origin="lower",
            )
            axarr[i_plot, 11].text(.5, .9, 'Star mass', horizontalalignment='center',
                                   transform=axarr[i_plot, 11].transAxes)

            axarr[i_plot, 12].imshow(
                self.map_density(4, tilt=viewpoint),
                norm=LogNorm(),
                cmap="PuOr",
                origin="lower",
            )
            axarr[i_plot, 12].text(.5, .9, 'Star density', horizontalalignment='center',
                                   transform=axarr[i_plot, 12].transAxes)

            coord_x, coord_y = self.map_particle_dot(4, tilt=viewpoint)
            axarr[i_plot, 13].plot(coord_x, coord_y, ',', c="C0", alpha=1)
            axarr[i_plot, 13].set_xlim([np.min(coord_x), np.max(coord_x)])
            axarr[i_plot, 13].set_ylim([np.min(coord_y), np.max(coord_y)])
            axarr[i_plot, 13].text(.5, .9, 'Star dot', horizontalalignment='center',
                                   transform=axarr[i_plot, 13].transAxes)

            axarr[i_plot, 14].imshow(
                self.map_particle_number(4, tilt=viewpoint),
                norm=LogNorm(),
                cmap="cividis",
                origin="lower",
            )
            axarr[i_plot, 14].text(.5, .9, 'Star particle number', horizontalalignment='center',
                                   transform=axarr[i_plot, 14].transAxes)

            plt.subplots_adjust(wspace=0., hspace=0.)
            plt.tight_layout()

        return fig, axarr

    def savefig(self, fig, *args, **kwargs):

        fig.savefig(*args, **kwargs)


if __name__ == '__main__':

    import pickle
    import os

    output_directory = '/home/altamura/nas/rksz-bahamas'
    simulation_type = 'hydro'
    redshifts = ['z003p000', 'z002p000', 'z001p000', 'z000p250', 'z000p000']
    cluster_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 200, 1000]

    for redshift in redshifts:

        # Boot up the BAHAMAS data
        files = read.find_files(simulation_type, redshift)
        fofs = read.fof_groups(files)
        csrm = read.csr_index_matrix(fofs)

        for job_id, n in enumerate(cluster_ids):

            # Parallelize across ranks
            # If image not allocated to specific rank, skip iteration
            if job_id % read.nproc != read.rank:
                continue

            try:

                if not os.path.isfile(f'{output_directory}/test_cluster_data.{redshift}_{n}.pickle'):
                    fof = read.fof_group(n, fofs)
                    cluster_dict = read.fof_particles(fof, csrm)

                    with open(f'{output_directory}/test_cluster_data.{redshift}_{n}.pickle', 'wb') as handle:
                        pickle.dump(cluster_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                if not os.path.isfile(f'{output_directory}/test.{redshift}_{n}.png'):
                    with open(f'{output_directory}/test_cluster_data.{redshift}_{n}.pickle', 'rb') as handle:
                        cluster_dict = pickle.load(handle)

                    cluster_data = read.class_wrap(cluster_dict).data
                    maps = Mapping(cluster_data)
                    fig, _ = maps.view_all()
                    maps.savefig(fig, f'{output_directory}/test.{redshift}_{n}.png', dpi=(maps.resolution * 15) // 30)

            except IndexError as e:
                # Some high-index halos may not have sufficient mass to be picked up at high
                # redshifts. Catch that exception and continue with other halos.
                read.pprint(e)
                continue
