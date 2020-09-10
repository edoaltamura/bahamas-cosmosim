import matplotlib
matplotlib.use('Agg')

import argparse
import copy
import numpy as np
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation.projection import scatter_parallel as scatter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import read

try:
    plt.style.use("mnras.mplstyle")
except:
    pass

partType_atlas = {
    '0': 'gas',
    '1': 'dark matter',
    '4': 'stars'
}
map_resolution = 1024


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def rotation_align_with_vector(coordinates: np.ndarray, rotation_center: np.ndarray, vector: np.ndarray) -> np.ndarray:

    # Normalise vector for more reliable handling
    vector /= np.linalg.norm(vector)

    # Get the derotation matrix:
    # axis='z' is the default and corresponds to face-on (looking down z-axis)
    # axis='y' corresponds to edge-on (maximum rotational signal)
    rotation_matrix = rotation_matrix_from_vector(vector, axis='y')

    if rotation_center is not None:
        # Rotate co-ordinates as required
        x, y, z = np.matmul(rotation_matrix, (coordinates - rotation_center).T)
        x += rotation_center[0]
        y += rotation_center[1]
        z += rotation_center[2]

    else:
        x, y, z = coordinates.T

    return np.vstack((x, y, z)).T


def density_map(particle_type: int, cluster_data) -> None:

    z = cluster_data.header.subfind_particles.Redshift
    CoP = cluster_data.subfind_tab.FOF.GroupCentreOfPotential
    M200c = cluster_data.subfind_tab.FOF.Group_M_Crit200
    R200c = cluster_data.subfind_tab.FOF.Group_R_Crit200
    R500c = cluster_data.subfind_tab.FOF.Group_R_Crit500
    M500c = cluster_data.subfind_tab.FOF.Group_M_Crit500
    coord = cluster_data.subfind_particles[f'PartType{particle_type}']['Coordinates']
    boxsize = cluster_data.boxsize
    DM_part_mass = cluster_data.mass_DMpart

    if particle_type == 1:

        masses = np.ones_like(coord[:, 0].value, dtype=np.float32) * DM_part_mass
        # Generate DM particle smoothing lengths
        smoothing_lengths = generate_smoothing_lengths(
            coord,
            boxsize,
            kernel_gamma=1.8,
            neighbours=57,
            speedup_fac=3,
            dimension=3,
        )

    else:
        masses = cluster_data.subfind_particles[f'PartType{particle_type}']['Mass']
        smoothing_lengths = cluster_data.subfind_particles[f'PartType{particle_type}']['SmoothingLength']

    # Run aperture filter
    read.pprint('[Check] Particle max x: ', np.max(np.abs(coord[:, 0] - CoP[0])), '6 x R500c: ', 6 * R500c)
    read.pprint('[Check] Particle max y: ', np.max(np.abs(coord[:, 1] - CoP[1])), '6 x R500c: ', 6 * R500c)
    read.pprint('[Check] Particle max z: ', np.max(np.abs(coord[:, 2] - CoP[2])), '6 x R500c: ', 6 * R500c)

    # Rotate particles
    # coord_rot = rotation_align_with_vector(coord.value, CoP, np.array([0, 0, 1]))
    coord_rot = coord

    # After derotation create a cubic aperture filter inscribed within a sphere of radius 5xR500c and
    # Centred in the CoP. Each semi-side of the aperture has length sqrt(3) / 2 * 5 * R500c.
    aperture = np.sqrt(3) / 2 * 5 * R500c
    mask = np.where(
        (np.abs(coord_rot[:, 0] - CoP[0]) <= aperture) &
        (np.abs(coord_rot[:, 1] - CoP[1]) <= aperture) &
        (np.abs(coord_rot[:, 2] - CoP[2]) <= aperture)
    )[0]

    # Gather and handle coordinates to be plotted
    x = coord_rot[mask, 0].value
    y = coord_rot[mask, 1].value

    map_input_m = np.asarray(masses.value, dtype=np.float32)
    map_input_h = np.asarray(smoothing_lengths.value, dtype=np.float32)
    mass_map = scatter(
        x=(x - aperture.value) / (2 * aperture.value),
        y=(y - aperture.value) / (2 * aperture.value),
        m=map_input_m[mask],
        h=map_input_h[mask] / (2 * aperture.value),
        res=map_resolution
    )
    mass_map_units = masses.units / coord.units ** 2

    # Mask zero values in the map with black
    mass_map = np.ma.masked_where(mass_map < 1, mass_map)
    read.pprint(mass_map)

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=map_resolution // 6)
    ax.set_aspect('equal')
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.imshow(
        mass_map.T,
        norm=LogNorm(),
        cmap="inferno",
        origin="lower",
        extent=(-aperture.value, aperture.value, -aperture.value, aperture.value)
    )

    t = ax.text(
        0.025,
        0.025,
        (
            f"Halo {cluster_id:d} {simulation_type}\n"
            f"Particles: {partType_atlas[str(particle_type)]}\n"
            f"$z={z:3.3f}$\n"
            f"$M_{{500c}}={latex_float(M500c.value)}$ M$_\odot$\n"
            f"$R_{{500c}}={latex_float(R500c.value)}$ Mpc\n"
            f"$M_{{200c}}={latex_float(M200c.value)}$ M$_\odot$\n"
            f"$R_{{200c}}={latex_float(R200c.value)}$ Mpc"
        ),
        color="white",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )
    t.set_bbox(dict(facecolor='black', alpha=0.2, edgecolor='none'))
    ax.text(
        0, (1-0.02) * R200c,
        r"$R_{200c}$",
        color="white",
        ha="center",
        va="top"
    )
    ax.text(
        0, 1.02 * R500c,
        r"$R_{500c}$",
        color="white",
        ha="center",
        va="bottom"
    )
    circle_r200 = plt.Circle((0, 0), R200c, color="white", fill=False, linestyle='--')
    circle_r500 = plt.Circle((0, 0), R500c, color="white", fill=False, linestyle='-')
    ax.add_artist(circle_r200)
    ax.add_artist(circle_r500)
    plt.tight_layout()
    fig.savefig(f"{output_directory}/halo{cluster_id}_{redshift}_densitymap_type{particle_type}.png")
    plt.close(fig)


if __name__ == '__main__':

    # Parse particle type flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--gas', default=False, action='store_true')
    parser.add_argument('--dark_matter', default=False, action='store_true')
    parser.add_argument('--stars', default=False, action='store_true')
    args = parser.parse_args()

    # -------------------------------------------------------------------- #
    # Edit these parameters
    simulation_type = 'hydro'
    redshift = 'z003p000'
    cluster_id = 0
    size_R200c = 1
    output_directory = '/local/scratch/altamura/bahamas/maps'
    # -------------------------------------------------------------------- #
    # Boot up the BAHAMAS data
    files = read.find_files(simulation_type, redshift)
    fofs = read.fof_groups(files)
    csrm = read.csr_index_matrix(fofs)
    fof = read.fof_group(cluster_id, fofs)
    cluster_data = read.class_wrap(read.fof_particles(fof, csrm)).data

    # Execute tasks
    if args.gas:
        read.pprint("Generating gas density map")
        density_map(0, cluster_data)
    if args.dark_matter:
        read.pprint("Generating dark_matter density map")
        density_map(1, cluster_data)
    if args.stars:
        read.pprint("Generating stars density map")
        density_map(4, cluster_data)

    read.pprint("Job done.")