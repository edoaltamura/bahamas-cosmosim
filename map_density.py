import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths
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
        return


def rescale(coord: np.ndarray) -> np.ndarray:
    """
    Rescaled the array of input to the range [x_min, x_max] linearly.
    This method is often used in the context of making maps with matplotlib.pyplot.inshow.
    The matrix to be accepted must contain arrays in the [0,1] range.
    """
    x_max = np.max(coord[:, 0])
    x_min = np.min(coord[:, 0])
    y_max = np.max(coord[:, 1])
    y_min = np.min(coord[:, 1])
    z_max = np.max(coord[:, 2])
    z_min = np.min(coord[:, 2])
    coord_max = np.max([x_max, y_max, z_max])
    coord_min = np.min([x_min, y_min, z_min])
    coord_range = coord_max - coord_min
    rescaled_coords = np.copy(coord)
    rescaled_coords[:, 0] -= x_min
    rescaled_coords[:, 1] -= y_min
    rescaled_coords[:, 2] -= z_min
    rescaled_coords[:, 0] /= coord_range
    rescaled_coords[:, 1] /= coord_range
    rescaled_coords[:, 2] /= coord_range
    assert np.max(rescaled_coords) <= 1.
    assert np.min(rescaled_coords) >= 0.
    return np.asarray(rescaled_coords, dtype=np.float64)



def dm_render(coordinates, masses, boxsize, resolution: int = map_resolution):
    # Generate smoothing lengths for the dark matter
    smoothing_lengths = generate_smoothing_lengths(
        coordinates,
        boxsize,
        kernel_gamma=1.8,
        neighbours=57,
        speedup_fac=2,
        dimension=3,
    )
    # Project the dark matter mass
    dm_map = scatter(
        coordinates[:, 0].value,
        coordinates[:, 1].value,
        masses.value,
        smoothing_lengths.value,
        resolution
    )
    return dm_map

def gas_density_map(cluster_data) -> None:

    z = cluster_data.header.subfind_particles.Redshift
    CoP = cluster_data.subfind_tab.FOF.GroupCentreOfPotential
    M200c = cluster_data.subfind_tab.FOF.Group_M_Crit200
    R200c = cluster_data.subfind_tab.FOF.Group_R_Crit200
    R500c = cluster_data.subfind_tab.FOF.Group_R_Crit500
    M500c = cluster_data.subfind_tab.FOF.Group_M_Crit500
    size = R200c * size_R200c

    coord = cluster_data.subfind_particles['PartType0']['Coordinates']
    coord[:, 0] -= - CoP[0]
    coord[:, 1] -= - CoP[1]
    coord[:, 2] -= - CoP[2]
    masses = cluster_data.subfind_particles['PartType0']['Mass']
    smoothing_lengths = cluster_data.subfind_particles['PartType0']['SmoothingLength']

    coord_map = rescale(coord.value)
    map_input_m = np.asarray(masses.value, dtype=np.float32)
    map_input_h = np.asarray(smoothing_lengths.value, dtype=np.float32)
    gas_mass = scatter(
        x=coord_map[:, 0],
        y=coord_map[:, 1],
        m=map_input_m,
        h=map_input_h,
        res=map_resolution
    )

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=map_resolution // 6)
    ax.set_aspect('equal')
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.imshow(gas_mass.T, norm=LogNorm(), cmap="inferno", origin="lower", extent=([-size.value, size.value] + [-size.value, size.value]))
    ax.set_ylabel(r"$y$ [Mpc]")
    ax.set_xlabel(r"$x$ [Mpc]")

    t = ax.text(
        0.025,
        0.025,
        (
            f"Halo {cluster_id:d} {simulation_type}\n"
            f"Particles: gas\n"
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
    t.set_bbox(dict(facecolor='black', alpha=0.6, edgecolor='none'))

    ax.text(
        0,
        0 + 1.05 * R200c,
        r"$R_{200c}$",
        color="grey",
        ha="center",
        va="bottom"
    )
    ax.text(
        0,
        0 + 1.05 * R500c,
        r"$R_{500c}$",
        color="grey",
        ha="center",
        va="bottom"
    )
    circle_r200 = plt.Circle((0, 0), R200c, color="grey", fill=False, linestyle='--')
    circle_r500 = plt.Circle((0, 0), R500c, color="grey", fill=False, linestyle='-')
    ax.add_artist(circle_r200)
    ax.add_artist(circle_r500)
    fig.savefig(f"{output_directory}/halo{cluster_id}_{redshift}_densitymap_type0_{size_R200c}r200.png")
    plt.close(fig)

def dm_density_map(cluster_data) -> None:

    z = cluster_data.header.subfind_particles.Redshift
    CoP = cluster_data.subfind_tab.FOF.GroupCentreOfPotential
    M200c = cluster_data.subfind_tab.FOF.Group_M_Crit200
    R200c = cluster_data.subfind_tab.FOF.Group_R_Crit200
    R500c = cluster_data.subfind_tab.FOF.Group_R_Crit500
    M500c = cluster_data.subfind_tab.FOF.Group_M_Crit500
    size = R200c * size_R200c
    DM_part_mass = cluster_data.mass_DMpart
    boxsize = cluster_data.boxsize

    coord = cluster_data.subfind_particles['PartType1']['Coordinates']
    coord[:, 0] -= - CoP[0]
    coord[:, 1] -= - CoP[1]
    coord[:, 2] -= - CoP[2]
    masses = np.ones_like(coord[:, 0], dtype=np.float32) * DM_part_mass
    dm_mass = dm_render(coord, masses, boxsize)

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=1024 // 6)
    ax.set_aspect('equal')
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.imshow(dm_mass.T, norm=LogNorm(), cmap="inferno", origin="lower", extent=([-size.value, size.value] + [-size.value, size.value]))
    ax.set_ylabel(r"$y$ [Mpc]")
    ax.set_xlabel(r"$x$ [Mpc]")

    ax.text(
        0.025,
        0.025,
        (
            f"Halo {cluster_id:d} {simulation_type}\n"
            f"Particles: dark matter\n"
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

    ax.text(
        0,
        0 + 1.05 * R200c,
        r"$R_{200c}$",
        color="grey",
        ha="center",
        va="bottom"
    )
    ax.text(
        0,
        0 + 1.05 * R500c,
        r"$R_{500c}$",
        color="grey",
        ha="center",
        va="bottom"
    )
    circle_r200 = plt.Circle((0, 0), R200c, color="grey", fill=False, linestyle='--')
    circle_r500 = plt.Circle((0, 0), R500c, color="grey", fill=False, linestyle='-')
    ax.add_artist(circle_r200)
    ax.add_artist(circle_r500)
    fig.savefig(f"{output_directory}/halo{cluster_id}_{redshift}_densitymap_type1_{size_R200c}r200.png")
    plt.close(fig)


def stars_density_map(cluster_data) -> None:

    z = cluster_data.header.subfind_particles.Redshift
    CoP = cluster_data.subfind_tab.FOF.GroupCentreOfPotential
    M200c = cluster_data.subfind_tab.FOF.Group_M_Crit200
    R200c = cluster_data.subfind_tab.FOF.Group_R_Crit200
    R500c = cluster_data.subfind_tab.FOF.Group_R_Crit500
    M500c = cluster_data.subfind_tab.FOF.Group_M_Crit500
    size = R200c * size_R200c

    coord = cluster_data.subfind_particles['PartType4']['Coordinates']
    coord[:, 0] -= - CoP[0]
    coord[:, 1] -= - CoP[1]
    coord[:, 2] -= - CoP[2]
    masses = cluster_data.subfind_particles['PartType4']['Mass']
    smoothing_lengths = cluster_data.subfind_particles['PartType4']['SmoothingLength']

    coord_map = rescale(coord.value)
    map_input_m = np.asarray(masses.value, dtype=np.float32)
    map_input_h = np.asarray(smoothing_lengths.value, dtype=np.float32)
    stars_mass = scatter(
        x=coord_map[:, 0],
        y=coord_map[:, 1],
        m=map_input_m,
        h=map_input_h,
        res=map_resolution
    )

    # Make figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=map_resolution // 6)
    ax.set_aspect('equal')
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.imshow(stars_mass.T, norm=LogNorm(), cmap="inferno", origin="lower", extent=([-size.value, size.value] + [-size.value, size.value]))
    ax.set_ylabel(r"$y$ [Mpc]")
    ax.set_xlabel(r"$x$ [Mpc]")

    ax.text(
        0.025,
        0.025,
        (
            f"Halo {cluster_id:d} {simulation_type}\n"
            f"Particles: stars\n"
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

    ax.text(
        0,
        1.02 * R200c,
        r"$R_{200c}$",
        color="grey",
        ha="center",
        va="bottom"
    )
    ax.text(
        0,
        1.02 * R500c,
        r"$R_{500c}$",
        color="grey",
        ha="center",
        va="bottom"
    )
    circle_r200 = plt.Circle((0, 0), R200c, color="grey", fill=False, linestyle='--')
    circle_r500 = plt.Circle((0, 0), R500c, color="grey", fill=False, linestyle='-')
    ax.add_artist(circle_r200)
    ax.add_artist(circle_r500)
    fig.savefig(f"{output_directory}/halo{cluster_id}_{redshift}_densitymap_type4_{size_R200c}r200.png")
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
        gas_density_map(cluster_data)
    if args.dark_matter:
        dm_density_map(cluster_data)
    if args.stars:
        stars_density_map(cluster_data)