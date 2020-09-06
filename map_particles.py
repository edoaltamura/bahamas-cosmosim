import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import argparse
import read

try:
    plt.style.use("mnras.mplstyle")
except:
    pass


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

# -------------------------------------------------------------------- #
# Edit these parameters
simulation_type = 'hydro'
redshift = 'z003p000'
cluster_id = 0
size_R200c = 10
output_directory = '/local/scratch/altamura/bahamas/maps'
# -------------------------------------------------------------------- #
# Boot up the BAHAMAS data
files = read.find_files(simulation_type, redshift)
fofs = read.fof_groups(files)
csrm = read.csr_index_matrix(fofs)
fof = read.fof_group(cluster_id, fofs)
cluster_data = read.class_wrap(read.fof_particles(fof, csrm)).data

redshift = cluster_data.header.subfind_particles.Redshift
CoP = cluster_data.subfind_tab.FOF.GroupCentreOfPotential
M200c = cluster_data.subfind_tab.FOF.Group_M_Crit200
R200c = cluster_data.subfind_tab.FOF.Group_R_Crit200
size = R200c * size_R200c
# -------------------------------------------------------------------- #

def particle_map_type(particle_type: int) -> None:

    coord = cluster_data.subfind_particles[f'PartType{particle_type}']['Coordinates']
    coord_x = coord[:, 0] - CoP[0]
    coord_y = coord[:, 1] - CoP[1]
    coord_z = coord[:, 2] - CoP[2]
    del coord

    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=1024 // 8)
    ax.set_aspect('equal')
    ax.plot(coord_x, coord_y, ',', c="C0", alpha=0.1)
    ax.set_xlim([-size.value, size.value])
    ax.set_ylim([-size.value, size.value])
    ax.set_ylabel(r"$y$ [Mpc]")
    ax.set_xlabel(r"$x$ [Mpc]")

    ax.text(
        0.025,
        0.025,
        (
            f"Halo {cluster_id:d} {simulation_type}\n"
            f"$z={redshift:3.3f}$\n"
            f"$M_{{200c}}={latex_float(M200c.value)}$ M$_\odot$\n"
            f"$R_{{200c}}={latex_float(R200c.value)}$ Mpc\n"
        ),
        color="black",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.text(
        0,
        0 + 1.05 * R200c,
        r"$R_{200c}$",
        color="black",
        ha="center",
        va="bottom"
    )
    ax.text(
        0,
        0 + 1.002 * 5 * R200c,
        r"$5 \times R_{200c}$",
        color="grey",
        ha="center",
        va="bottom"
    )
    circle_r200 = plt.Circle((0, 0), R200c, color="black", fill=False, linestyle='-')
    circle_5r200 = plt.Circle((0, 0), 5 * R200c, color="grey", fill=False, linestyle='--')
    ax.add_artist(circle_r200)
    ax.add_artist(circle_5r200)
    fig.savefig(f"{output_directory}/halo{cluster_id}_particlemap_type{particle_type}_{size_R200c}r200.png")
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gas', default=False, action='store_true')
    parser.add_argument('--dark_matter', default=False, action='store_true')
    parser.add_argument('--stars', default=False, action='store_true')
    args = parser.parse_args()

    if args.gas:
        particle_map_type(0)
    if args.dark_matter:
        particle_map_type(1)
    if args.stars:
        particle_map_type(4)