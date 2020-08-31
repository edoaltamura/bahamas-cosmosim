import warnings
import datetime
from mpi4py import MPI
warnings.filterwarnings("ignore")

def run(redshift: str = None) -> None:
    from .__init__ import pprint
    from .read import (
        find_files,
        fof_header,
        fof_groups,
        snap_groupnumbers,
        snap_coordinates,
        cluster_data
    )
    from .utils import (
        file_benchmarks,
        record_benchmarks,
        time_checkpoint,
        report_file
    )

    # Upper level relative imports
    from .__init__ import rank, Cluster, save_report, save_group

    # -----------------------------------------------------------------------
    # Set simulation parameters
    REDSHIFT = redshift

    # Initialise benchmarks
    # file_benchmarks(REDSHIFT)
    # halo_load_time = []

    # Initialise snapshot output file
    # snap_file = report_file(REDSHIFT)

    # -----------------------------------------------------------------------
    # Load snapshot data
    pprint(f'[+] BAHAMAS HYDRO: redshift {REDSHIFT}')
    files = find_files(REDSHIFT)
    header = fof_header(files)
    fofs = fof_groups(files)
    number_halos = len(fofs['idx'])
    pprint('Number of halos', number_halos)
    # snap_partgn = snap_groupnumbers(fofgroups = fofs)
    snap_coords = snap_coordinates(fofgroups = fofs)

    for i in range(number_halos):
        try:
            # Extract data from subfind output
            # start_1 = datetime.datetime.now()
            # halo_data = cluster_data(i, header, fofgroups=fofs, groupNumbers=snap_partgn)
            halo_data = cluster_data(i, header, fofgroups=fofs, coordinates=snap_coords)
            # record_benchmarks(REDSHIFT, ('load', i, time_checkpoint(start_1)))

            # Parse data into Cluster object
            # start_2 = datetime.datetime.now()
            cluster = Cluster.from_dict(simulation_name='bahamas', data=halo_data)
            del halo_data

            # Try pushing results into an h5 file
            # halo_report = save_report(cluster)
            del cluster

        except:
            pprint(f"[-] ERROR Processing cluster{i} failed")
        # else:
        #     if rank == 0: save_group(snap_file, f"/halo_{i:05d}/", halo_report)
        #     del halo_report
        #
        # record_benchmarks(REDSHIFT, ('compute', i, time_checkpoint(start_2)))
        #
        # # -----------------------------------------------------------------------
        # if i%50 == 0:
        #     # Time it
        #     halo_load_time.append((datetime.datetime.now() - start_1).total_seconds())
        #     if number_halos < 5 or len(halo_load_time) < 6:
        #         completion_time = sum(halo_load_time)/len(halo_load_time) * number_halos
        #     else:
        #         completion_time = sum(halo_load_time[-5:]) / 4 * (number_halos-i+1)
        #     pprint(f"[x] ({len(halo_load_time):d}/{number_halos:d}) Estimated completion time: {datetime.timedelta(seconds=completion_time)}")
        # -----------------------------------------------------------------------

    # MPI.COMM_WORLD.Barrier()
    # if rank==0:
    #     snap_file.close()


