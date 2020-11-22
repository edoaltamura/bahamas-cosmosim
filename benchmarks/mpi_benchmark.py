from mpi4py import MPI
import datetime
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

for msg_length in [10, 1000, 10000, 100000, 1000000]:

    # master process
    if rank == 0:
        data = np.ones(msg_length)
        print(f"Message size: {data.nbytes}")
        startdelta = 0.
        stopdelta = 0.
        transmitdelta = 0.
        # master process sends data to worker processes by
        # going through the ranks of all worker processes
        for i in range(1, size):
            comm.send(datetime.datetime.now(), dest=i, tag=i)
            comm.send(data, dest=i, tag=i * 2)
            comm.send(datetime.datetime.now(), dest=i, tag=i * 3)

    # worker processes
    else:
        # each worker process receives data from master process
        startdata = comm.recv(source=0, tag=rank)
        recvstart = datetime.datetime.now()
        data = comm.recv(source=0, tag=rank * 2)
        stopdata = comm.recv(source=0, tag=rank * 3)
        recvstop = datetime.datetime.now()

        # if a spawned node, report communication latencies in microseconds
        startdelta = float((recvstart - startdata).microseconds)
        stopdelta = float((recvstop - stopdata).microseconds)
        transmitdelta = float((stopdata - startdata).microseconds)


    startdelta = comm.reduce(startdelta, op=MPI.SUM, root=0)
    stopdelta = comm.reduce(stopdelta, op=MPI.SUM, root=0)
    transmitdelta = comm.reduce(transmitdelta, op=MPI.SUM, root=0)

    startdelta_min = comm.reduce(startdelta, op=MPI.MIN, root=0)
    stopdelta_min = comm.reduce(stopdelta, op=MPI.MIN, root=0)
    transmitdelta_min = comm.reduce(transmitdelta, op=MPI.MIN, root=0)

    startdelta_max = comm.reduce(startdelta, op=MPI.MAX, root=0)
    stopdelta_max = comm.reduce(stopdelta, op=MPI.MAX, root=0)
    transmitdelta_max = comm.reduce(transmitdelta, op=MPI.MAX, root=0)

    if rank == 0:
        print(f'start difference (usec) : {startdelta / (size - 1):.0f} | min {startdelta_min:.0f} | max {startdelta_max:.0f} ')
        print(f'stop difference (usec) : {stopdelta / (size - 1):.0f} | min {stopdelta_min:.0f} | max {stopdelta_max:.0f} ')
        print(f'transmit difference (usec) : {transmitdelta / (size - 1):.0f} | min {transmitdelta_min:.0f} | max {transmitdelta_max:.0f} ')

    comm.Barrier()  # wait for all hosts
