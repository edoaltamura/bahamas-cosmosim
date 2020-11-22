from mpi4py import MPI
import datetime
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# master process
if rank == 0:
    data = np.ones(1000000)
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
    startdelta = recvstart - startdata
    stopdelta = recvstop - stopdata
    transmitdelta = stopdata - startdata


startdelta = comm.reduce(startdelta, op=MPI.SUM, root=0)
stopdelta = comm.reduce(stopdelta, op=MPI.SUM, root=0)
transmitdelta = comm.reduce(transmitdelta, op=MPI.SUM, root=0)

if rank == 0:
    print('start difference (uS) : ' + str(startdelta.microseconds / (size - 1)))
    print('stop difference (uS) : ' + str(stopdelta.microseconds / (size - 1)))
    print('transmit difference (uS) : ' + str(transmitdelta.microseconds / (size - 1)) + '\n')

comm.Barrier()  # wait for all hosts
