from mpi4py import MPI
import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:  # launching process
    start = datetime.datetime.now()
    comm.send(start, dest=1, tag=11)
    stop = datetime.datetime.now()
    comm.send(stop, dest=1, tag=12)
elif rank == 1:  # spawned processes
    startdata = comm.recv(source=0, tag=11)
    recvstart = datetime.datetime.now()
    stopdata = comm.recv(source=0, tag=12)
    recvstop = datetime.datetime.now()

comm.Barrier()  # wait for all hosts

# if a spawned node, report communication latencies in microseconds
if rank == 1:
    startdelta = recvstart - startdata
    print('start difference (uS) : ' + str(startdelta.microseconds) + '\n')
    stopdelta = recvstop - stopdata
    print('stop difference (uS) : ' + str(stopdelta.microseconds) + '\n')
    transmitdelta = stopdata - startdata
    print('transmit difference (uS) : ' + str(transmitdelta.microseconds) + '\n')
