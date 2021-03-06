import matplotlib

matplotlib.use('Agg')

from mpi4py import MPI
import datetime
import numpy as np
from matplotlib import pyplot as plt
import resource

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


for iteration, msg_length in enumerate(np.logspace(4., 8, 20, dtype=np.int)):

    msg_bytes = None

    startdelta_sum = 0.
    stopdelta_sum = 0.
    transmitdelta_sum = 0.
    startdelta_max = 0.
    stopdelta_max = 0.
    transmitdelta_max = 0.

    # master process
    if rank == 0:
        data = np.ones((msg_length, 3))
        msg_bytes = data.nbytes
        print(f"\n({iteration + 1}/100) Message size: {sizeof_fmt(msg_bytes)}")
        startdelta = 0.
        stopdelta = 0.
        transmitdelta = 0.

        # master process sends data to worker processes by
        # going through the ranks of all worker processes
        for i in range(1, size):
            comm.send(datetime.datetime.now(), dest=i, tag=i)
            comm.send(data, dest=i, tag=i * 2)
            comm.send(datetime.datetime.now(), dest=i, tag=i * 3)
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # worker processes
    else:
        # each worker process receives data from master process
        startdata = comm.recv(source=0, tag=rank)
        recvstart = datetime.datetime.now()
        data = comm.recv(source=0, tag=rank * 2)
        stopdata = comm.recv(source=0, tag=rank * 3)
        recvstop = datetime.datetime.now()

        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # if a spawned node, report communication latencies in microseconds
        startdelta = float((recvstart - startdata).microseconds) * 1e-3
        stopdelta = float((recvstop - stopdata).microseconds) * 1e-3
        transmitdelta = float((recvstop - startdata).microseconds) * 1e-3

    startdelta_sum = comm.reduce(startdelta, op=MPI.SUM, root=0)
    stopdelta_sum = comm.reduce(stopdelta, op=MPI.SUM, root=0)
    transmitdelta_sum = comm.reduce(transmitdelta, op=MPI.SUM, root=0)

    startdelta_max = comm.reduce(startdelta, op=MPI.MAX, root=0)
    stopdelta_max = comm.reduce(stopdelta, op=MPI.MAX, root=0)
    transmitdelta_max = comm.reduce(transmitdelta, op=MPI.MAX, root=0)

    msg_bytes = comm.bcast(msg_bytes, root=0)
    shared_memory = comm.reduce(mem, op=MPI.SUM, root=0)

    if rank == 0:
        with open('benchmark.txt', 'a+') as out_file:
            print(f"{msg_bytes} {transmitdelta_sum / (size - 1)} {transmitdelta_max}", file=out_file)
        print(f'start difference (msec) : {startdelta_sum / (size - 1):.2f} | max {startdelta_max:.2f} ')
        print(f'stop difference (msec) : {stopdelta_sum / (size - 1):.2f} | max {stopdelta_max:.2f} ')
        print(f'transmit difference (msec) : {transmitdelta_sum / (size - 1):.2f} | max {transmitdelta_max:.2f} ')
        print(f"Shared memory: {sizeof_fmt(shared_memory)}")

    comm.Barrier()  # wait for all hosts

if rank == 0:
    data = np.loadtxt('benchmark.txt').T
    plt.scatter(data[0], data[1], s=2)
    plt.scatter(data[0], data[2], s=2)
    plt.xlabel('Message size [Bytes]')
    plt.ylabel('Transmission time [milliseconds]')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('benchmark.png')
