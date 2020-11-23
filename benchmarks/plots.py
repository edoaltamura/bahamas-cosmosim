import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('../../../PhD_Y2/benchmark.txt', dtype='float64').T
plt.scatter(data[0], data[1], s=2)
plt.scatter(data[0], data[2], s=2)
plt.xlabel('Message size [Bytes]')
plt.ylabel('Transmission time [milliseconds]')
plt.axvline(x=(100*1024**2))
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.scatter(data[0], data[0] / data[1] * 8.e3 / 1024 ** 2, s=2)
plt.scatter(data[0], data[0] / data[2] * 8.e3 / 1024 ** 2, s=2)
plt.xlabel('Message size [Bytes]')
plt.ylabel('Interconnect speed [MBps]')
plt.axhline(y=1024)
plt.axvline(x=(100*1024**2))
plt.xscale('log')
plt.yscale('log')
plt.show()