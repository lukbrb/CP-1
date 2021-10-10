import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("test/test_simulation.dat")
energy, distance = data.T

plt.figure()
plt.title("Energy evolution")
plt.plot(energy)
plt.yscale('linear')
# plt.xlim([0, 2000])

plt.figure()
plt.title("Average distance evolution")
plt.plot(distance)
plt.figure()
plt.title("Distance vs energy")
plt.plot(energy, distance, '.')
plt.xscale('log')
plt.show()
