import random
import copy
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

INFO = "Remplaçant la méthode append de numpy par celle native de Python"
start = time.time()
density = 0.0155  # NUmber of atom per cubic angstrom

# Set the size of the box (in Angstroms)
box_size = (15.0, 15.0, 15.0)
cubic_angstrom = box_size[0] * box_size[1] * box_size[2]
# Set the number of atoms in the box
n_atoms = int(cubic_angstrom * density)
print("Number of atoms in the box :", n_atoms)
# Set the number of Monte Carlo moves to perform
num_moves = 5000

# The maximum amount that the atom can be translated by
max_translate = 0.2  # angstroms

# Simulation temperature
temperature = 115.79  # Melting point of krypton

# Give the Lennard Jones parameters for the atoms
# (these are the OPLS parameters for Krypton)
sigma = 3.624  # angstroms
epsilon = 0.317  # kcal mol-1

# Create an array to hold the coordinates of the atoms
# coords = []
#
# # Randomly generate the coordinates of the atoms in the box
# for i in range(0,n_atoms):
#     # Note "random.uniform(x,y)" would generate a random number
#     # between x and y
#     coords.append([random.uniform(0, box_size[0]),
#                    random.uniform(0, box_size[1]),
#                    random.uniform(0, box_size[2])])

coord_x = np.random.uniform(0, box_size[0], n_atoms)
coord_y = np.random.uniform(0, box_size[1], n_atoms)
coord_z = np.random.uniform(0, box_size[2], n_atoms)
coords = np.array([coord_x, coord_y, coord_z]).T

# coords = np.random.uniform(0, box_size[0], size=(n_atoms, 3))

# @profile
def wrap_into_box(coordinates, boundaries=box_size):
    """Subroutine to wrap the coordinates into a box"""
    """ Function to replace in the box the particles."""

    for i in range(len(coordinates)):
        if coordinates[i] > boundaries[i]:
            coordinates[i] -= boundaries[i]
        if coordinates[i] < 0:
            coordinates[i] += boundaries[i]

    return coordinates

# @profile
def get_distance(coords1, coords2=None):
    """ Function to calcul the distance between two points, given their coordinates.
        If no arguments passed for the second vector, returns the modulus of the first vector.
    """
    if coords2 is None:
        coords2 = np.zeros_like(coords1)
    coords1 = np.asarray(coords1)
    coords2 = np.asarray(coords2)
    r = (coords2 - coords1) ** 2
    return np.sqrt(np.sum(r))


def get_vdw_energy(r_ij, eps_ij, ro_ij):
    r6_ij = (ro_ij / r_ij) ** 6
    return 4 * eps_ij * (r6_ij ** 2 - r6_ij)

# @profile
def make_periodic(dl, box):
    """Subroutine to apply periodic boundaries"""
    for i in range(len(dl)):
        if np.abs(dl[i]) > 0.5*box[i]:
            dl[i] -= np.copysign(box[i], dl[i])
    return dl

# @profile
def calculate_energy(_coords, eps=epsilon, sig=sigma):
    e_vdw = 0
    distance_interatoms = list()

    for i, j in itertools.combinations(range(n_atoms), 2):
        dl = _coords[j, :] - _coords[i, :]

        # Apply periodic boundaries
        dl = make_periodic(dl, box_size)

        # Calculate the distance between the atoms
        r = np.sqrt(np.sum(dl**2))

        # E_LJ = 4*epsilon[ (sigma/r)^12 - (sigma/r)^6 ]
        e_lj = get_vdw_energy(r, eps, sig)

        e_vdw += e_lj
        distance_interatoms.append(r)
        # return the total energy of the atoms

    return e_vdw, np.mean(np.array([distance_interatoms]))


# calculate kT
k_boltz = 1.987206504191549E-003  # kcal mol-1 K-1

kT = k_boltz * temperature

# The total number of accepted moves
naccept = 0

# The total number of rejected moves
nreject = 0

# Print the initial PDB file
# print_pdb(0)

energies = list()
distances = list()
# Now perform all of the moves
for move in tqdm(range(1, num_moves + 1)):

    # calculate the old energy
    old_energy, old_distance = calculate_energy(coords)
    energies.append(old_energy)
    distances.append(old_distance)
    # Pick a random atom (random.randint(x,y) picks a random
    # integer between x and y, including x and y)
    atom = np.random.randint(0, n_atoms - 1)

    # save the old coordinates
    old_coords = coords.copy()

    # Make the move - translate by a delta in each dimension
    dl = np.random.uniform(-max_translate, max_translate)

    coords[atom] += dl

    # wrap the coordinates back into the box
    coords[atom, :] = wrap_into_box(coords[atom, :])

    # calculate the new energy
    new_energy, new_distance = calculate_energy(coords)

    # Automatically accept if the energy goes down
    if new_energy <= old_energy:
        naccept += 1
        energies.append(new_energy)
        distances.append(new_distance)

    else:
        # Now apply the Monte Carlo test - compare
        # exp( -(E_new - E_old) / kT ) >= rand(0,1)
        x = np.exp(-(new_energy - old_energy) / kT)

        if x >= random.uniform(0.0, 1.0):
            naccept += 1
            total_energy = new_energy
            energies.append(total_energy)
            distances.append(new_distance)
        else:
            # reject the move - restore the old coordinates
            nreject += 1

            # restore the old conformation
            coords = copy.deepcopy(old_coords)

            total_energy = old_energy
            energies.append(total_energy)
            distances.append(old_distance)


with open("test_simulation.dat", "w") as f:
    # f.write("energy\tdistance")
    for energy, distance in zip(energies, distances):
        f.write(f"{energy}\t{distance}\n")
# # plt.figure()
# plt.title("Energy as a function of distance")
# plt.plot(distances, energies)
# plt.savefig("distance_energy.pdf")
print(f"Acceptés : {100*naccept/num_moves}%")
print(f"Rejetés : {100*nreject/num_moves}%")

# with open("time.dat", "a") as f:
#     f.write(f"Temps : {time.time() - start} - {INFO}\n")
