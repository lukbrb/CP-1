import numpy as np
import itertools
import time
import matplotlib.pyplot as plt

N_ATOMS = 25
VDW_EPS = 0.0157
sigma = 3.624
epsilon = 0.317
Kb = 8.314462618
E0_O = 118 * Kb
d_0 = 0.346  # oxygen diameter in nm
RADIUS = 0.6
KB = 1.987206504191549E-003     # kcal mol-1 K-1
TEMPERATURE = 298.15


def get_distance(coords1, coords2):
    coords1 = np.asarray(coords1)
    coords2 = np.asarray(coords2)
    r = (coords2 - coords1) ** 2
    return np.sqrt(np.sum(r))


def get_vdw_energy(r_ij, eps_ij, ro_ij):
    r6_ij = (ro_ij / r_ij) ** 6
    return 4 * eps_ij * (r6_ij ** 2 - r6_ij)


def get_vdw_energy_n(coords, eps=epsilon, sig=sigma):
    e_vdw = 0
    distance_interatoms = np.array([])

    for i, j in itertools.combinations(range(N_ATOMS), 2):
        atom1, atom2 = coords[i], coords[j]
        r_ij = get_distance(atom1, atom2)
        radius_atoms = sig + sig
        eps_ij = eps * eps
        e_vdw += get_vdw_energy(r_ij, eps_ij, radius_atoms)
        distance_interatoms = np.append(distance_interatoms, r_ij)

        return e_vdw, np.mean(distance_interatoms)


def monte_carlo_simulation(disp_mag):

    coordinates = np.random.uniform(0.0, 1, size=(N_ATOMS, 3))
    energie, distance_ = get_vdw_energy_n(coordinates)
    print("ÉNERGIE : ", energie)
    print("DISTANCE : ", distance_)
    return energie, distance_


def choose_random_atom(disp_mag):
    """ Return the displacement of a an atom randomly picked in the gas.
        Args :
            - coords : coordinates of all the particles (atoms) in the gas.

        return : An array with the displacement an an atom.
    """
    random_atom = np.random.randint(N_ATOMS)
    # Random displacement over each axis
    dx = np.random.uniform(-disp_mag, disp_mag)
    dy = np.random.uniform(-disp_mag, disp_mag)
    dz = np.random.uniform(-disp_mag, disp_mag)

    return random_atom, [dx, dy, dz]


def move_random_atom(coords, atom, disp):
    random_atom = atom
    dx, dy, dz = disp
    coords[random_atom, :] = dx, dy, dz

    return coords


def update_positions(coords, disp):
    return coords + disp


def check_boundary(coords, box_size, radius):
    """ Function to replace in the box the particles."""

    x_coord, y_coord, z_coord = coords[:, 0], coords[:, 1], coords[:, 2]
    x_bound, y_bound, z_bound = box_size

    x_coord[x_coord > x_bound] = x_bound - radius
    x_coord[x_coord < 0] = x_bound + radius

    y_coord[y_coord > y_bound] = y_bound - radius
    y_coord[y_coord < 0] = y_bound + radius

    z_coord[z_coord > z_bound] = z_bound - radius
    z_coord[z_coord < 0] = z_bound + radius

    _coords = np.array([x_coord, y_coord, z_coord]).T

    return _coords


# noinspection PyPep8Naming
def simulate(n_conf=10_000, eps=epsilon, sig=sigma):
    box_size = [15, 15, 15]  # Dimension of the box in Angstrom
    disp_mag = 0.5  # Magnitude of the displacement in Angstrom

    coordinates = np.random.normal(0, box_size[0], size=(N_ATOMS, 3))   # Works if the box is a cube
                                                # Should create three arrays for each dimension instead

    n_accept = 0
    n_reject = 0
    proba_list = list()

    energie_prev, distance_prev = get_vdw_energy_n(coordinates, eps, sig)
    _energies = np.array([energie_prev])
    _distances = np.array([distance_prev])
    new_coordinates = coordinates.copy()

    for conf in range(n_conf):

        # rand_disp = np.random.normal(loc=0.0, scale=disp_mag, size=(N_ATOMS, 3))  # random displacement

        # coordinates = update_positions(coordinates, rand_disp)
        atom_to_move, displacement = choose_random_atom(disp_mag)
        new_coordinates = move_random_atom(new_coordinates, atom_to_move, displacement)
        new_coordinates = check_boundary(new_coordinates, box_size, sigma)

        energie_new, distance_new = get_vdw_energy_n(new_coordinates, eps, sig)
        _energies = np.append(_energies, energie_new)
        _distances = np.append(_distances, distance_new)
        dE = energie_new - energie_prev
        if dE < 0:
            n_accept += 1
        else:
            proba = np.exp(min(1, -dE/(KB * TEMPERATURE)))
            proba_list.append(proba)

            if proba >= np.random.random():
                conf += 1
                n_accept += 1
                energie_prev = energie_new
            else:
                # coordinates = update_positions(coordinates, -rand_disp)  # We go back to the same coordinates
                new_coordinates = coordinates
                n_reject += 1

    print(proba_list)
    return _energies, _distances


if __name__ == '__main__':

    # start = time.perf_counter()
    # energies = np.array([])
    # deplacements = np.linspace(0.01, 1.5, 300)
    # for deplacement in deplacements:
    #     energie, distance = monte_carlo_simulation(deplacement)
    #     energies = np.append(energies, energie)
    # end = time.perf_counter()
    # print(f"Calculé en {end - start} secondes")
    #
    # plt.figure()
    # plt.loglog(deplacements, energies, "o", label="energy")
    # plt.show()

    # monte_carlo_simulation(1/4)

    energies, distances = simulate(5000)

    with open("data_simulation.dat", "w") as f:
        # f.write("energy\tdistance")
        for energy, distance in zip(energies, distances):
            f.write(f"{energy}\t{distance}\n")
    plt.figure()
    plt.title("Energy as a function of distance")
    plt.plot(distances, energies)
    plt.savefig("distance_energy.pdf")

    # x = np.linspace(0, 17, 50)
    # y = x.copy()
    # z = x.copy()
    #
    # res = check_boundary(np.array([x, y, z]), [15, 15, 15], sigma)
    # print(res.shape)
    # res2 = check_boundary(res, [15, 15, 15], sigma)
    # print(res2.shape)
