""" Class Monte Carlo simulation """
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from .MonoAtomicGas import atom, gas, parameters


class MonteCarlo:
    def __init__(self, n_atoms, atom_name="Kr", ndim=3, box_dims=(15, 15, 15)):
        self.N_ATOMS = n_atoms
        self.BOX_DIMS = box_dims
        self.NDIM = ndim

        self.KB = parameters.k_boltz
        self.ATOM_NAME = atom_name
        self.TOTCONF = 10_000
        self.TEMPERATURE = 298.15
        self.ENERGY = 0
        self.MAG_MOVE = 0.5
        self.N_ACCEPT = 0
        self.N_REJECT = 0
        self.kT = self.KB * self.TEMPERATURE

        self.filename = f"data/sim_results_{int(self.TEMPERATURE)}.dat"
        self.comments = str()  # Comment for the "data/info.txt" file.
        self.flag = "INFO"

    def _print_params(self):
        """ Print the parameters used for the simulation. """

        texte = f""" \t\t METROPOLIS MONTE CARLO SIMULATION
        ====================================================
        PARAMETERS :
        \t-Type of atom : {self.ATOM_NAME}
        \t-Number of atoms : {self.N_ATOMS}
        \t-Number of dimensions : {self.NDIM}
        \t-Dimensions of the box : {self.BOX_DIMS}
        \t-Temperature : {self.TEMPERATURE} [K]
        \t-Number of trials : {self.TOTCONF}
        \t-Displacement magnitude : {self.MAG_MOVE}
        ====================================================
        """
        print(texte)

    def _create_atoms(self):
        coords = np.zeros(len(self.BOX_DIMS))
        for i in range(len(self.BOX_DIMS)):
            coords[i] = np.random.uniform(0, self.BOX_DIMS[i])
        return atom.Atom(atom_name=self.ATOM_NAME, coordinates=coords)

    def _create_gas(self):
        liste_atoms = [self._create_atoms() for _ in range(self.N_ATOMS)]
        return gas.Gaz(liste_atoms)

    def run(self):
        self._write_simulation_info()
        start = time.time()
        self._print_params()
        gaz = self._create_gas()
        energies = list()
        distances = list()

        for trial in tqdm(range(1, self.TOTCONF + 1)):
            old_energy, old_distance = gaz.get_energy()
            energies.append(old_energy)
            distances.append(old_distance)
            random_index = np.random.randint(0, self.N_ATOMS)
            random_atom = gaz.atoms[random_index]

            # save the old coordinates
            old_coords = gaz.coordinates.copy()

            # Make the move - translate by a delta in each dimension
            dl = np.random.uniform(-self.MAG_MOVE, self.MAG_MOVE, self.NDIM)

            random_atom.coordinates += dl

            # make sure the coordinates of the atom are still in the box
            random_atom.wrap_into_box(self.BOX_DIMS)
            # gaz.atoms[random_index] = random_atom   # We could update the parameters of the atom but all the data is
            # now encrypted in gaz.coordinates
            gaz.coordinates[random_index] = random_atom.coordinates
            # calculate the new energy
            new_energy, new_distance = gaz.get_energy()

            # Automatically accept if the energy goes down
            if new_energy <= old_energy:
                energies.append(new_energy)
                distances.append(new_distance)
                self.N_ACCEPT += 1

            else:
                # Now apply the Monte Carlo test - compare
                x = np.exp(-(new_energy - old_energy) / self.kT)

                if x >= np.random.uniform(0.0, 1.0):
                    self.N_ACCEPT += 1
                    energies.append(new_energy)
                    distances.append(new_distance)
                else:
                    # reject the move - restore the old coordinates
                    self.N_REJECT += 1

                    # restore the old conformation
                    gaz.coordinates = old_coords.copy()

                    total_energy = old_energy
                    energies.append(total_energy)
                    distances.append(old_distance)

        distances = np.sqrt(np.array(distances))  # compute the real distance
        density = self._compute_average_density(distances)

        print(f"Acceptés : {100 * self.N_ACCEPT / self.TOTCONF}%")
        print(f"Rejetés : {100 * self.N_REJECT / self.TOTCONF}%")

        # We don't want to save all the data if we're just testing the script
        if self.flag.upper() == "INFO":
            print("Computation done, saving data ...")
            # SAVE EQULIBRIUM COORDINATES
            np.savetxt(f"data/coordinates_{self.TEMPERATURE}.dat", gaz.coordinates)

            # PLOTS
            self._save_plot(energies, "energy", labels=("Moves", "Energy [kcal/mol]"), scale=("linear", "log"))
            self._save_plot(distances, "distances", labels=("Moves", "Average distance [Angstrom]"))
            self._save_plot(energies, "energy_vs_distance", x_tab=distances, labels=("Average distance [A]",
                                                                                     "Energy [kcal/mol]"),
                            scale=("linear", "log"))
            self._save_plot(density, "density", labels=("Moves", "Number of atom per $A^3$"))
            self._save_plot(energies, "energy_vs_density", x_tab=density, labels=("Average density [$N/A^3$]",
                                                                                  "Energy [kcal/mol]"),
                            scale=("linear", "log"))
            # SAVING DATA
            data_filename = f"data/sim_results_{int(self.TEMPERATURE)}.dat"

            self._write_data(data_filename, energies, distances)

        print(f"Simulation is done\nTotal time : {time.time() - start}")

    @staticmethod
    def _compute_average_density(average_dist):
        inverted_average_dist = 1/np.asarray(average_dist)
        return (inverted_average_dist + 1) * (4 * inverted_average_dist + (inverted_average_dist - 1)**2)

    @staticmethod
    def _write_data(filename, x_tab, y_tab):
        with open(filename, "w") as f:
            for energy, distance in zip(x_tab, y_tab):
                f.write(f"{energy}\t{distance}\n")

    def _save_plot(self, y_tab, filename, x_tab=None, title=str(), labels=(str(), str()), scale=("linear", "linear")):
        filename = f"plots/{filename}_{int(self.TEMPERATURE)}.pdf"
        if x_tab is None:
            x_tab = np.arange(len(y_tab))

        plt.figure()
        plt.title(title)
        plt.plot(x_tab, y_tab, ".", label=f"T = {self.TEMPERATURE} K")
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.xscale(scale[0])
        plt.yscale(scale[1])
        plt.savefig(filename)

    def _write_simulation_info(self):
        with open("data/info.txt", "a") as f:
            f.write(f"[{self.flag.upper()}]\tDATE: {time.asctime()}\tTOTCONF: {self.TOTCONF}\t"
                    f"TEMPERATURE: {self.TEMPERATURE}\tNATOMS: {self.N_ATOMS}\tCOMMENTS: {self.comments}\n")
