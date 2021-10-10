""" Class Gas """
from typing import List
import numpy as np
import itertools
from .atom import Atom


class Gaz:
    def __init__(self, atoms: List[Atom], box_dims=(15, 15, 15), ndim=3):
        self.N_ATOMS = len(atoms)
        self.BOX_DIMS = box_dims
        self.NDIM = ndim
        self.E_TOT = 0
        self.atoms = np.array(atoms)
        self.coordinates = np.array([atom.coordinates for atom in self.atoms])

    def get_energy(self):
        e_vdw = 0
        distance_interatoms = list()

        for i, j in itertools.combinations(range(self.N_ATOMS), 2):
            atom1, atom2 = self.atoms[i], self.atoms[j]
            dist_axis = atom2.coordinates - atom1.coordinates

            # Apply periodic boundaries
            dist_axis = self._make_periodic(dist_axis, self.BOX_DIMS)

            # Calculate the distance between the atoms
            # r = np.sqrt(np.sum(dist_axis ** 2))
            r2 = np.sum(dist_axis ** 2)  # We don't need the real distance and the sqrt is computational expensive
            eps_12 = atom1.epsilon * atom2.epsilon
            sig_12 = atom1.sigma + atom2.sigma
            e_vdw += self._vdw_interactions(r2, eps_12, sig_12)

            distance_interatoms.append(r2)  # We will take the square root at the end

        return e_vdw, np.mean(np.array([distance_interatoms]))

    def _vdw_interactions(self, r_12, eps_12, sig_12):
        r6_12 = sig_12**6 / r_12**3
        return 4 * eps_12 * (r6_12 ** 2 - r6_12)

    def _make_periodic(self, dl, box):
        """Subroutine to apply periodic boundaries"""
        assert len(dl) == len(box), " Problem with the dimensions of the box"
        for i in range(len(dl)):
            if np.abs(dl[i]) > 0.5*box[i]:
                dl[i] -= np.copysign(box[i], dl[i])
        return dl
