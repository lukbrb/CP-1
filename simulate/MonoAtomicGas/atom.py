""" Class Atom"""
import numpy as np
from .parameters import VAN_DER_WAALS


class Atom:
    def __init__(self, atom_name, coordinates):
        # TODO: Prévenir une KeyError
        self.epsilon, self.sigma = VAN_DER_WAALS[atom_name]
        self.coordinates = coordinates
        self.NDIM = len(self.coordinates)

    def wrap_into_box(self, boundaries):
        for i in range(self.NDIM):
            if self.coordinates[i] > boundaries[i]:
                self.coordinates[i] -= boundaries[i]
            if self.coordinates[i] < 0:
                self.coordinates[i] += boundaries[i]
