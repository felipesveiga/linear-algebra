import numpy as np
from elimination import GaussianElimination

class GaussJordanElimination:
    def __init__(self, A:np.ndarray):
        self.__gaussian_elimination = GaussianElimination(A, np.identity(A.shape[0]))

    def invert(self):
        gaussian_elimination = self.__gaussian_elimination.eliminate()
        M = np.concatenate((gaussian_elimination.A_, gaussian_elimination.b_), axis=1)