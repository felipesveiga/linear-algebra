import numpy as np
from linalg.elimination import GaussianElimination 
from linalg.elimination import GaussJordanElimination

class LU:
    def __init__(self, A:np.ndarray):
        self.A = A
        self._gaussian_elimination = GaussianElimination(A).eliminate()

    def factorize(self):
        L = np.linalg.multi_dot((self._gaussian_elimination.P_, self._gaussian_elimination.E_))
        return L,self._gaussian_elimination.U_