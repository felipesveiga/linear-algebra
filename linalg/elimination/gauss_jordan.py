import numpy as np
from linalg.elimination import GaussianElimination
from linalg.elimination import eliminate

class GaussJordanElimination:
    def __init__(self, A:np.ndarray):
        self.__gaussian_elimination = GaussianElimination(A, np.identity(A.shape[0]))

    @staticmethod
    def __eliminate_column(M:np.ndarray, idxs_eliminate:np.ndarray[int], i:int)->np.ndarray:
        for j in idxs_eliminate:
            M[j] = eliminate(M, i, j, i)
        return M

    @staticmethod
    def __divide_pivot(M:np.ndarray)->np.ndarray:
        for i in range(M.shape[0]):
            M[i] /= M[i,i]
        return M

    def invert(self):
        gaussian_elimination = self.__gaussian_elimination.eliminate()
        M = np.concatenate((gaussian_elimination.A_, gaussian_elimination.b_), axis=1)
        for i in range(-(M.shape[0]-1), 0):
            idxs_eliminate = np.argwhere(M[:i, i]).flatten()
            if idxs_eliminate.size>0:
                M = self.__eliminate_column(M, idxs_eliminate, i)
            else:
                continue
        return self.__divide_pivot(M) 