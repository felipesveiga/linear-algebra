import numpy as np
from linalg.elimination import GaussianElimination
from linalg.elimination.eliminate import eliminate

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

    def __jordan(self)->np.ndarray:
        M = np.concatenate((self.__gaussian_elimination.A_, self.__gaussian_elimination.b_), axis=1)
        N = M.shape[0] 
        for i in range(1, N):
            idxs_eliminate = np.argwhere(M[:i, i]).flatten()
            if idxs_eliminate.size>0:
                M = self.__eliminate_column(M, idxs_eliminate, i)
            else:
                continue
        return M

    def invert(self):
        self.__gaussian_elimination =  self.__gaussian_elimination.eliminate()
        M = self.__jordan()
        return self.__divide_pivot(M)[:, M.shape[0]:]