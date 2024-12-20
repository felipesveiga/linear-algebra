import numpy as np
from linalg.elimination import GaussianElimination
from linalg.elimination.eliminate import eliminate

class GaussJordanElimination:
    '''
        Applies the Gauss-Jordan Elimination

        Parameter
        ---------
        `A`: `np.ndarray`
            The matrix to be inverted.
        
        Method
        ------
        `invert`: Finds A^{-1}.
    '''
    def __init__(self, A:np.ndarray):
        self.__gaussian_elimination = GaussianElimination(A, np.identity(A.shape[0]))

    @staticmethod
    def __eliminate_column(M:np.ndarray, idxs_eliminate:np.ndarray[int], i:int)->np.ndarray:
        '''
            Eliminates all components above a certain pivot.

            Parameters
            ----------
            `M`: `np.ndarray`
                The block-matrix containing A and I.
            `idxs_eliminate`: `np.ndarray[int]`
                An array with the components' indices to be eliminated.
            `i`: int
                The index of the column in which elimination will take place. 
            
            Returns
            -------
            The matrix M with the components from column i eliminated.
        '''
        for j in idxs_eliminate:
            M[j] = eliminate(M, i, j, i)
        return M

    @staticmethod
    def __divide_pivot(M:np.ndarray)->np.ndarray:
        '''
            Divides each row from a provided block-matrix by its pivot.

            Parameter
            ---------
            `M`: `np.ndarray`
                The block-matrix.
            
            Returns
            -------
            The "normalized" block matrix.
        '''
        for i in range(M.shape[0]):
            M[i] /= M[i,i]
        return M

    def __jordan(self)->np.ndarray:
        '''
            Applies the Jordan Elimination step.
        '''
        M = np.concatenate((self.__gaussian_elimination.A_, self.__gaussian_elimination.b_), axis=1)
        N = M.shape[0] 
        for i in range(1, N):
            idxs_eliminate = np.argwhere(M[:i, i]).flatten()
            if idxs_eliminate.size>0:
                M = self.__eliminate_column(M, idxs_eliminate, i)
            else:
                continue
        return self.__divide_pivot(M)[:, M.shape[0]:]

    def invert(self):
        '''
            Inverts the matrix by Gauss-Jordan Elimination
        '''
        self.__gaussian_elimination =  self.__gaussian_elimination.eliminate()
        return self.__jordan()