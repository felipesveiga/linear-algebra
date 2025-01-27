import numpy as np
from linalg.elimination import GaussianElimination 
from linalg.elimination import GaussJordanElimination
from typing import List

class LU:
    '''
        Proceeds the LU decomposition with partial pivoting.

        Parameter
        ---------
        `A`: np.ndarray`
            The matrix to be factorized.

        Attribute
        ---------
        `gaussian_elmination_`: `GaussianElimination` 
            The `elimination.GaussianElimination` object used in the matrix elimination. 
            The user can access both its swap and elmination matrices.

    '''
    def __init__(self, A:np.ndarray):
        self.A = A
        self.gaussian_elimination_ = GaussianElimination(A).eliminate()

    def __d_u(self)->List[np.ndarray]:
        '''
            Breaks the U matrix into DU form.
        '''
        D, U = np.zeros([self.gaussian_elimination_.U_.shape[0] for _ in range(2)]), self.gaussian_elimination_.U_
        for i in range(U.shape[0]):
            d = U[i,i]
            D[i,i] = d
            U[i] = U[i]/d 
        return list((D, U))

    def factorize(self, d_u:bool=False)->List[np.ndarray]:
        '''
            Makes the LU decomposition.

            Parameter
            ---------
            `d_u`: bool, defaults to False
                A boolean indicating whether the factorization be returned 
                in the LDU format.

            Returns
            -------
            A list containing the products of the factorization.
        '''
        L = np.linalg.multi_dot((self.gaussian_elimination_.P_, self.gaussian_elimination_.E_))
        L = GaussJordanElimination(L).invert()
        return [L] + self.__d_u() if d_u else [L] + [self.gaussian_elimination_.U_]