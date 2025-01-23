import numpy as np
from warnings import warn
from typing import Tuple

class GaussianElimination:
    '''
        Proceeds the Gaussian Elimination of a linear system.

        Parameter
        ---------
        `A`: `np.ndarray`
            The coefficient matrix.

        Attributes
        ----------
        `P_`: np.ndarray
            The row reorder matrix.
        `E_`: np.ndarray
            The elimination matrix.
        `U_`: `np.ndarray` | Tuple[np.ndarray]
            The eliminated coefficient matrix, or a tuple with such array
            along with the eliminated target vector.
    '''
    def __init__(self, A:np.ndarray):
        self.__check_A(A)
        self.A = A.astype(np.float32)
        self.P_ = np.identity(A.shape[0]) 
        self.E_ = []
        self.U_ = None

    @staticmethod
    def __check_A(A:np.ndarray)->None:
        ''' 
            Proceeds with arguments validation.

            Parameter
            ---------
            `A`: `np.ndarray`
                The coefficients matrix.
        '''
        assert A.shape[0] == A.shape[1], '`A` must be a square matrix'
        assert np.linalg.det(A) != 0, 'Singular Matrix'  

    def __swap_P_rows(self, i:int)->None:
        '''
            Exchange a given row by another in case of the first's
            pivot is 0

            Parameter
            ---------
            `i`: int
                The index of the row that might face exchange.
        '''
        candidate_subs = np.argwhere(self.A[i+1:, i]!=0).flatten()
        if candidate_subs.size>0:
            idx_sub = candidate_subs[0]+(i+1)
            self.P_[[i, idx_sub]] = self.P_[[idx_sub, i]]
        else:
            raise ArithmeticError('System will face breakdown')

    def __configure_P(self)->None:
        '''
            Creates the class' `P_` matrix specialized in row
            exchanges.
        '''
        for i in range(self.A.shape[0]):
            if self.A[i,i] == 0:
                self.__swap_P_rows(i)
            else:
                continue
    
    def __configure_E(self)->None:
        '''
            Creates the elimination matrix `E_`. 
        '''
        A = self.P_ @ self.A
        for i in range(1, A.shape[0]):
            for j in range(0, i):
                E = np.identity(A.shape[0])
                E[i,i-1] = -A[i,j] / A[j,j]
                self.E_.insert(0, E)
        self.E_ = np.linalg.multi_dot(self.E_) if len(self.E_)>1 else np.array(self.E_[0])
        
    def __eliminate(self, b:np.ndarray=None)->Tuple[np.ndarray] | np.ndarray:
        '''
            Creates the eliminated coefficient matrix `U_`.

            Parameter
            ---------
            `b`: `np.ndarray` | None
                The target variables vector.
            
            Returns
            -------
            The eliminated coefficients matrix and target vector, if this last one is provided.
        '''
        M = np.concatenate((self.A, b), axis=1) if b is not None else self.A
        M = np.linalg.multi_dot((self.E_, self.P_, M)) 
        return (M[:, :-1], M[:, -1]) if b is not None else M

    def __solve(self, A:np.ndarray, b:np.ndarray)->np.ndarray:
        '''
            Solves the provided linear system.

            Parameters
            ----------
            `A`: `np.ndarray`
                The coefficients matrix.
            `b`: `np.ndarray`
                The target vector.

            Returns
            -------
            The solution vector.
        '''
        solution, N = np.array([0. for i in range(A.shape[0])]),  -(A.shape[0]+1)
        for i in range(-1, N, -1):
            subtract = np.sum(A[i, i+1:]*solution[i+1:]) if i<-1 else 0
            solution[i] = (b[i] - subtract)/A[i,i]
        return np.array(solution)

    def eliminate(self, b:np.ndarray=None):
        '''
           Proceeds the Gaussian Elimination.
            
           By invoking this method, the class is going 
           to acquire the new attributes `P_` and `E_`. 

           Parameter
           ---------
           `b`: `np.ndarray` | None
                The target vector.
        '''
        self.__configure_P()
        self.__configure_E()
        self.U_ = self.__eliminate(b)
        return self

    def solve(self, b:np.ndarray=None)->np.ndarray:
        '''
            Solves the provided linear system.

            Parameter
            ---------
            `b`: `np.ndarray` | None
                The target vector. Must be declared only if
                it was not passed in the `eliminate` or you are
                directly solving the system.
            
            Returns
            -------
            An array with the solutions. 
        '''
        if b is None:
            if isinstance(self.U_, tuple):
                return self.__solve(*self.U_) 
            else:
                raise ValueError('You must specify a `b` vector')
        else:
            if isinstance(self.U_, np.ndarray):
                b = np.linalg.multi_dot((self.E_, self.P_, b))
                return self.__solve(self.U_, b) 
            else:
                warn('Elimination was already carried out considedring `b`. You did not have to inform it again on `solve`', UserWarning)
                return self.__solve(*self.U_)