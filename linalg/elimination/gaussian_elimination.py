import numpy as np
from typing import Tuple

class GaussianElimination:
    def __init__(self, A:np.ndarray):
        self.__check_A(A)
        self.A = A.astype(np.float32)
        self.P_ = np.identity(A.shape[0]) 
        self.E_ = []

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
        candidate_subs = np.argwhere(self.A[i+1:, i]!=0).flatten()
        if candidate_subs.size>0:
            idx_sub = candidate_subs[0]+(i+1)
            self.P_[[i, idx_sub]] = self.P_[[idx_sub, i]]
        else:
            raise ArithmeticError('System will face breakdown')

    def __configure_P(self)->None:
        for i in range(self.A.shape[0]):
            if self.A[i,i] == 0:
                self.__swap_P_rows(i)
            else:
                continue
    
    def __configure_E(self)->None:
        A = self.P_ @ self.A
        for i in range(1, A.shape[0]):
            for j in range(0, i):
                E = np.identity(A.shape[0])
                E[i,i-1] = -A[i,j] / A[j,j]
                self.E_.insert(0, E)
        self.E_ = np.linalg.multi_dot(self.E_) if len(self.E_)>1 else np.array(self.E_[0])
        
    def __eliminate(self, b:np.ndarray=None)->Tuple[np.ndarray] | np.ndarray:
        print(b)
        M = np.concatenate((self.A, b), axis=1) if b is not None else self.A
        M = np.linalg.multi_dot((self.E_, self.P_, M)) 
        return (M[:, :-1], M[:, -1]) if b is not None else M

    def __solve(self, b:np.ndarray)->np.ndarray:
        solution, N = np.array([0. for i in range(self.U_.shape[0])]),  -(self.U_.shape[0]+1)
        for i in range(-1, N, -1):
            subtract = np.sum(self.U_[i, i+1:]*solution[i+1:]) if i<-1 else 0
            solution[i] = (b[i] - subtract)/self.U_[i,i]
        return np.array(solution)

    def eliminate(self, b:np.ndarray=None)->Tuple[np.ndarray]:
        self.__configure_P()
        self.__configure_E()
        self.U_ = self.__eliminate(b)
        return self

    def solve(self, b:np.ndarray=None)->np.ndarray:
        if b is None:
            raise ValueError('You must specify a `b` vector')
        b = b.reshape(-1,1) if b.ndim==1 else b
        if 'U_' not in self.__dict__:
            self.eliminate(b)
            return self.__solve(*self.U_)
        else:
            if isinstance(self.U_, tuple):
                return self.__solve(*self.U_) 
            else:
                return self.__solve(self.U_, b)