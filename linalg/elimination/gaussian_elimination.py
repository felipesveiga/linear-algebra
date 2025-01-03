import numpy as np

class GaussianElimination:
    def __init__(self, A:np.ndarray):
        self.__check_A(A)
        self.A = A.astype(np.float32)
        self._P = np.identity(A.shape[0]) 
        self._E = []#[np.identity(A.shape[0]) for i in range(1, A.shape[0]) for j in range(i)]

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
            self._P[[i, idx_sub]] = self._P[[idx_sub, i]]
        else:
            raise ArithmeticError('System will face breakdown')

    def __configure_P(self)->None:
        for i in range(self.A.shape[0]):
            if self.A[i,i] == 0:
                self.__swap_P_rows(i)
            else:
                continue
    
    def __configure_E(self)->None:
        A = self._P @ self.A
        for i in range(1, A.shape[0]):
            for j in range(0, i):
                E = np.identity(A.shape[0])
                E[i,i-1] = -A[i,j] / A[j,j]
                self._E.insert(0, E)
        self._E = np.linalg.multi_dot(self._E)
        
    def eliminate(self, b:np.ndarray=None):
        self.__configure_P()
        self.__configure_E()
        M = np.concatenate((self.A, b), axis=0) if b else self.A
        return np.linalg.multi_dot((self._E, self._P, M)) 