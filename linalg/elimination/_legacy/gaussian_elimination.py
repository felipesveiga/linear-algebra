import numpy as np

class GaussianElimination:
    '''
        Solves square linear systems by applying the Gaussian Elimination algorithm

        Parameters
        ----------
        `A`: `np.ndarray`
            The coefficients matrix.
        `b`: `np.ndarray`
            The results vector.

        Methods
        -------
        `eliminate`: Applies the Gaussian Elimination over `A`
        `solve`: Solves the system.

        Attributes
        ----------
        `A_`: The matrix `A` after elimination is applied (also known as U).
        `b_`: The vector `b` after elimination is applied.
    '''
    def __init__(self, A:np.ndarray, b:np.ndarray):
        self.__check_args(A,b)
        A, b = A.astype(np.float32), b.astype(np.float32)
        self.A, self.A_ = A, A
        self.b, self.b_ = b, b

    @staticmethod
    def __check_args(A:np.ndarray, b:np.ndarray)->None:
        ''' 
            Proceeds with arguments validation.

            Parameters
            ----------
            `A`: `np.ndarray`
                The coefficients matrix.
            `b`: `np.ndarray`
                The results vector.
        '''
        assert A.shape[0] == A.shape[1], '`A` must be a square matrix'
        assert np.linalg.det(A) != 0, 'Singular Matrix'  
        assert A.shape[0] == b.shape[0], '`b`\'s dimensionality must match `A`\'s number of row vectors'

    def __substitute(self, i:int, idx_sub:int)->None:
        '''
            Applies the replacement of lines, in the event of a temporary breakdown.

            Parameters
            ----------
            `i`: int
                Index of the line to be replaced.
            `idx_sub`: int
                Index of the replacement line.
        '''
        self.A_[[i, idx_sub]] = self.A_[[idx_sub, i]]
        self.b_[[i, idx_sub]] = self.b_[[idx_sub, i]]
        
    def __check_breakdown(self, i:int)->float:
        '''
            Checks whether the system is going to face breakdown.

            Parameter
            ---------
            `i`: int
                The index of the currently scrutinized row.

            Returns
            -------
            The value of the real pivot of the interation.
        '''
        if self.A_[i,i] == 0:
            candidate_subs = np.argwhere(self.A_[i+1:, i]!=0).flatten()
            if candidate_subs.size>0:
                idx_sub = candidate_subs[0]+(i+1)
                self.__substitute(i, idx_sub)
            else:
                raise ArithmeticError('System will face breakdown')
        return self.A_[i,i]
    
    def eliminate(self):
        '''
            Applies the Gaussian Elimination. 
        '''
        for i in range(self.A_.shape[0]):
            pivot = self.__check_breakdown(i)
            for j in range(i+1, self.A_.shape[0]):
                multiplier = self.A_[j,i]/pivot
                self.A_[j] = self.A_[j]- (self.A_[i] * multiplier)
                self.b_[j] = self.b_[j]- (self.b_[i] * multiplier)
        return self

    def solve(self)->np.ndarray:
        '''
            Solves the presented system.
        '''
        self.eliminate()
        solution, N = np.array([0. for i in range(self.A_.shape[0])]),  -(self.A_.shape[0]+1)
        for i in range(-1, N, -1):
            subtract = np.sum(self.A_[i, i+1:]*solution[i+1:]) if i<-1 else 0
            solution[i] = (self.b_[i] - subtract)/self.A_[i,i]
        return np.array(solution)