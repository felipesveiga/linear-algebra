import numpy as np

def eliminate(M:np.ndarray, i:int, j:int, p:int)->np.ndarray:
    '''
        Eliminates a given matrix row's component based on the component of
        same position from another row.

        Parameters
        ----------
        `M`: `np.ndarray`
            The matrix.
        `i`: int
            The index of the row that will base elimination.
        `j`: int
            The index of the row to be elimated.
        `p`: int
            The index of the component to be eliminated.
        
        Returns
        -------
        The row vector in eliminated form.
    '''
    multiplier = M[j, p] / M[i,p]
    return M[j] - (multiplier * M[i])