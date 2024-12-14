import numpy as np
from utils.core.core import _assure_dimension

def dot(x:np.ndarray, y:np.ndarray)->float:
    '''
        Computes the dot product between two arrays.

        Parameters
        ----------
        `x`: `np.ndarray`
            The first array.
        `y`: `np.ndarray`
            The second array.

        Returns
        -------
        The result of the dot product.
    '''
    assert len(x.shape) == len(y.shape) == 1, 'You must provide vectors as inputs'
    assert x.shape == y.shape, 'Input vectors must have the same shape'
    x, y = x.astype(np.float32), y.astype(np.float32)
    output = 0
    for i in range(x.shape[0]):
        output += (x[i] * y[i])
    return output

def matmul(X:np.ndarray, Y:np.ndarray)->np.ndarray:
    '''
        Executes matrix multiplication between two given arrays.

        Parameters
        ----------
        `X`: `np.ndarray`
            The first array.
        `Y`: `np.ndarray`
            The second array.
    '''
    _assure_dimension(X=X, Y=Y)
    assert X.shape[1] == Y.shape[0], 'Invalid shapes, assure that `X.shape[1] == Y.shape[0]`' 
    output = np.zeros((X.shape[0], Y.shape[1]), dtype=np.float32)
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            output[i,j] = dot(X[i, :], Y[:, j])
    return output
