import numpy as np
from linalg.utils.core.core import _treat_X

def dot(x:np.ndarray, y:np.ndarray)->float:
    assert len(x.shape) == len(y.shape) == 1, 'You must provide vectors as inputs'
    assert x.shape == y.shape, 'Input vectors must have the same shape'
    x, y = x.astype(np.float32), y.astype(np.float32)
    output = 0
    for i in range(x.shape[0]):
        output += (x[i] * y[i])
    return output

def matmul(X:np.ndarray, Y:np.ndarray)->np.ndarray:
    X = _treat_X(X)
    assert X.shape[1] == Y.shape[0], 'Invalid shapes, assure that `X.shape[1] == Y.shape[0]`' 
    output = np.zeros((X.shape[0], Y.shape[1]), dtype=np.float32)
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            output[i,j] = dot(X[i, :], Y[:, j])
    return output
