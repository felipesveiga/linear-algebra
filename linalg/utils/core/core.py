import numpy as np

def _treat_X(X:np.ndarray)->np.ndarray:
    return X.reshape(1,-1) if len(X.shape)==1 else X