import numpy as np

def _assure_dimension(**kwargs):
    '''
        Checks whether the provided array has more than one dimension. 
        
        Parameter
        --------
        **kwargs
    '''
    for name, array in kwargs.items():
        assert len(array.shape)>1, f'''The provided array is unidimensional. Please invoke `{name}.reshape(1,-1)` if it contains data of a single instance or `{name}.reshape(-1,1)` if it contains a single feature from multiple instances.'''