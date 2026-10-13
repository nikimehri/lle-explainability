import numpy as np 

def predict (X: np.ndarray) -> np.ndarray:
    scores = X [:,0] ** 2 + X [:, 1] ** 2 - 1.5
    return (scores > 0).astype(int)

"""Note that this is a simple binary classifier that can be changed later and is only used to demonstrate the LLE algorithm right now. 
The prediction rule right now is that a score greater than 0 is classified with a label of 1. 
A score less than or equal to 0 is classified with a label of 0."""