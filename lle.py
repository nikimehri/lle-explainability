import numpy as np 

def compute_lle_weights(
        X : np.ndarray, 
        neighbours : np.ndarray, 
        reg: float = 1e-3,
    ) -> np.ndarray:
    Z = neighbours - X
    C = Z @ Z.T
    """Some papers also regularize the local Gram matrix here:
    trace = np.trace(C)
    if trace > 0:
    regularization = reg * trace
    else:
    regularization * np.eye(len(neighbours))"""
    C += np.eye(len(neighbours)) * (
        reg * max(np.trace(C), 1e-12)
    )
    ones=np.ones(len(neighbours))
    weights = np.linalg.solve(C, ones)
    weights /= np.sum(weights)
    return weights

"""A hyperparameter search can be done later on for reg to find an optimal value if needed."""