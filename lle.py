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

def build_weight_matrix(
        P : np.ndarray,
        k_graph : int,
        reg : float = 1e-3,
    ) -> np.ndarray:
    """Row i holds the LLE weights reconstructing P[i] from its k_graph nearest other points in P (zeros elsewhere)."""
    N = len(P)
    if k_graph >= N:
        raise ValueError("k_graph must be smaller than the number of points in P.")
    W = np.zeros((N, N))
    for i in range(N):
        distances = np.linalg.norm(P - P[i], axis=1)
        distances[i] = np.inf  # a point is not its own neighbour
        idx = np.argsort(distances)[:k_graph]
        W[i, idx] = compute_lle_weights(P[i], P[idx], reg)
    return W


def lle_embedding(
        W : np.ndarray,
        d : int,
    ) -> np.ndarray:
    """Low-dimensional coordinates Y (N x d) minimising sum_i ||y_i - sum_j W_ij y_j||^2
    subject to sum_i y_i = 0 and (1/N) Y^T Y = I."""
    N = W.shape[0]
    IW = np.eye(N) - W
    eigvals, eigvecs = np.linalg.eigh(IW.T @ IW)  # ascending eigenvalues
    # Skip the first (constant, eigenvalue ~ 0) eigenvector; the rest are orthogonal to it, so Y sums to 0.
    # Eigenvectors have unit norm, so scale by sqrt(N) to get (1/N) Y^T Y = I.
    return eigvecs[:, 1:d + 1] * np.sqrt(N)
