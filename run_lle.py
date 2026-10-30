import numpy as np 


from perturbations import generate_perturbation
from neighbours import select_neighbours
from lle import compute_lle_weights, build_weight_matrix, lle_embedding
from model import get_model

"""The values for X, K, sigma, n_samples, and rng can be changed to test different scenarios."""

def main():
    rng=np.random.default_rng(42)
    rf, data, target = get_model()
    row = 86  # boundary row: forest probabilities ~0.63 / 0.37
    X = data[row]
    c = rf.predict(X.reshape(1, -1))[0]
    proba_fn = lambda x: rf.predict_proba(x)[:, c]
    #X = np.array([0.5, 0.6])
    K = 100 
    k_graph = 15  # neighbours per point when building W within P; must be < K + 1
    d = 3  # embedding dimension; must be < input dimension
    samples = generate_perturbation(
        X = X, 
        n_samples = 50000,
        sigma = 0.1, 
        rng=rng, 
    )
    neighbours, distances = select_neighbours(
        X = X, 
        samples = samples,
        predict_proba_fn = proba_fn,
        K = K,
        n_candidates = 20000
    )
    weights = compute_lle_weights(X, neighbours)
    reconstructed=weights @ neighbours
    error = np.linalg.norm(X - reconstructed)
    print(f"Original point: {X}")
    print(f"Reconstructed point: {reconstructed}")
    print(f"Reconstruction error: {error}")
    print("Number of neighbors:", len(neighbours))
    print("Original label:", rf.predict(X.reshape(1, -1))[0])
    print("Original probability:", rf.predict_proba(X.reshape(1, -1))[:, c][0])
    print("Neighbor labels:", rf.predict(neighbours))
    print("Neighbor probabilities:", rf.predict_proba(neighbours)[:, c])
    print("LLE weights:", weights)
    print("Sum of weights:", np.sum(weights))
    print("Neighbor distances:", distances)

    # Low-dimensional embedding of the local set P = [X] + neighbours (row 0 is X)
    P = np.vstack([X, neighbours])
    W = build_weight_matrix(P, k_graph)
    Y = lle_embedding(W, d)
    print("Embedding of X:", Y[0])
    print("Embedding of neighbours:", Y[1:].ravel())
    print("Embedding mean (should be ~0):", Y.mean(axis=0))
    print("Embedding covariance (should be ~I):", Y.T @ Y / len(Y))
    print("Embedding cost:", np.sum((Y - W @ Y) ** 2))


if __name__ == "__main__":
    main()
