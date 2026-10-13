import numpy as np 

from model import predict
from perturbations import generate_perturbation
from neighbours import select_neighbours
from lle import compute_lle_weights

"""The values for X, K, sigma, n_samples, and rng can be changed to test different scenarios."""

def main():
    rng=np.random.default_rng(42)
    X = np.array([0.5, 0.6])
    K = 10 
    samples = generate_perturbation(
        X = X, 
        n_samples = 1000,
        sigma = 0.2, 
        rng=rng, 
    )
    neighbours, distances = select_neighbours(
        X = X, 
        samples = samples,
        predict_fn = predict,
        K = K,
    )
    weights = compute_lle_weights(X, neighbours)
    reconstructed=weights @ neighbours
    error = np.linalg.norm(X - reconstructed)
    print(f"Original point: {X}")
    print(f"Reconstructed point: {reconstructed}")
    print(f"Reconstruction error: {error}")
    print("Number of neighbors:", len(neighbours))
    print("Original label:", predict(X.reshape(1, -1))[0])
    print("Neighbor labels:", predict(neighbours))
    print("LLE weights:", weights)
    print("Sum of weights:", np.sum(weights))
    print("Neighbor distances:", distances)

if __name__ == "__main__":
    main()
