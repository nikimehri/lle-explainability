import numpy as np
from dataclasses import dataclass

from perturbations import generate_perturbation
from neighbours import select_neighbours
from lle import compute_lle_weights, build_weight_matrix, lle_embedding
from model import get_model

"""The values in LLEConfig can be changed to test different scenarios. report_lle.py uses this same config and pipeline."""

@dataclass(frozen = True)
class LLEConfig:
    row : int = 86              # boundary row
    K : int = 100               # neighbours kept around X
    k_graph : int = 15          # neighbours per point when building W within P; must be < K + 1
    d : int = 3                 # embedding dimension; must be < input dimension
    n_samples : int = 5000     # perturbations drawn around X
    sigma : float = 0.3        # perturbation scale 
    seed : int = 42


def run_lle(config : LLEConfig = LLEConfig()) -> dict:
    """Perturb X, select neighbours, compute X's LLE reconstruction weights, then build W over P = [X] + neighbours and embed it."""
    rng = np.random.default_rng(config.seed)
    rf, data, target = get_model()
    X = data[config.row]
    c = rf.predict(X.reshape(1, -1))[0]
    proba_fn = lambda x: rf.predict_proba(x)[:, c]
    samples = generate_perturbation(
        X = X,
        n_samples = config.n_samples,
        sigma = config.sigma,
        rng = rng,
    )
    neighbours = select_neighbours(
        X = X,
        samples = samples,
        predict_proba_fn = proba_fn,
        K = config.K,
    )
    distances = np.linalg.norm(neighbours - X, axis=1)  # reporting only
    weights = compute_lle_weights(X, neighbours)
    reconstructed = weights @ neighbours

    # Low-dimensional embedding of the local set P = [X] + neighbours (row 0 is X)
    P = np.vstack([X, neighbours])
    W = build_weight_matrix(P, config.k_graph)
    Y = lle_embedding(W, config.d)
    return {
        "config" : config, "rf" : rf, "X" : X, "label" : c, "true_label" : int(target[config.row]),
        "prob_X" : float(rf.predict_proba(X.reshape(1, -1))[0, c]),
        "neighbours" : neighbours, "distances" : distances,
        "neighbour_label" : rf.predict(neighbours), "neighbour_prob" : rf.predict_proba(neighbours)[:, c],
        "weights" : weights, "reconstructed" : reconstructed,
        "error" : float(np.linalg.norm(X - reconstructed)),
        "P" : P, "W" : W, "Y" : Y,
        "embedding_cost" : float(np.sum((Y - W @ Y) ** 2)),
    }


def main():
    r = run_lle()
    print(f"Original point: {r['X']}")
    print(f"Reconstructed point: {r['reconstructed']}")
    print(f"Reconstruction error: {r['error']}")
    print("Number of neighbors:", len(r["neighbours"]))
    print("Original label:", r["label"])
    print("Original probability:", r["prob_X"])
    print("Neighbor labels:", r["neighbour_label"])
    print("Neighbor probabilities:", r["neighbour_prob"])
    print("LLE weights:", r["weights"])
    print("Sum of weights:", np.sum(r["weights"]))
    print("Neighbor distances:", r["distances"])
    Y = r["Y"]
    print("Embedding of X:", Y[0])
    print("Embedding of neighbours:", Y[1:].ravel())
    print("Embedding mean (should be ~0):", Y.mean(axis=0))
    print("Embedding covariance (should be ~I):", Y.T @ Y / len(Y))
    print("Embedding cost:", r["embedding_cost"])


if __name__ == "__main__":
    main()
