import numpy as np

def select_neighbours(
        X : np.ndarray,
        samples : np.ndarray,
        predict_proba_fn,
        K : int,
        n_candidates : int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
    if n_candidates < K:
        raise ValueError("n_candidates must be >= K.")
    if len(samples) < n_candidates:
        raise ValueError(f"Only {len(samples)} samples, need {n_candidates} candidates.")

    # Stage 1: keep the samples whose probability is closest to the inputs
    p_x = predict_proba_fn(X.reshape(1, -1))[0]
    p_samples = predict_proba_fn(samples)
    prob_gap = np.abs(p_samples - p_x)
    candidates = samples[np.argsort(prob_gap)[:n_candidates]]

    # Stage 2: among these samples, take the K nearest by Euclidean distance
    distances = np.linalg.norm(candidates - X, axis=1)
    indices = np.argsort(distances)[:K]
    return candidates[indices], distances[indices]
