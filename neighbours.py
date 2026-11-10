import numpy as np

def select_neighbours(
        X : np.ndarray,
        samples : np.ndarray,
        predict_proba_fn,
        K : int,
    ) -> np.ndarray:
    """The K samples whose probability is closest to X's. No distance in input space is used."""
    if len(samples) < K:
        raise ValueError(f"Only {len(samples)} samples, need K={K}.")

    p_x = predict_proba_fn(X.reshape(1, -1))[0]
    prob_gap = np.abs(predict_proba_fn(samples) - p_x)
    return samples[np.argsort(prob_gap, kind="stable")[:K]]
