import numpy as np 

def select_neighbours(
        X : np.ndarray, 
        samples : np.ndarray, 
        predict_fn, 
        K : int, 
    ) -> tuple[np.ndarray, np.ndarray]:
    label = predict_fn(X.reshape (1,-1)) [0]
    samples_labels = predict_fn(samples)
    candidates = samples[samples_labels == label]

    if len(candidates) < K: 
        raise ValueError(f"Not enough candidates for label {label} to select {K} neighbours.") 
        """Previously discussed not having an error, and instead, using a confidence score. Look into this for future improvements."""
    distances = np.linalg.norm(candidates - X, axis=1)
    indices = np.argsort(distances)[:K]
    return candidates[indices], distances[indices]
