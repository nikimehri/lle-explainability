import numpy as np 

def generate_perturbation(
        X : np.ndarray,
        n_samples : int, 
        sigma : float, 
        rng: np.random.Generator, 
    ) -> np.ndarray:
    noise = rng.normal(
        loc = 0,
        scale = sigma,
        size = (n_samples, len(X))
    )
    return X + noise 
