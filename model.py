import numpy as np
import sklearn
import sklearn.datasets
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#Use predict and predict_probability functions to get the predictions and probabilities of toy models. 
def predict_probability (X: np.ndarray) -> np.ndarray:
    scores = X [:,0] ** 2 + X [:, 1] ** 2 - 1.5
    return 1 / (1 + np.exp(-scores))

def predict (X: np.ndarray) -> np.ndarray:
    return (predict_probability(X) > 0.5).astype(int)

"""Note that this is a simple binary classifier that can be changed later and is only used to demonstrate the LLE algorithm right now.
predict_proba returns the probability of label 1 (a sigmoid of the score).
The prediction rule right now is that a probability greater than 0.5 (score greater than 0) is classified with a label of 1.
A probability less than or equal to 0.5 (score less than or equal to 0) is classified with a label of 0."""

def get_model(seed: int = 0):
    dataset = load_breast_cancer()
    data = StandardScaler().fit_transform(dataset.data)  # z-units, so sigma and Euclidean distance treat all features alike
    rf = RandomForestClassifier(random_state=seed)
    rf.fit(data, dataset.target)
    return rf, data, dataset.target
