**This is the repo for PaLLE (Paths through Locally Linear Embeddings for Feature Explanation).**

PaLLE selects perturbations with model predictions similar to a given input and organizes them using Locally Linear Embedding. It then decodes paths through this representation to examine joint feature changes, check whether predictions are preserved, and estimate each feature's contribution along the path.

**Data:**

Breast Cancer is included with scikit-learn through `sklearn.datasets.load_breast_cancer()` and is loaded automatically.

Download [Spambase](https://archive.ics.uci.edu/dataset/94/spambase) and save the archive to `data/spambase/spambase.zip`.

Download [Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) and extract `airfoil_self_noise.dat` to `data/airfoil/`.

HMDA: [download here](https://ffiec.cfpb.gov/data-browser/).

**To run:**

```sh
python3 -m pip install -r requirements.txt
python3 experiments.py classification --datasets breast_cancer
python3 experiments.py classification --datasets spambase
python3 experiments.py airfoil
```

**Hyperparameters:**

Note that the values for `n_samples`, `K`, `k_graph`, `d`, `sigma`, reconstruction regularization `reg`, decoder `alpha` and `gamma`, evaluation tolerances, seeds, validation grids and MLP decoder settings can be changed in **DEFAULTS** in **experiments.py**. Classifier architecture and learning-rate defaults are in **model_data** in **evaluate_full_feature.py**. These can be changed to test different scenarios. Use a new `--output` directory when changing settings, e.g. `--output research/generated/my_run`.
