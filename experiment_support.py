"""Fitting LIME for comparison experiments with PaLLE."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from lime.lime_base import LimeBase
import numpy as np
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "research/results.json"


@lru_cache(maxsize=1)
def load_results():
    return json.loads(RESULTS.read_text())


def check_original_core():
    for filename, expected in load_results()["core_sha256"].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Original LLE core differs from the reported study: {filename}")


def fit_lime_plane(points, responses, query, query_probability, width):
    design = np.vstack([np.zeros(len(query)), points - query])
    probabilities = np.r_[query_probability, responses]
    fitter = LimeBase(lambda distances: np.exp(-distances ** 2 / (2 * width ** 2)))
    intercept, explanation, _, _ = fitter.explain_instance_with_data(
        design, np.column_stack([1 - probabilities, probabilities]),
        np.linalg.norm(design, axis=1), 1, len(query),
        feature_selection="none", model_regressor=Ridge(alpha=.001),
    )
    coefficients = np.zeros(len(query))
    for index, value in explanation:
        coefficients[index] = value
    return float(intercept), coefficients


def clustered_statistic(records, value_fn):
    blocks = {}
    for record in records:
        value = value_fn(record)
        if value is not None:
            blocks.setdefault(record["model_seed"], {}).setdefault(record["test_row"], []).append(value)
    if not blocks:
        return None
    rng = np.random.default_rng(42)
    row_arrays = [np.array([np.mean(values) for values in people.values()]) for people in blocks.values()]
    mean = float(np.mean([values.mean() for values in row_arrays]))
    row_bootstraps = np.array([rng.choice(values, size=(3000, len(row_arrays), len(values)), replace=True).mean(axis=2) for values in row_arrays])
    seed_choices = rng.integers(len(row_arrays), size=(3000, len(row_arrays)))
    bootstrap = np.mean(row_bootstraps[seed_choices, np.arange(3000)[:, None], np.arange(len(row_arrays))[None, :]], axis=1)
    return {"mean": mean, "hierarchical_ci95": np.quantile(bootstrap, [.025, .975]).tolist(), "model_seeds": len(blocks), "model_query_units": sum(len(people) for people in blocks.values())}


def informative(path):
    contributions = np.array(path["reference_contributions"])
    return bool(path["success_005"] and contributions[contributions > 0].sum() >= .01 and -contributions[contributions < 0].sum() >= .01)
