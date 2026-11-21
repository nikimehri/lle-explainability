import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import lru_cache
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
from threadpoolctl import threadpool_limits

from evaluate_full_feature import model_data
from experiment_support import fit_lime_plane
from full_feature_methods import (AnchoredDecoder, LinearDecoder, NonlinearDecoder,
                                  extend_coordinates, finite_sensitivities,
                                  independent_selected, original_core)
from lle import lle_embedding
from run_lle import LLEConfig

ROOT = Path(__file__).resolve().parent
DATASETS = ("breast_cancer", "spambase")

# Change these defaults or pass a JSON file with --config. Model settings are
# separate from decoder settings; all fitted models use training data only.
DEFAULTS = {
    "lle": {"n_samples": 5000, "K": 100, "k_graph": 15, "d": 3, "sigma": .3, "reg": .001},
    "decoder": {"alpha": .01, "gamma": None},
    "evaluation": {"drift_tolerances": [.05, .01], "min_rms": .1, "path_points": 33,
                   "reference_points": 257, "contrast_step": .02, "compensation_min": .01},
    "classification": {"model_seeds": [42, 43, 44], "perturbation_seeds": [42, 43, 44],
                       "queries_per_model": 20, "query_seed": 42, "endpoint_quantiles": [.25, .5, .75],
                       "validation_dimensions": [3, 8, 15, 25],
                       "validation_alphas": [.001, .01, .1, 1.], "validation_gammas": [.03, .1, .3, 1.]},
    "training": {"epochs": 400, "split_seed": 42},
    "mlp": {"widths": [32, 32], "alpha": .01, "seeds": [701, 702, 703],
            "max_iter": 1000, "max_fun": 20000, "tol": 1e-6},
    "airfoil": {"model_seeds": [42, 43, 44], "perturbation_seeds": [42, 43, 44],
                "test_points": 1000, "probe_step": .02, "vote_neighbors": 20,
                "uninformative_threshold": 1e-6, "covariance_neighbors": 128,
                "hidden_layers": [[32, 32], [64, 32]], "max_epochs": 2000,
                "patience": 100, "batch_size": 64, "learning_rate": .001, "weight_decay": .0001},
    "bootstrap": {"draws": 3000, "seed": 42},
}


@lru_cache(maxsize=8)
def classifier(dataset, seed, config_json):
    config = json.loads(config_json)
    return model_data(dataset, seed, Path(config["model_directory"]),
                      config["classification_data"].get(dataset), config["training"])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def execute_jobs(worker, jobs, workers):
    if workers == 1:
        for job in jobs:
            print(worker(job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for future in as_completed(futures):
                print(future.result(), flush=True)


def core_config(config, seed):
    return SimpleNamespace(**(asdict(LLEConfig()) | config["lle"] | {"seed": seed}))


def fit_decoder(config, core):
    return NonlinearDecoder(core["P"], core["Y"], **config["decoder"])


def endpoints(y, quantiles):
    order = np.argsort(np.linalg.norm(y[1:] - y[0], axis=1), kind="stable") + 1
    return order[np.floor(np.asarray(quantiles) * (len(order) - 1)).astype(int)]


def path_metrics(path_fn, response, gradient, query, slopes, config):
    cfg = config["evaluation"]
    path = path_fn(np.linspace(0, 1, cfg["path_points"]))
    prediction = response(path)
    dense = path_fn(np.linspace(0, 1, cfg["reference_points"]))
    reference_gradient = gradient(dense)
    reference = np.sum(.5 * (reference_gradient[:-1] + reference_gradient[1:]) * np.diff(dense, axis=0), axis=0)
    estimates = finite_sensitivities(response, path, cfg["contrast_step"])
    attribution = np.sum(.5 * (estimates[:-1] + estimates[1:]) * np.diff(path, axis=0), axis=0)
    lime = slopes * (path[-1] - query)
    rms = float(np.sqrt(np.mean((path[-1] - query) ** 2)))
    drift = float(np.max(abs(prediction - prediction[0])))
    return {"endpoint_rms": rms, "max_probability_drift": drift,
            "qualified": [bool(rms >= cfg["min_rms"] and drift <= e) for e in cfg["drift_tolerances"]],
            "same_class": bool(np.all((prediction >= .5) == (prediction[0] >= .5))),
            "reference_contributions": reference.tolist(),
            "feature_contributions": attribution.tolist(),
            "reference_mae": float(np.mean(abs(attribution - reference))),
            "lime_mae": float(np.mean(abs(lime - reference))),
            "completeness_error": float(abs(attribution.sum() - (prediction[-1] - prediction[0]))),
            "predictions": prediction.tolist(), "features": path.tolist()}


def score_decoder(decoder, core, target_ids, test, test_y, response, gradient, query, slopes, config):
    paths = []
    for j in target_ids:
        start, delta = core["Y"][0], core["Y"][j] - core["Y"][0]
        paths.append(path_metrics(lambda t, s=start, d=delta: decoder.predict(s + t[:, None] * d),
                                  response, gradient, query, slopes, config))
    return {"feature_mae": float(np.mean(abs(decoder.predict(test_y) - test))), "paths": paths}


class FixedMLP:
    """Query-anchored tanh MLP decoder; initializations are averaged in reporting."""
    def __init__(self, p, y, settings, seed):
        self.query = p[0]
        self.anchor = y[0]
        options = {k: settings[k] for k in ("alpha", "max_iter", "max_fun", "tol")}
        self.model = MLPRegressor(hidden_layer_sizes=tuple(settings["widths"]),
                                  activation="tanh", solver="lbfgs", random_state=seed,
                                  early_stopping=False, **options)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model.fit(y, p - self.query)
        self.warnings = [str(w.message) for w in caught]
        self.offset = self.model.predict(y[:1])[0]

    def predict(self, y):
        return self.query + self.model.predict(np.atleast_2d(y)) - self.offset


def classification_task(dataset, model_seed, row, seed, predictor, scaler, test_set, names, config):
    query = test_set[row]
    cls = int(predictor.predict_proba(query[None])[0].argmax())
    response = lambda v: predictor.predict_proba(v)[:, cls]
    gradient = lambda v: predictor.gradient(v, cls)
    cfg = core_config(config, seed)
    if cfg.d >= len(query):
        raise ValueError("Embedding dimension must be smaller than the number of input features")
    core = original_core(query, response, cfg)
    validation = independent_selected(query, response, cfg, seed + 10000)
    test = independent_selected(query, response, cfg, seed + 20000)
    ids = endpoints(core["Y"], config["classification"]["endpoint_quantiles"])
    test_y = extend_coordinates(test, core, cfg.k_graph, cfg.reg)
    _, slopes = fit_lime_plane(core["candidates"], response(core["candidates"]), query,
                                float(response(query[None])[0]), .75 * np.sqrt(len(query)))
    methods = {}
    models = {"rbf": fit_decoder(config, core),
              "linear": LinearDecoder(core["P"], core["Y"], alpha=config["decoder"]["alpha"])}
    for s in config["mlp"]["seeds"]:
        models[f"mlp_{s}"] = FixedMLP(core["P"], core["Y"], config["mlp"], s)
    for name, model in models.items():
        methods[name] = score_decoder(model, core, ids, test, test_y, response, gradient, query, slopes, config)
        if isinstance(model, FixedMLP):
            methods[name]["optimizer_warnings"] = model.warnings
    choices = []
    for d in sorted(set([cfg.d] + config["classification"]["validation_dimensions"])):
        if d >= min(len(query), len(core["P"])):
            continue
        y = core["Y"] if d == cfg.d else lle_embedding(core["W"], d)
        alternate = core | {"Y": y}
        vy = extend_coordinates(validation, alternate, cfg.k_graph, cfg.reg)
        for gamma in config["classification"]["validation_gammas"]:
            for alpha in config["classification"]["validation_alphas"]:
                model = AnchoredDecoder(y, core["P"], alpha, gamma)
                error = float(np.mean(abs(model.predict(vy) - validation)))
                choices.append((error, d, model, alternate))
    _, d, model, alternate = min(choices, key=lambda item: item[0])
    methods["dimension_selected"] = score_decoder(model, alternate, ids, test,
        extend_coordinates(test, alternate, cfg.k_graph, cfg.reg), response, gradient, query, slopes, config)
    methods["dimension_selected"].update(dimension=d, alpha=model.alpha, gamma=model.gamma)
    direct = []
    for j in ids:
        delta = core["P"][j] - query
        direct.append(path_metrics(lambda t, v=delta: query + t[:, None] * v,
                                   response, gradient, query, slopes, config))
    methods["direct_input_line"] = {"paths": direct, "feature_mae": None}
    return {"dataset": dataset, "model_seed": model_seed, "test_row": int(row),
            "perturbation_seed": seed, "explained_class": cls, "methods": methods,
            "endpoint_rows": ids.tolist(), "feature_names": names,
            "feature_mean": scaler.mean_.tolist(), "feature_scale": scaler.scale_.tolist()}


def hierarchical(records, metric, config, condition_key="test_row"):
    """Hierarchical model/query (or model/condition) bootstrap."""
    from collections import defaultdict
    blocks = defaultdict(lambda: defaultdict(list))
    for r in records:
        value = metric(r)
        if value is not None:
            blocks[r["model_seed"]][r[condition_key]].append(value)
    if not blocks:
        return None
    rows = [np.array([np.mean(v) for v in b.values()]) for b in blocks.values()]
    mean = float(np.mean([r.mean() for r in rows]))
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    n = config["bootstrap"]["draws"]
    boot = np.array([rng.choice(r, size=(n, len(rows), len(r)), replace=True).mean(axis=2) for r in rows])
    choices = rng.integers(len(rows), size=(n, len(rows)))
    samples = boot[choices, np.arange(n)[:, None], np.arange(len(rows))[None, :]].mean(axis=1)
    return {"mean": mean, "ci95": np.quantile(samples, [.025, .975]).tolist()}


def classification_summary(records, config):
    result = {}
    def average(values):
        return float(np.mean(values)) if values else None
    for dataset in sorted({r["dataset"] for r in records}):
        rr = [r for r in records if r["dataset"] == dataset]
        groups = {"rbf": ["rbf"], "linear": ["linear"],
                  "mlp_fixed": [f"mlp_{s}" for s in config["mlp"]["seeds"]],
                  "dimension_selected": ["dimension_selected"], "direct_input_line": ["direct_input_line"]}
        cell = {"tasks": len(rr), "paths": sum(len(r["methods"]["rbf"]["paths"]) for r in rr), "methods": {}}
        for name, arms in groups.items():
            metrics = {}
            for index, tolerance in enumerate(config["evaluation"]["drift_tolerances"]):
                metrics[f"coverage_{tolerance}"] = hierarchical(rr, lambda r, a=arms, i=index:
                    average([float(p["qualified"][i]) for m in a for p in r["methods"][m]["paths"]]), config)
            metrics["rms"] = hierarchical(rr, lambda r, a=arms:
                average([p["endpoint_rms"] for m in a for p in r["methods"][m]["paths"]]), config)
            metrics["mae"] = hierarchical(rr, lambda r, a=arms:
                average([r["methods"][m]["feature_mae"] for m in a if r["methods"][m]["feature_mae"] is not None]), config)
            cell["methods"][name] = metrics
        cell["cohorts"] = {}
        for cohort in ("boundary", "random"):
            subset = [r for r in rr if r["cohort"] == cohort]
            cell["cohorts"][cohort] = {m: {str(t): hierarchical(subset, lambda r, m=m, i=i:
                average([float(p["qualified"][i]) for p in r["methods"][m]["paths"]]), config)
                for i, t in enumerate(config["evaluation"]["drift_tolerances"])} for m in ("rbf", "dimension_selected")}
        def eligible(r):
            minimum = config["evaluation"]["compensation_min"]
            return [p for p in r["methods"]["rbf"]["paths"] if p["qualified"][0]
                    and sum(v for v in p["reference_contributions"] if v > 0) >= minimum
                    and -sum(v for v in p["reference_contributions"] if v < 0) >= minimum]
        cell["attribution"] = {"paths": sum(len(eligible(r)) for r in rr),
            **{key: hierarchical(rr, lambda r, key=key: average([p[key] for p in eligible(r)]), config)
               for key in ("reference_mae", "lime_mae")}}
        cell["qualified_paths_crossing_class"] = sum(p["qualified"][0] and not p["same_class"]
                                                      for r in rr for p in r["methods"]["rbf"]["paths"])
        result[dataset] = cell
    return result


def classification_worker(job):
    dataset, seed, row, cloud, cohort, path, config = job
    if not path.exists():
        import torch
        torch.set_num_threads(1)
        with threadpool_limits(1):
            model, scaler, x, _, names, _ = classifier(dataset, seed, json.dumps(config, sort_keys=True))
            r = classification_task(dataset, seed, row, cloud, model, scaler, x, names, config)
            r["cohort"] = cohort
            write_json(path, r)
    return "classification: " + path.stem


def run_classification(output, config, limit, datasets, workers):
    import torch
    torch.set_num_threads(1)
    folder = output / "classification"
    tasks = folder / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset in datasets:
        for seed in config["classification"]["model_seeds"]:
            model, scaler, x, _, names, _ = classifier(dataset, seed, json.dumps(config, sort_keys=True))
            p = model.predict_proba(x)[:, 1]
            pool = np.arange(len(x))
            n = config["classification"]["queries_per_model"]
            if n < 2 or n > len(pool):
                raise ValueError("queries_per_model must be between 2 and the held-out set size")
            boundary = pool[np.argsort(abs(p[pool] - .5), kind="stable")[:n // 2]]
            random = np.random.default_rng(config["classification"]["query_seed"]).choice(np.setdiff1d(pool, boundary), n - len(boundary), replace=False)
            for row in np.r_[boundary, random]:
                for cloud in config["classification"]["perturbation_seeds"]:
                    if limit is not None and len(jobs) >= limit:
                        break
                    name = f"{dataset}_m{seed}_r{row}_s{cloud}"
                    path = tasks / (name + ".json")
                    jobs.append((dataset, seed, row, cloud, "boundary" if row in boundary else "random", path, config))
    execute_jobs(classification_worker, jobs, workers)
    records = [json.loads(job[-2].read_text()) for job in sorted(jobs, key=lambda job: job[-2])]
    summary = classification_summary(records, config)
    write_json(folder / "summary.json", {"smoke_test": limit is not None, "datasets": summary})
    tolerances = config["evaluation"]["drift_tolerances"]
    headings = [f"d={config['lle']['d']}, {t}" for t in tolerances] + [f"selected d, {tolerances[0]}"]
    lines = ["PaLLE classification results", "", f"Smoke test: {limit is not None}", "",
             "Cohort coverage", "", "| Dataset | Cohort | " + " | ".join(headings) + " |",
             "|---|---|" + "---:|" * len(headings)]
    for dataset, cell in summary.items():
        for cohort, c in cell["cohorts"].items():
            get = lambda m, t: f"{100*c[m][str(t)]['mean']:.1f}" if c[m][str(t)] else "--"
            cells = [get("rbf", t) for t in tolerances] + [get("dimension_selected", tolerances[0])]
            lines.append(f"| {dataset} | {cohort} | " + " | ".join(cells) + " |")
    lines += ["", "Path attribution MAE (probability units)", "",
              "| Dataset | Eligible paths | Finite contrasts | LIME slope × displacement |", "|---|---:|---:|---:|"]
    for dataset, cell in summary.items():
        a = cell["attribution"]
        fmt = lambda v: f"{v['mean']:.8f}" if v else "--"
        lines.append(f"| {dataset} | {a['paths']}/{cell['paths']} | {fmt(a['reference_mae'])} | {fmt(a['lime_mae'])} |")
    lines += ["", "Decoder comparison (MLP initializations averaged)", "",
              "| Dataset | Decoder | Coverage (%) | RMS | Reconstruction MAE |", "|---|---|---:|---:|---:|"]
    tolerance = config["evaluation"]["drift_tolerances"][0]
    for dataset, cell in summary.items():
        for name in ("rbf", "mlp_fixed", "linear"):
            m = cell["methods"][name]
            lines.append(f"| {dataset} | {name} | {100*m[f'coverage_{tolerance}']['mean']:.1f} | {m['rms']['mean']:.3f} | {m['mae']['mean']:.3f} |")
    (folder / "tables.md").write_text("\n".join(lines) + "\n")


def airfoil_data(config):
    raw = np.loadtxt(config["airfoil_data"])
    if raw.shape != (1503, 6) or not np.isfinite(raw).all():
        raise ValueError("Expected the UCI Airfoil file with 1503 rows, five inputs and a response")
    x, target = raw[:, :5], raw[:, 5]
    _, groups = np.unique(x[:, 1:], axis=0, return_inverse=True)
    train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(x, groups=groups))
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(x[train_val], groups=groups[train_val]))
    train, val = train_val[a], train_val[b]
    scaler = StandardScaler().fit(x[train])
    mean, scale = float(target[train].mean()), float(target[train].std())
    return {"raw": x, "points": scaler.transform(x), "target": target, "train": train, "validation": val,
            "test": test, "groups": groups, "scaler": scaler, "target_mean": mean, "target_scale": scale}


class AirfoilModel:
    def __init__(self, values):
        self.weights = [np.array(v) for v in values["weights"]]
        self.biases = [np.array(v) for v in values["biases"]]

    def predict(self, points):
        h = np.asarray(points)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.tanh(h @ w.T + b)
        return (h @ self.weights[-1].T + self.biases[-1])[:, 0]


def airfoil_queries(data):
    ids = np.unique(data["groups"][data["test"]])
    centres = np.array([data["points"][data["test"][data["groups"][data["test"]] == g], 1:].mean(axis=0) for g in ids])
    chosen = [int(np.argmin(np.linalg.norm(centres - centres.mean(axis=0), axis=1)))]
    while len(chosen) < 6:
        distance = cdist(centres, centres[chosen]).min(axis=1)
        distance[chosen] = -1
        chosen.append(int(distance.argmax()))
    rows = []
    for i in chosen:
        group = data["test"][data["groups"][data["test"]] == ids[i]]
        order = group[np.argsort(data["raw"][group, 0], kind="stable")]
        rows.extend(order[np.floor(np.array([.25, .75]) * (len(order) - 1)).astype(int)].tolist())
    return rows


def airfoil_cloud(data, query, n, seed, config):
    training = data["points"][data["train"]]
    ids = np.argsort(np.linalg.norm(training - query, axis=1), kind="stable")[:config["airfoil"]["covariance_neighbors"]]
    covariance = np.cov(training[ids], rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 1e-6 * values.mean())
    values *= query.size * config["lle"]["sigma"] ** 2 / values.sum()
    root = vectors @ np.diag(np.sqrt(values))
    lower, upper = np.minimum(training.min(axis=0), query), np.maximum(training.max(axis=0), query)
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(200):
        p = query + rng.normal(size=(max(n, 1024), query.size)) @ root.T
        batches.append(p[np.all((p >= lower) & (p <= upper), axis=1)])
        if sum(map(len, batches)) >= n:
            return np.vstack(batches)[:n]
    raise RuntimeError("Insufficient supported Airfoil perturbations")


def forward_effects(response, x, step):
    base = response(x)
    result = np.empty_like(x)
    for start in range(0, len(x), 256):
        batch = x[start:start + 256]
        plus = (batch[:, None, :] + step * np.eye(x.shape[1])).reshape(-1, x.shape[1])
        result[start:start + len(batch)] = response(plus).reshape(len(batch), -1) - base[start:start + len(batch), None]
    return result


def nearest_vote(training, test, effects, config):
    a = config["airfoil"]
    ids = np.flatnonzero(abs(effects).sum(axis=1) >= a["uninformative_threshold"])
    k = a["vote_neighbors"]
    if len(ids) < k:
        raise ValueError("Insufficient informative labels")
    distance = cdist(test, training[ids])
    neighbours = np.argpartition(distance, k - 1, axis=1)[:, :k]
    weights = 1 / np.maximum(np.take_along_axis(distance, neighbours, axis=1), 1e-9)
    weights /= weights.sum(axis=1, keepdims=True)
    labels = abs(effects[ids]).argmax(axis=1)
    votes = np.zeros((len(test), effects.shape[1]))
    np.add.at(votes, (np.repeat(np.arange(len(test)), k), labels[neighbours].ravel()), weights.ravel())
    return votes.argmax(axis=1)


def airfoil_task(data, model, model_seed, row, seed, config):
    query = data["points"][row]
    cfg = core_config(config, seed)
    if cfg.d >= query.size:
        raise ValueError("Embedding dimension must be smaller than the number of input features")
    training = airfoil_cloud(data, query, cfg.n_samples, seed, config)
    test = airfoil_cloud(data, query, config["airfoil"]["test_points"], 100000 + row, config)
    base = model.predict(training)
    selected = np.argsort(abs(base - model.predict(query[None])[0]), kind="stable")[:cfg.K]
    from lle import build_weight_matrix
    p = np.vstack([query, training[selected]])
    W = build_weight_matrix(p, cfg.k_graph, cfg.reg)
    core = {"P": p, "Y": lle_embedding(W, cfg.d)}
    train_y = extend_coordinates(training, core, cfg.k_graph, cfg.reg)
    train_y[selected] = core["Y"][1:]
    test_y = extend_coordinates(test, core, cfg.k_graph, cfg.reg)
    effects = forward_effects(model.predict, training, config["airfoil"]["probe_step"])
    reference = forward_effects(model.predict, test, config["airfoil"]["probe_step"])
    truth = abs(reference).argmax(axis=1)
    if not np.all(abs(reference).sum(axis=1) >= config["airfoil"]["uninformative_threshold"]):
        raise ValueError("Uninformative evaluation effects; report rather than silently filtering")
    pca = PCA(n_components=cfg.d, svd_solver="full").fit(p)
    predicted = {"palle": nearest_vote(train_y, test_y, effects, config),
                 "pca": nearest_vote(pca.transform(training), pca.transform(test), effects, config),
                 "input_knn": nearest_vote(training, test, effects, config)}
    explainer = LimeTabularExplainer(data["points"][data["train"]], mode="regression",
        discretize_continuous=False, sample_around_instance=True, feature_selection="none", random_state=42)
    lime = []
    for i, point in enumerate(test):
        s = int(np.random.SeedSequence([20261115, model_seed, row, seed, i]).generate_state(1)[0])
        explainer.random_state.seed(s)
        explanation = explainer.explain_instance(point, model.predict, num_features=query.size, num_samples=cfg.n_samples)
        coefficients = np.zeros(query.size)
        for j, value in explanation.local_exp[1]:
            coefficients[j] = value / explainer.scaler.scale_[j]
        lime.append(int(abs(coefficients).argmax()))
    predicted["lime"] = np.array(lime)
    return {"model_seed": model_seed, "test_row": row, "condition_id": airfoil_queries(data).index(row) // 2,
            "perturbation_seed": seed, "truth": truth.tolist(),
            "predictions": {m: v.tolist() for m, v in predicted.items()},
            "accuracy": {m: float(np.mean(v == truth)) for m, v in predicted.items()}}


def airfoil_worker(job):
    model_seed, row, seed, path, config = job
    if not path.exists():
        with threadpool_limits(1):
            data = airfoil_data(config)
            values = json.loads((Path(config["model_directory"]) / "airfoil.json").read_text())["seeds"][str(model_seed)]
            write_json(path, airfoil_task(data, AirfoilModel(values), model_seed, row, seed, config))
    return "airfoil: " + path.stem


def airfoil_stat(records, metric, config):
    """Hierarchical bootstrap over models and operating conditions."""
    models = sorted({r["model_seed"] for r in records})
    conditions = sorted({r["condition_id"] for r in records})
    cells = [[np.mean([metric(r) for r in records if r["model_seed"] == m and r["condition_id"] == c])
              if any(r["model_seed"] == m and r["condition_id"] == c for r in records) else np.nan
              for c in conditions] for m in models]
    if not cells or not np.isfinite(cells).all():
        return hierarchical(records, metric, config, "condition_id")
    values = np.asarray(cells)
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    draws = config["bootstrap"]["draws"]
    mi = rng.integers(0, len(models), (draws, len(models)))
    ci = rng.integers(0, len(conditions), (draws, len(models), len(conditions)))
    samples = values[mi[:, :, None], ci].mean(axis=(1, 2))
    return {"mean": float(values.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist()}


def run_airfoil(output, config, limit, workers):
    folder = output / "airfoil"
    tasks = folder / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    data = airfoil_data(config)
    train_airfoil_models(data, config)
    jobs = []
    for model_seed in config["airfoil"]["model_seeds"]:
        for row in airfoil_queries(data):
            for seed in config["airfoil"]["perturbation_seeds"]:
                if limit is not None and len(jobs) >= limit:
                    break
                name = f"m{model_seed}_r{row}_s{seed}"
                path = tasks / (name + ".json")
                jobs.append((model_seed, row, seed, path, config))
    execute_jobs(airfoil_worker, jobs, workers)
    records = [json.loads(job[-2].read_text()) for job in sorted(jobs, key=lambda job: job[-2])]
    summary = {"smoke_test": limit is not None, "tasks": len(records), "methods": {}}
    import itertools
    for method in ("palle", "pca", "lime", "input_knn"):
        accuracy = airfoil_stat(records, lambda r, m=method: r["accuracy"][m], config)
        agreement = []
        for model_seed in config["airfoil"]["model_seeds"]:
            for row in airfoil_queries(data):
                rr = [r for r in records if r["model_seed"] == model_seed and r["test_row"] == row]
                if len(rr) != len(config["airfoil"]["perturbation_seeds"]) or len(rr) < 2:
                    continue
                agreement.append({"model_seed": model_seed, "condition_id": rr[0]["condition_id"],
                    "agreement": float(np.mean([np.mean(np.array(a["predictions"][method]) == b["predictions"][method])
                                                 for a, b in itertools.combinations(rr, 2)]))})
        stability = airfoil_stat(agreement, lambda r: r["agreement"], config) if agreement else None
        summary["methods"][method] = {"accuracy": accuracy, "refit_agreement": stability}
    write_json(folder / "summary.json", summary)
    lines = ["Airfoil dominant forward-response feature", "", f"Smoke test: {limit is not None}", "",
             "| Method | Accuracy (%) [95% CI] | Refit agreement (%) [95% CI] |", "|---|---:|---:|"]
    for method, values in summary["methods"].items():
        def fmt(v):
            return "--" if v is None else f"{100*v['mean']:.1f} [{100*v['ci95'][0]:.1f}, {100*v['ci95'][1]:.1f}]"
        lines.append(f"| {method} | {fmt(values['accuracy'])} | {fmt(values['refit_agreement'])} |")
    (folder / "results.md").write_text("\n".join(lines) + "\n")


def train_airfoil_models(data, config):
    """Select architecture on validation operating conditions, then refit per seed."""
    path = Path(config["model_directory"]) / "airfoil.json"
    if path.exists():
        return
    options = config["airfoil"]
    x = torch.tensor(data["points"][data["train"]], dtype=torch.float32)
    target = (data["target"] - data["target_mean"]) / data["target_scale"]
    y = torch.tensor(target[data["train"]], dtype=torch.float32)[:, None]
    vx = torch.tensor(data["points"][data["validation"]], dtype=torch.float32)
    vy = torch.tensor(target[data["validation"]], dtype=torch.float32)[:, None]

    def fit(widths, seed):
        torch.manual_seed(seed)
        layers = []
        sizes = [x.shape[1], *widths, 1]
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(torch.nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(torch.nn.Tanh())
        network = torch.nn.Sequential(*layers)
        optimizer = torch.optim.Adam(network.parameters(), lr=options["learning_rate"],
                                     weight_decay=options["weight_decay"])
        generator = torch.Generator().manual_seed(seed)
        best, stale, best_state, best_epoch = float("inf"), 0, None, 0
        for epoch in range(options["max_epochs"]):
            for ids in torch.randperm(len(x), generator=generator).split(options["batch_size"]):
                optimizer.zero_grad()
                loss = ((network(x[ids]) - y[ids]) ** 2).mean()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                score = float(((network(vx) - vy) ** 2).mean())
            if score < best - 1e-5:
                best, stale, best_state, best_epoch = score, 0, copy.deepcopy(network.state_dict()), epoch + 1
            else:
                stale += 1
            if stale >= options["patience"]:
                break
        network.load_state_dict(best_state)
        linear = [layer for layer in network if isinstance(layer, torch.nn.Linear)]
        return {"weights": [layer.weight.detach().numpy().tolist() for layer in linear],
                "biases": [layer.bias.detach().numpy().tolist() for layer in linear],
                "validation_mse": best, "best_epoch": best_epoch, "hidden_layers": widths}

    first_seed = options["model_seeds"][0]
    trials = [fit(widths, first_seed) for widths in options["hidden_layers"]]
    selected = min(trials, key=lambda m: m["validation_mse"])
    models = {str(first_seed): selected}
    for seed in options["model_seeds"][1:]:
        models[str(seed)] = fit(selected["hidden_layers"], seed)
    for model in models.values():
        prediction = AirfoilModel(model).predict(data["points"][data["test"]])
        model["test_rmse"] = float(np.sqrt(np.mean((prediction - target[data["test"]]) ** 2)) * data["target_scale"])
    write_json(path, {"seeds": models, "architecture_validation": trials,
                      "train_indices": data["train"].tolist(), "validation_indices": data["validation"].tolist(),
                      "test_indices": data["test"].tolist(), "feature_mean": data["scaler"].mean_.tolist(),
                      "feature_scale": data["scaler"].scale_.tolist(),
                      "target_mean": data["target_mean"], "target_scale": data["target_scale"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("classification", "airfoil", "csv", "all"))
    parser.add_argument("--data", type=Path, help="CSV (header, binary target last), Airfoil .dat, or Spambase archive")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output", type=Path, default=ROOT / "research/generated/experiments")
    parser.add_argument("--config", type=Path, help="Optional JSON overriding sections of DEFAULTS")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Maximum tasks per experiment for a smoke test")
    parser.add_argument("--model-seeds", nargs="+", type=int)
    parser.add_argument("--perturbation-seeds", nargs="+", type=int)
    parser.add_argument("--rows", type=int, help="Held-out classification queries per model")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--neighbors", type=int)
    parser.add_argument("--graph-neighbors", type=int)
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--reg", type=float)
    parser.add_argument("--epochs", type=int, help="Classification training epochs")
    parser.add_argument("--learning-rate", type=float, help="Classification learning rate")
    parser.add_argument("--hidden-layers", nargs="+", type=int, help="Classification hidden widths")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if args.experiment == "csv" and args.data is None:
        parser.error("csv requires --data with a numeric CSV and binary target in the last column")
    if args.data is not None and args.experiment in ("all", "classification") and args.datasets != ["spambase"]:
        parser.error("--data requires csv, airfoil, or classification --datasets spambase")
    config = copy.deepcopy(DEFAULTS)
    if args.config:
        for key, value in json.loads(args.config.read_text()).items():
            if key not in config or not isinstance(value, dict):
                parser.error(f"Unknown configuration section: {key}")
            config[key].update(value)
    for key, name in (("n_samples", "n_samples"), ("K", "neighbors"), ("k_graph", "graph_neighbors"),
                      ("d", "dimension"), ("sigma", "sigma"), ("reg", "reg")):
        if getattr(args, name) is not None:
            config["lle"][key] = getattr(args, name)
    for key in ("model_seeds", "perturbation_seeds"):
        if getattr(args, key) is not None:
            config["classification"][key] = config["airfoil"][key] = getattr(args, key)
    if args.rows is not None:
        config["classification"]["queries_per_model"] = args.rows
    for key in ("epochs", "learning_rate", "hidden_layers"):
        if getattr(args, key) is not None:
            config["training"][key] = getattr(args, key)
    c = config["lle"]
    if not (0 < c["d"] < c["K"] <= c["n_samples"] and 0 < c["k_graph"] <= c["K"]
            and c["sigma"] > 0 and c["reg"] > 0 and config["training"]["epochs"] > 0):
        parser.error("Require positive scale, regularization and epochs, d < K <= n_samples, and graph neighbours <= K")
    config["model_directory"] = str(args.output.resolve() / "models")
    config["classification_data"] = {"spambase": str(ROOT / "data/spambase/spambase.zip")}
    config["airfoil_data"] = str(ROOT / "data/airfoil/airfoil_self_noise.dat")
    if args.data:
        if args.experiment == "airfoil":
            config["airfoil_data"] = str(args.data.resolve())
        else:
            config["classification_data"]["csv" if args.experiment == "csv" else "spambase"] = str(args.data.resolve())
    files = []
    datasets = ["csv"] if args.experiment == "csv" else args.datasets
    if args.experiment in ("classification", "csv", "all"):
        for dataset in datasets:
            if dataset != "breast_cancer":
                files.append(config["classification_data"][dataset])
    if args.experiment in ("airfoil", "all"):
        files.append(config["airfoil_data"])
    for filename in files:
        if not Path(filename).is_file():
            parser.error(f"Dataset missing: {filename}. See README.md for download instructions.")
    hashes = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    # The requested subset may change on a resumed run; data fingerprints accumulate.
    config_path = args.output / "configuration.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        previous_hashes = previous.pop("data_sha256", {})
        if previous != config or any(k in previous_hashes and previous_hashes[k] != v for k, v in hashes.items()):
            parser.error("Configuration or data changed; select a fresh --output directory")
        hashes = previous_hashes | hashes
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config | {"data_sha256": hashes})
    write_json(args.output / "versions.json", {"python": sys.version,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "scikit-learn", "torch", "lime", "threadpoolctl")},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("experiments.py", "evaluate_full_feature.py", "full_feature_methods.py",
                                       "experiment_support.py", "lle.py", "neighbours.py", "perturbations.py", "run_lle.py")}})
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(1):
        if args.experiment in ("classification", "csv", "all"):
            run_classification(args.output, config, args.limit, datasets, args.workers)
        if args.experiment in ("airfoil", "all"):
            run_airfoil(args.output, config, args.limit, args.workers)


if __name__ == "__main__":
    main()
