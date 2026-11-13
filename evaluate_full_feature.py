"""Held-out evaluation of the original full-feature LLE plus explanation layers."""

import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiment_support import fit_lime_plane, ROOT
from full_feature_methods import AnchoredDecoder, NonlinearDecoder, finite_sensitivities, independent_selected, original_core, extend_coordinates, select_decoder
from lle import lle_embedding
from run_lle import LLEConfig


class Network:
    def __init__(self, network):
        layers = [layer for layer in network if isinstance(layer, torch.nn.Linear)]
        self.weights = [layer.weight.detach().numpy().astype(float) for layer in layers]
        self.biases = [layer.bias.detach().numpy().astype(float) for layer in layers]
        self.probability_evaluations = 0
        self.gradient_audit_points = 0

    def forward(self, points):
        hidden = points
        masks = []
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            hidden = hidden @ weight.T + bias
            masks.append(hidden > 0)
            hidden = np.maximum(hidden, 0)
        logits = hidden @ self.weights[-1].T + self.biases[-1]
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        return probabilities / probabilities.sum(axis=1, keepdims=True), masks

    def predict_proba(self, points):
        self.probability_evaluations += len(points)
        return self.forward(points)[0]

    def gradient(self, points, class_index):
        self.gradient_audit_points += len(points)
        probabilities, masks = self.forward(points)
        gradients = np.repeat((self.weights[-1][1] - self.weights[-1][0])[None], len(points), axis=0)
        for weight, mask in zip(self.weights[-2::-1], masks[::-1]):
            gradients = (gradients * mask) @ weight
        gradients *= (probabilities[:, 0] * probabilities[:, 1])[:, None]
        return gradients if class_index == 1 else -gradients


def model_data(dataset, seed, directory=None):
    metadata = json.loads((ROOT / "research/models.json").read_text())[dataset][str(seed)]
    weights = torch.load(ROOT / "research/models.pt", weights_only=True, map_location="cpu")[dataset][str(seed)]
    if dataset == "breast_cancer":
        data = load_breast_cancer()
        points, labels, names = data.data, data.target, data.feature_names.tolist()
        split = metadata
    elif dataset == "spambase":
        archive_path = ROOT / "data/spambase/spambase.zip"
        if not archive_path.exists():
            raise FileNotFoundError("Cache the UCI Spambase archive at data/spambase/spambase.zip before rerunning real-data evaluation; saved summaries and figures do not require this download")
        with zipfile.ZipFile(archive_path) as archive:
            values = np.loadtxt(io.BytesIO(archive.read("spambase.data")), delimiter=",")
            names = [line.split(":")[0].strip() for line in archive.read("spambase.names").decode().splitlines() if line.startswith(("word_freq_", "char_freq_", "capital_run_length_"))]
        points, labels = values[:, :-1], values[:, -1].astype(np.int64)
        split = metadata["split"]
        if points.shape != (4601, 57) or len(names) != 57:
            raise ValueError("Spambase archive does not match the recorded study")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    scaler = StandardScaler().fit(points[split["train_indices"]])
    test_indices = split["test_indices"]
    widths = [points.shape[1], *metadata["hidden_layers"], 2]
    layers = []
    for position, (inputs, outputs) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(torch.nn.Linear(inputs, outputs))
        if position < len(widths) - 2:
            layers.append(torch.nn.ReLU())
    network = torch.nn.Sequential(*layers)
    network.load_state_dict(weights)
    return Network(network), scaler, scaler.transform(points[test_indices]), labels[test_indices], names, metadata


def endpoint_indices(embedding):
    order = np.argsort(np.linalg.norm(embedding[1:] - embedding[0], axis=1), kind="stable") + 1
    positions = np.floor(np.array([.25, .5, .75]) * (len(order) - 1)).astype(int)
    return order[positions]


def paths_for_decoder(decoder, embedding, indices):
    result = []
    for index in indices:
        start, finish = embedding[0], embedding[index]
        result.append((int(index), lambda fractions, start=start, finish=finish: decoder.predict(start + fractions[:, None] * (finish - start))))
    return result


def path_metrics(path_fn, probability, predictor, class_index, query, sigma, scaler, dataset, lime_slopes):
    fractions = np.linspace(0, 1, 33)
    path = path_fn(fractions)
    probabilities = probability(path)
    extent = float(np.sqrt(np.mean((path[-1] - query) ** 2)))
    drift = float(np.max(abs(probabilities - probabilities[0])))
    reference_path = path_fn(np.linspace(0, 1, 257))
    exact = predictor.gradient(reference_path, class_index)
    reference_contributions = np.sum(.5 * (exact[:-1] + exact[1:]) * np.diff(reference_path, axis=0), axis=0)
    attributions = {}
    for step in (.02, .05):
        gradients = finite_sensitivities(probability, path, step)
        contributions = np.sum(.5 * (gradients[:-1] + gradients[1:]) * np.diff(path, axis=0), axis=0)
        residual = float(contributions.sum() - (probabilities[-1] - probabilities[0]))
        norm = np.linalg.norm(contributions) * np.linalg.norm(reference_contributions)
        attributions[str(step)] = {
            "feature_contributions": contributions.tolist(), "completeness_error": abs(residual),
            "reference_mae": float(np.mean(abs(contributions - reference_contributions))),
            "reference_cosine": float(contributions @ reference_contributions / norm) if norm > 1e-10 else None,
            "total_absolute_contribution": float(np.sum(abs(contributions))),
        }
    lime_contributions = lime_slopes * (path[-1] - query)
    raw = scaler.inverse_transform(path)
    invalid = raw < -1e-10
    if dataset == "spambase":
        invalid[:, :54] |= raw[:, :54] > 100 + 1e-10
    return {
        "endpoint_rms": extent, "max_probability_drift": drift,
        "nontrivial": extent >= sigma / 3,
        "success_005": bool(extent >= sigma / 3 and drift <= .05),
        "success_001": bool(extent >= sigma / 3 and drift <= .01),
        "raw_domain_valid_fraction": float(np.mean(~np.any(invalid, axis=1))),
        "attributions": attributions,
        "reference_total_absolute_contribution": float(np.sum(abs(reference_contributions))),
        "reference_completeness_error": float(abs(reference_contributions.sum() - (probabilities[-1] - probabilities[0]))),
        "lime_path_contribution_mae": float(np.mean(abs(lime_contributions - reference_contributions))),
        "probabilities": probabilities.tolist(), "decoded_features": path.tolist(),
        "reference_contributions": reference_contributions.tolist(),
    }


def evaluate(predictor, scaler, query, dataset, seed):
    query_count_before = predictor.probability_evaluations
    audit_count_before = predictor.gradient_audit_points
    config = replace(LLEConfig(), seed=seed)
    class_index = int(np.argmax(predictor.predict_proba(query[None])[0]))
    probability = lambda points: predictor.predict_proba(points)[:, class_index]
    core = original_core(query, probability, config)
    validation = independent_selected(query, probability, config, seed + 10000)
    test = independent_selected(query, probability, config, seed + 20000)
    constant_error = float(np.mean(abs(test - query)))
    _, lime_slopes = fit_lime_plane(core["candidates"], probability(core["candidates"]), query, float(probability(query[None])[0]), .75 * np.sqrt(len(query)))
    test_coordinates = extend_coordinates(test, core, config.k_graph)
    validation_coordinates = extend_coordinates(validation, core, config.k_graph)
    legacy = NonlinearDecoder(core["P"], core["Y"])
    _, constrained = select_decoder(core["Y"], core["P"], validation_coordinates, validation)
    interpolating = AnchoredDecoder(core["Y"], core["P"], .0001, .3)
    methods = {
        "original_d3_legacy": (legacy, core["Y"], test_coordinates, 3),
        "original_d3_constrained": (constrained, core["Y"], test_coordinates, 3),
        "original_d3_interpolating": (interpolating, core["Y"], test_coordinates, 3),
    }
    choices = []
    for dimensions in (3, 8, 15, 25):
        if dimensions >= len(query):
            continue
        embedding = core["Y"] if dimensions == 3 else lle_embedding(core["W"], dimensions)
        alternative = {**core, "Y": embedding}
        validation_embedding = extend_coordinates(validation, alternative, config.k_graph)
        score, decoder = select_decoder(embedding, core["P"], validation_embedding, validation)
        choices.append((score, dimensions, decoder, embedding, alternative))
    _, dimensions, decoder, embedding, alternative = min(choices, key=lambda item: item[0])
    methods["original_dimension_tuned"] = (decoder, embedding, extend_coordinates(test, alternative, config.k_graph), dimensions)
    pca = PCA(n_components=3).fit(core["P"])
    pca_scaler = StandardScaler().fit(pca.transform(core["P"]))
    pca_embedding = pca_scaler.transform(pca.transform(core["P"]))
    _, pca_decoder = select_decoder(pca_embedding, core["P"], pca_scaler.transform(pca.transform(validation)), validation)
    methods["pca_d3_control"] = (pca_decoder, pca_embedding, pca_scaler.transform(pca.transform(test)), 3)
    methods["pca_d3_legacy_decoder"] = (NonlinearDecoder(core["P"], pca_embedding), pca_embedding, pca_scaler.transform(pca.transform(test)), 3)
    methods["pca_d3_interpolating"] = (AnchoredDecoder(pca_embedding, core["P"], .0001, .3), pca_embedding, pca_scaler.transform(pca.transform(test)), 3)
    common_endpoints = endpoint_indices(core["Y"])
    results = {}
    for name, (decoder, embedding, test_embedding, dimensions) in methods.items():
        decoded = decoder.predict(test_embedding)
        paths = [path_metrics(path_fn, probability, predictor, class_index, query, config.sigma, scaler, dataset, lime_slopes) for _, path_fn in paths_for_decoder(decoder, embedding, common_endpoints)]
        results[name] = {
            "feature_mae": float(np.mean(abs(decoded - test))),
            "probability_mae": float(np.mean(abs(probability(decoded) - probability(test)))),
            "dimensions": dimensions, "paths": paths,
            "alpha": decoder.alpha if isinstance(decoder, AnchoredDecoder) else .01,
            "gamma": decoder.gamma,
        }
    direct_paths = []
    for index in common_endpoints:
        endpoint = core["P"][index]
        direct_paths.append(path_metrics(lambda fractions, endpoint=endpoint: query + fractions[:, None] * (endpoint - query), probability, predictor, class_index, query, config.sigma, scaler, dataset, lime_slopes))
    results["direct_selected_line"] = {"feature_mae": None, "probability_mae": None, "paths": direct_paths}
    exact_gradient = predictor.gradient(query[None], class_index)[0]
    gradient_norm = np.linalg.norm(exact_gradient)
    finite_gradient = finite_sensitivities(probability, query[None], .02)[0]
    return {
        "class": class_index, "query_probability": float(probability(query[None])[0]),
        "constant_feature_mae": constant_error, "input_dimensions": len(query),
        "graph_components": int(connected_components(core["W"] != 0, directed=False)[0]),
        "embedding_cost": float(np.sum((core["Y"] - core["W"] @ core["Y"]) ** 2)),
        "selected_probability_std": float(np.std(probability(core["P"]))),
        "local_finite_sensitivity_cosine": float(finite_gradient @ exact_gradient / (np.linalg.norm(finite_gradient) * gradient_norm)) if np.linalg.norm(finite_gradient) * gradient_norm > 1e-10 else None,
        "local_lime_sensitivity_cosine": float(lime_slopes @ exact_gradient / (np.linalg.norm(lime_slopes) * gradient_norm)) if np.linalg.norm(lime_slopes) * gradient_norm > 1e-10 else None,
        "methods": results,
        "endpoint_training_rows": common_endpoints.tolist(),
        "blackbox_point_evaluations": predictor.probability_evaluations - query_count_before,
        "whitebox_gradient_audit_points": predictor.gradient_audit_points - audit_count_before,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("breast_cancer", "spambase"), default=["breast_cancer", "spambase"])
    parser.add_argument("--model-seeds", nargs="+", type=int, choices=(42, 43, 44), default=[42, 43, 44])
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("research/generated/full_feature/matched"))
    args = parser.parse_args()
    if args.rows < 2 or args.repeats < 1:
        parser.error("rows must be at least two and repeats positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_directory = args.output_dir / "models"
    model_directory.mkdir(exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings["original_config"] = {key: value for key, value in vars(LLEConfig()).items()}
    settings["core_sha256"] = {filename: hashlib.sha256(Path(filename).read_bytes()).hexdigest() for filename in ("run_lle.py", "lle.py", "neighbours.py", "perturbations.py")}
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1), (args.output_dir / "records.jsonl").open("w") as output:
        for dataset in args.datasets:
            for model_seed in args.model_seeds:
                predictor, scaler, test, labels, names, metadata = model_data(dataset, model_seed, model_directory)
                (model_directory / f"{dataset}_{model_seed}.json").write_text(json.dumps(metadata, indent=2) + "\n")
                probabilities = predictor.predict_proba(test)[:, 1]
                eligible = np.setdiff1d(np.arange(len(test)), [829, 21, 114] if dataset == "spambase" else [])
                boundary = eligible[np.argsort(abs(probabilities[eligible] - .5), kind="stable")[:args.rows // 2]]
                remaining = np.setdiff1d(eligible, boundary)
                random_rows = np.random.default_rng(42).choice(remaining, args.rows - len(boundary), replace=False)
                for row in np.r_[boundary, random_rows]:
                    for repeat in range(args.repeats):
                        started = time.monotonic()
                        record = evaluate(predictor, scaler, test[row], dataset, 42 + repeat)
                        record.update({"dataset": dataset, "model_seed": model_seed, "test_row": int(row), "perturbation_seed": 42 + repeat, "cohort": "boundary" if row in boundary else "random", "true_label": int(labels[row]), "feature_names": names, "seconds": time.monotonic() - started})
                        output.write(json.dumps(record, allow_nan=False) + "\n")
                        output.flush()
                    print(f"{dataset}, model {model_seed}, query {row}: complete", flush=True)


if __name__ == "__main__":
    main()
