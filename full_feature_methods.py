"""Exact original LLE stages with separately fitted, query-anchored decoders."""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from lle import build_weight_matrix, compute_lle_weights, lle_embedding
from neighbours import select_neighbours
from perturbations import generate_perturbation


class NonlinearDecoder:
    def __init__(self, points, embedding, alpha=.01, gamma=None):
        self.query = points[0].copy()
        self.query_coordinate = embedding[0].copy()
        self.gamma = 1 / embedding.shape[1] if gamma is None else gamma
        self.model = KernelRidge(kernel="rbf", gamma=self.gamma, alpha=alpha).fit(embedding, points - self.query)
        self.offset = self.model.predict(embedding[:1])[0]

    def predict(self, coordinates):
        return self.query + self.model.predict(np.atleast_2d(coordinates)) - self.offset

    def jacobian(self, coordinate):
        differences = np.asarray(coordinate) - self.model.X_fit_
        kernel = np.exp(-self.gamma * np.sum(differences ** 2, axis=1))
        derivatives = -2 * self.gamma * kernel[:, None] * differences
        return self.model.dual_coef_.T @ derivatives


class LinearDecoder:
    """Affine ridge reconstruction with the same query correction as the RBF decoder."""

    def __init__(self, points, embedding, alpha=.01):
        self.query = points[0].copy()
        self.query_coordinate = embedding[0].copy()
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=False, solver="svd").fit(embedding, points - self.query)

    def predict(self, coordinates):
        return self.query + self.model.predict(np.atleast_2d(coordinates) - self.query_coordinate)

    def jacobian(self, coordinate):
        return self.model.coef_.copy()


def finite_sensitivities(probability_fn, points, step):
    offsets = step * np.eye(points.shape[1])
    plus = (points[:, None, :] + offsets[None, :, :]).reshape(-1, points.shape[1])
    minus = (points[:, None, :] - offsets[None, :, :]).reshape(-1, points.shape[1])
    return ((probability_fn(plus) - probability_fn(minus)) / (2 * step)).reshape(len(points), -1)


def original_core(query, probability_fn, config):
    candidates = generate_perturbation(query, config.n_samples, config.sigma, np.random.default_rng(config.seed))
    neighbours = select_neighbours(query, candidates, probability_fn, config.K)
    reg = getattr(config, "reg", 1e-3)
    query_weights = compute_lle_weights(query, neighbours, reg=reg)
    points = np.vstack([query, neighbours])
    weights = build_weight_matrix(points, config.k_graph, reg=reg)
    embedding = lle_embedding(weights, config.d)
    return {"P": points, "W": weights, "Y": embedding, "weights": query_weights, "candidates": candidates}


def independent_selected(query, probability_fn, config, seed):
    candidates = generate_perturbation(query, config.n_samples, config.sigma, np.random.default_rng(seed))
    return select_neighbours(query, candidates, probability_fn, config.K)


def extend_coordinates(points, core, count, reg=1e-3):
    coordinates = []
    for point in points:
        indices = np.argsort(np.linalg.norm(core["P"] - point, axis=1), kind="stable")[:count]
        weights = compute_lle_weights(point, core["P"][indices], reg=reg)
        coordinates.append(weights @ core["Y"][indices])
    return np.array(coordinates)


class AnchoredDecoder:
    def __init__(self, coordinates, points, alpha, gamma):
        self.coordinates = coordinates.copy()
        self.anchor = coordinates[0].copy()
        self.query = points[0].copy()
        self.gamma = gamma
        self.alpha = alpha
        self.anchor_training_kernel = np.exp(-gamma * cdist(self.anchor[None], coordinates, "sqeuclidean"))
        kernel = self.kernel(coordinates)
        self.model = KernelRidge(alpha=alpha, kernel="precomputed").fit(kernel, points - self.query)

    def kernel(self, coordinates):
        cross = np.exp(-self.gamma * cdist(coordinates, self.coordinates, "sqeuclidean"))
        anchor = np.exp(-self.gamma * cdist(coordinates, self.anchor[None], "sqeuclidean"))
        return cross - anchor - self.anchor_training_kernel + 1

    def predict(self, coordinates):
        return self.query + self.model.predict(self.kernel(np.atleast_2d(coordinates)))

    def jacobian(self, coordinate):
        differences = np.asarray(coordinate) - self.coordinates
        kernel = np.exp(-self.gamma * np.sum(differences ** 2, axis=1))
        derivatives = -2 * self.gamma * kernel[:, None] * differences
        anchor_difference = np.asarray(coordinate) - self.anchor
        anchor_derivative = -2 * self.gamma * np.exp(-self.gamma * np.sum(anchor_difference ** 2)) * anchor_difference
        return self.model.dual_coef_.T @ (derivatives - anchor_derivative)


def select_decoder(coordinates, points, validation_coordinates, validation_points):
    choices = []
    for gamma in (.03, .1, .3, 1.):
        for alpha in (.001, .01, .1, 1.):
            decoder = AnchoredDecoder(coordinates, points, alpha=alpha, gamma=gamma)
            error = float(np.mean(abs(decoder.predict(validation_coordinates) - validation_points)))
            choices.append((error, decoder))
    return min(choices, key=lambda choice: choice[0])


def decoded_paths(decoder, embedding, count=3, length=33):
    endpoint_indices = np.argsort(np.linalg.norm(embedding[1:] - embedding[0], axis=1), kind="stable")[:count] + 1
    paths = []
    for endpoint in endpoint_indices:
        coordinates = embedding[0] + np.linspace(0, 1, length)[:, None] * (embedding[endpoint] - embedding[0])
        paths.append((int(endpoint), coordinates, decoder.predict(coordinates)))
    return paths


def pca_coordinates(points, validation, test, dimensions):
    estimator = PCA(n_components=dimensions).fit(points)
    return estimator.transform(points), estimator.transform(validation), estimator.transform(test)
