"""
Simple LLE report for one row X of a tabular dataset and any classifier:
  1. the K nearest real points to X and how many of them the model puts in X's class,
  2. the neighbours that rebuild X best (largest LLE weights),
  3. a 2D picture of X among its neighbours (LLE embedding).
"""
import os
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lle import compute_lle_weights, build_weight_matrix, lle_embedding


@dataclass
class ReportConfig:
    K : int = 50            # nearest real points used as X's neighbours
    k_graph : int = 15      # neighbours each point is rebuilt from when building the embedding
    d : int = 5             # embedding dimension (must be < number of features)
    reg : float = 1e-1      # regularisation for X's own weights
    embedding_reg : float = 1e-3  # regularisation for the weights behind the embedding
    top_examples : int = 5  # rows shown in the examples table


def explain_instance(
        index : int,
        data : np.ndarray,
        predict_proba_fn,
        feature_names : list,
        labels : np.ndarray | None = None,
        config : ReportConfig = ReportConfig(),
    ) -> dict:
    n, D = data.shape
    if config.K >= n:
        raise ValueError("K must be smaller than the number of rows in data.")
    if config.d >= D:
        raise ValueError("d must be smaller than the number of features.")

    Z = (data - data.mean(axis=0)) / np.where(data.std(axis=0) == 0, 1.0, data.std(axis=0))
    probs = predict_proba_fn(data)
    pred = probs.argmax(axis=1)
    c = int(pred[index])

    # K nearest other real points
    others = np.delete(np.arange(n), index)
    dist = np.linalg.norm(Z[others] - Z[index], axis=1)
    order = np.argsort(dist)[:config.K]
    nb, nb_dist = others[order], dist[order]

    # LLE weights of X (positive part, renormalised to sum to 1)
    w = compute_lle_weights(Z[index], Z[nb], config.reg)
    wp = np.clip(w, 0, None)
    wp = wp / wp.sum()

    # Embedding of X (row 0) and its neighbours
    P = np.vstack([Z[index], Z[nb]])
    Y = lle_embedding(build_weight_matrix(P, config.k_graph, config.embedding_reg), config.d)

    # The feature most correlated with each of the first two axes (to label the plot)
    axis_feature = []
    for a in range(2):
        corr = [abs(np.corrcoef(Y[:, a], P[:, j])[0, 1]) if P[:, j].std() > 0 else 0 for j in range(D)]
        axis_feature.append(feature_names[int(np.argmax(corr))])

    in_class = pred[nb] == c
    return {
        "index" : index, "class" : c, "prob" : float(probs[index, c]),
        "true_label" : None if labels is None else int(labels[index]),
        "neighbours" : nb, "distances" : nb_dist, "weights" : w, "positive_weights" : wp,
        "neighbour_pred" : pred[nb], "neighbour_prob" : probs[nb, c],
        "neighbour_true" : None if labels is None else labels[nb],
        "agreement" : float(in_class.mean()), "agreement_weighted" : float(wp @ in_class),
        "Y" : Y, "axis_feature" : axis_feature, "config" : config,
    }


def _title(r : dict, cn) -> str:
    t = f"Row {r['index']}: model predicts {cn(r['class'])} (probability {r['prob']:.2f})"
    if r["true_label"] is not None:
        t += f", true label {cn(r['true_label'])}"
    return t


def text_report(r : dict, class_names : list | None = None) -> str:
    cn = (lambda k: class_names[k]) if class_names is not None else (lambda k: str(k))
    cfg = r["config"]
    top = np.argsort(-r["weights"])[:cfg.top_examples]
    lines = [
        _title(r, cn),
        f"  {r['agreement']:.0%} of its {cfg.K} nearest points are in the same class "
        f"({r['agreement_weighted']:.0%} when weighted by LLE weight)",
        "  Points that rebuild X best (row: weight, class, distance):",
    ]
    for j in top:
        lines.append(f"    {r['neighbours'][j]:4d}: {r['weights'][j]:+.3f}, {cn(r['neighbour_pred'][j])}, {r['distances'][j]:.2f}")
    return "\n".join(lines)


def plot_report(r : dict, path : str, class_names : list | None = None) -> None:
    cn = (lambda k: class_names[k]) if class_names is not None else (lambda k: str(k))
    cfg, Y, c = r["config"], r["Y"], r["class"]
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1]})

    # Left: X among its neighbours. Colour = how sure the model is of X's class, size = LLE weight.
    sc = ax.scatter(Y[1:, 0], Y[1:, 1], c=r["neighbour_prob"], cmap="coolwarm_r", vmin=0, vmax=1,
                    s=25 + 900 * r["positive_weights"], alpha=0.85)
    ax.scatter(Y[0, 0], Y[0, 1], marker="*", s=450, c="gold", edgecolors="black", zorder=5, label="X")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"axis 1 (follows {r['axis_feature'][0]})")
    ax.set_ylabel(f"axis 2 (follows {r['axis_feature'][1]})")
    ax.set_title("X and its nearest points\nbigger dot = helps rebuild X more", fontsize=10)
    fig.colorbar(sc, ax=ax, shrink=0.8, label=f"model probability of {cn(c)}")
    ax.legend(loc="upper left")

    # Right: the points that rebuild X best
    tx.axis("off")
    top = np.argsort(-r["weights"])[:cfg.top_examples]
    cells = [[str(r["neighbours"][j]), f"{r['weights'][j]:+.2f}", cn(r["neighbour_pred"][j]),
              f"{r['distances'][j]:.2f}"] for j in top]
    tbl = tx.table(cellText=cells, colLabels=["row", "weight", "model class", "distance"], bbox=[0, 0.3, 1, 0.6], cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tx.set_title("Points that rebuild X best", fontsize=10)
    tx.text(0.5, 0.1, f"{r['agreement']:.0%} of the {cfg.K} nearest points share X's predicted class",
            ha="center", fontsize=11, transform=tx.transAxes)

    fig.suptitle(_title(r, cn), fontsize=12)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_reports(
        indices,
        data : np.ndarray,
        predict_proba_fn,
        feature_names : list,
        labels : np.ndarray | None = None,
        class_names : list | None = None,
        config : ReportConfig = ReportConfig(),
        out_dir : str = "lle_reports",
    ) -> list:
    results = []
    for i in indices:
        r = explain_instance(i, data, predict_proba_fn, feature_names, labels, config)
        plot_report(r, os.path.join(out_dir, f"row_{i}.png"), class_names)
        print(text_report(r, class_names), "\n")
        results.append(r)
    return results


def main():
    """Example only: the dataset and model are chosen here and nowhere else."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier

    dataset = load_breast_cancer()
    model = RandomForestClassifier(random_state = 0).fit(dataset.data, dataset.target)
    make_reports(
        indices = [38, 73],
        data = dataset.data,
        predict_proba_fn = model.predict_proba,
        feature_names = list(dataset.feature_names),
        labels = dataset.target,
        class_names = list(dataset.target_names),
        config = ReportConfig(),
        out_dir = "lle_reports",
    )


if __name__ == "__main__":
    main()
