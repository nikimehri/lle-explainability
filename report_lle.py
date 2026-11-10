"""
Report and visualisation of what run_lle.py computes, using the same config and the same pipeline (run_lle.run_lle):
  1. X's LLE reconstruction weights over its selected neighbours (and how well they rebuild X),
  2. the embedding of P = [X] + neighbours produced by lle_embedding from the weight matrix W.
No hyperparameters live here: change LLEConfig in run_lle.py.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_lle import LLEConfig, run_lle


def text_report(r : dict, top : int = 5) -> str:
    cfg, w = r["config"], r["weights"]
    classes, counts = np.unique(r["neighbour_label"], return_counts = True)
    class_counts = ", ".join(f"class {k}: {n}" for k, n in zip(classes, counts))
    order = np.argsort(-np.abs(w))[:top]
    lines = [
        f"Row {cfg.row}: model predicts class {r['label']} (probability {r['prob_X']:.2f}), true label {r['true_label']}",
        f"  Neighbours: K={cfg.K} chosen as the probability-closest of {cfg.n_samples} perturbations (sigma={cfg.sigma})",
        f"  Reconstruction error ||X - sum w_i n_i|| = {r['error']:.4f}; weights sum to {w.sum():.3f}",
        f"  Predicted class of the {cfg.K} neighbours: {class_counts} "
        f"({np.mean(r['neighbour_label'] == r['label']):.0%} match X's class {r['label']})",
        f"  Embedding: d={cfg.d}, k_graph={cfg.k_graph}, cost = {r['embedding_cost']:.4f}",
        "  Neighbours with the largest |weight| (neighbour #: weight, predicted class, probability, distance to X):",
    ]
    for j in order:
        lines.append(f"    {j:4d}: {w[j]:+.3f}, class {r['neighbour_label'][j]}, {r['neighbour_prob'][j]:.2f}, {r['distances'][j]:.3f}")
    return "\n".join(lines)


def plot_report(r : dict, path : str) -> None:
    cfg, Y, w, lab = r["config"], r["Y"], r["weights"], r["neighbour_label"]
    palette = {0 : "tab:orange", 1 : "tab:blue"}  # predicted class -> colour
    colour = [palette.get(int(k), "tab:gray") for k in lab]
    fig, ax_y = plt.subplots(figsize=(7, 5))
    handles = [plt.Line2D([], [], marker="o", ls="", color=palette.get(int(k), "tab:gray"), label=f"predicted class {k}")
               for k in np.unique(lab)]

    # lle_embedding of P = [X] + neighbours (first two axes), coloured by predicted class
    ax_y.scatter(Y[1:, 0], Y[1:, 1], c=colour, s=40, alpha=0.85)
    ax_y.scatter(Y[0, 0], Y[0, 1], marker="*", s=450, c="gold", edgecolors="black", zorder=5)
    ax_y.set_xlabel("embedding axis 1"); ax_y.set_ylabel("embedding axis 2")
    ax_y.set_title(f"lle_embedding of X + neighbours (d={cfg.d}, k_graph={cfg.k_graph}, cost {r['embedding_cost']:.3f})", fontsize=10)
    ax_y.legend(handles=[*handles, plt.Line2D([], [], marker="*", ls="", color="gold", markeredgecolor="black", markersize=14, label="X")], loc="upper left")

    fig.suptitle(f"Row {cfg.row}: model predicts class {r['label']} (probability {r['prob_X']:.2f}), true label {r['true_label']}",
                 fontsize=12, y=1.04)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(config : LLEConfig = LLEConfig(), out_dir : str = "lle_reports") -> dict:
    r = run_lle(config)
    print(text_report(r))
    plot_report(r, os.path.join(out_dir, f"row_{config.row}.png"))
    return r


if __name__ == "__main__":
    main()
