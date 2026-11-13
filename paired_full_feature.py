"""Compute paired attribution uncertainty from the saved matched full-feature study."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiment_support import clustered_statistic, informative, load_results, RESULTS


def paired_comparison(records_source, summary, summary_sha256):
    records = []
    compact = records_source.resolve() == RESULTS.resolve()
    if compact:
        records = load_results()["full_feature"]["paired_tasks"]
    else:
        with records_source.open() as input_file:
            for line in input_file:
                record = json.loads(line)
                paths = record["methods"]["original_d3_legacy"]["paths"]
                qualified = [path for path in paths if informative(path)]
                ours = [path["attributions"]["0.02"]["reference_mae"] for path in qualified]
                lime = [path["lime_path_contribution_mae"] for path in qualified]
                differences = [left - right for left, right in zip(ours, lime)]
                records.append({"dataset": record["dataset"], "model_seed": record["model_seed"], "test_row": record["test_row"], "perturbation_seed": record["perturbation_seed"], "cohort": record["cohort"], "generated_paths": len(paths), "preserving_paths": sum(path["success_005"] for path in paths), "qualifying_paths": len(qualified), "path_attribution_mae": float(np.mean(ours)) if ours else None, "lime_mae": float(np.mean(lime)) if lime else None, "paired_mae_difference": float(np.mean(differences)) if differences else None})
    if len(records) != summary["tasks"]:
        raise ValueError("Raw task count differs from the matched study summary")
    identities = [(record["dataset"], record["model_seed"], record["test_row"], record["perturbation_seed"]) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate matched task identities")
    datasets = {}
    query_units = []
    for dataset in ("breast_cancer", "spambase"):
        selected = [record for record in records if record["dataset"] == dataset]
        expected = summary["datasets"][dataset]["original_d3_legacy"]
        generated = sum(record["generated_paths"] for record in selected)
        qualified = sum(record["qualifying_paths"] for record in selected)
        if generated != expected["paths"] or qualified != expected["informative_compensating_paths"]:
            raise ValueError(f"Audit subset differs from the published summary: {dataset}")
        estimates = {metric: clustered_statistic(selected, lambda record, metric=metric: record[metric]) for metric in ("path_attribution_mae", "lime_mae", "paired_mae_difference")}
        for metric, published in (("path_attribution_mae", expected["attribution_0.02"]["reference_mae"]), ("lime_mae", expected["lime_path_contribution_mae"])):
            np.testing.assert_allclose(estimates[metric]["mean"], published["mean"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(estimates[metric]["hierarchical_ci95"], published["hierarchical_ci95"], rtol=0, atol=1e-12)
            if estimates[metric]["model_query_units"] != published["model_query_units"]:
                raise ValueError("Model-query support differs from the published summary")
        np.testing.assert_allclose(estimates["paired_mae_difference"]["mean"], estimates["path_attribution_mae"]["mean"] - estimates["lime_mae"]["mean"], rtol=0, atol=1e-12)
        units = {}
        for record in selected:
            if record["paired_mae_difference"] is not None:
                units.setdefault((record["model_seed"], record["test_row"]), []).append(record)
        for (model_seed, test_row), tasks in units.items():
            query_units.append({"dataset": dataset, "model_seed": model_seed, "test_row": test_row, "qualifying_tasks": len(tasks), "qualifying_paths": sum(task["qualifying_paths"] for task in tasks), **{metric: float(np.mean([task[metric] for task in tasks])) for metric in estimates}})
        model_estimates = []
        for model_seed in dict.fromkeys(record["model_seed"] for record in selected):
            model_units = [unit for unit in query_units if unit["dataset"] == dataset and unit["model_seed"] == model_seed]
            model_estimates.append({"model_seed": model_seed, "model_query_units": len(model_units), **{metric: float(np.mean([unit[metric] for unit in model_units])) for metric in estimates}})
        datasets[dataset] = {"tasks": len(selected), "generated_paths": generated, "qualifying_paths": qualified, **estimates, "per_model": model_estimates}
    if compact:
        records_sha256 = load_results()["provenance"]["full_feature_raw_records_sha256"]
    else:
        with records_source.open("rb") as source_file:
            records_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
    output = {"records_sha256": records_sha256, "summary_sha256": summary_sha256, "difference_definition": "finite-contrast path-attribution MAE minus matched-cloud LIME path-contribution MAE; negative favours path attribution", "estimand": "Average within-path differences over qualifying paths within each perturbation task; average qualifying tasks within each model/query; average queries within each model; average the three model means equally.", "bootstrap": {"replicates": 3000, "seed": 42, "interval": "percentile 2.5% to 97.5%", "hierarchy": "resample model seeds and model/query units within each sampled model; keep the paired difference together", "within_query": "Repeated perturbation tasks and endpoints are averaged, not treated as independent observations", "implementation": "report_workshop.clustered_statistic, matching the existing marginal intervals"}, "conditional_subset": "Original d=3 LLE + fixed nonlinear decoder paths with endpoint RMS >=0.1 SD, sampled drift <=0.05, and exact-gradient positive and negative contribution totals each >=0.01.", "limitations": "Only three fitted models; exploratory hierarchical intervals conditional on this audit subset. Unequal classifier-query budgets. Path-preservation coverage does not establish reconstruction quality or LLE-specific superiority.", "datasets": datasets}
    return output, records, query_units


def write_analysis(output, analysis, tasks, query_units):
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_mae_analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    for filename, rows in (("paired_mae_tasks.csv", tasks), ("paired_mae_query_units.csv", query_units)):
        with (output / filename).open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, help="Optional full raw log; otherwise use the checked-in compact paired tasks")
    parser.add_argument("--output-dir", type=Path, default=Path("research/generated/paired_comparison"))
    args = parser.parse_args()
    saved = load_results()["full_feature"]
    analysis, tasks, query_units = paired_comparison(args.records or RESULTS, saved["summary"], saved["paired_analysis"]["summary_sha256"])
    analysis["input_kind"] = "raw path records" if args.records else "compact paired task averages"
    analysis["bootstrap"]["implementation"] = "experiment_support.clustered_statistic; unchanged extraction from the original analysis"
    write_analysis(args.output_dir, analysis, tasks, query_units)
    for dataset, result in analysis["datasets"].items():
        print(dataset, json.dumps(result["paired_mae_difference"]))
        print("Per-model paired differences:", [model["paired_mae_difference"] for model in result["per_model"]])


if __name__ == "__main__":
    main()
