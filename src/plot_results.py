from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from utils import ROOT_DIR, ensure_output_dirs, get_logger


def comparison_bar(frame: pd.DataFrame, name_column: str, value_column: str, output_path: Path, title: str) -> None:
    """Save a bar chart for experiment comparison."""

    plt.figure(figsize=(10, 5))
    plt.bar(frame[name_column], frame[value_column])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(value_column.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def resolve_metric_column(frame: pd.DataFrame, preferred: List[str]) -> str:
    """Return the first available metric column from a preference list."""

    for column in preferred:
        if column in frame.columns:
            return column
    raise KeyError(f"None of the expected metric columns were found: {preferred}")


def main() -> None:
    """Plot experiment and ablation comparison figures from saved metric files."""

    parser = argparse.ArgumentParser(description="Plot experiment comparison charts.")
    parser.add_argument("--metrics-jsons", nargs="+", default=[])
    parser.add_argument("--ablation-summary", type=Path, default=ROOT_DIR / "outputs" / "tables" / "ablation_summary.csv")
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    logger = get_logger("plot_results", dirs["logs"] / "plot_results.log")

    if args.metrics_jsons:
        rows: List[dict] = []
        for path in args.metrics_jsons:
            with Path(path).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if "final_summary" in payload:
                rows.append(
                    {
                        "experiment_name": payload["experiment_name"],
                        "test_accuracy": payload["final_summary"]["test_accuracy"],
                        "best_val_accuracy": payload["final_summary"]["best_val_accuracy"],
                        "warmup_on_real": payload.get("warmup_on_real", False),
                        "curriculum": payload.get("curriculum", False),
                    }
                )
            else:
                rows.append(payload)
        experiments = pd.DataFrame(rows)
        experiments.to_csv(dirs["tables"] / "experiment_comparison.csv", index=False)
        comparison_bar(
            experiments,
            "experiment_name",
            "test_accuracy",
            dirs["plots"] / "experiment_comparison_accuracy.png",
            "Experiment Comparison: Test Accuracy",
        )
        logger.info("Saved experiment comparison plots.")

    if args.ablation_summary.exists():
        ablations = pd.read_csv(args.ablation_summary)
        value_column = resolve_metric_column(
            ablations,
            ["final_test_accuracy", "final_test_accuracy_mean", "test_accuracy_mean"],
        )
        comparison_bar(
            ablations,
            "experiment_name",
            value_column,
            dirs["plots"] / "ablation_comparison_accuracy.png",
            "Ablation Comparison: Test Accuracy",
        )
        logger.info("Saved ablation comparison plots.")


if __name__ == "__main__":
    main()
