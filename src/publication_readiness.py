from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from utils import ROOT_DIR, ensure_output_dirs, save_json


REQUIRED_MAIN_EXPERIMENTS = [
    "exp1_real_only",
    "exp2_real_plus_all_synthetic",
    "exp3_real_plus_selected_synthetic",
]

REQUIRED_ABLATIONS = [
    "ablation_confidence_only",
    "ablation_confidence_clip",
    "ablation_confidence_clip_dino",
    "ablation_confidence_clip_dino_balance",
    "ablation_full_attribute",
]


def record(results: List[Dict[str, str]], level: str, check: str, details: str) -> None:
    """Append one audit finding."""

    results.append({"level": level, "check": check, "details": details})


def exists(path: Path) -> bool:
    """Return whether a file exists."""

    return path.exists() and path.is_file()


def load_csv_if_present(path: Path) -> pd.DataFrame | None:
    """Load a CSV file if it exists."""

    if not exists(path):
        return None
    return pd.read_csv(path)


def evaluate_main_benchmark(tables_dir: Path, results: List[Dict[str, str]]) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Validate the main multi-seed benchmark artifacts."""

    runs_path = tables_dir / "publication_benchmark_runs.csv"
    summary_path = tables_dir / "publication_benchmark_summary.csv"
    config_path = tables_dir / "publication_benchmark_config.json"

    runs = load_csv_if_present(runs_path)
    summary = load_csv_if_present(summary_path)

    if exists(runs_path):
        record(results, "pass", "benchmark_runs_file", f"Found {runs_path.name}.")
    else:
        record(results, "fail", "benchmark_runs_file", f"Missing {runs_path.name}.")

    if exists(summary_path):
        record(results, "pass", "benchmark_summary_file", f"Found {summary_path.name}.")
    else:
        record(results, "fail", "benchmark_summary_file", f"Missing {summary_path.name}.")

    if exists(config_path):
        record(results, "pass", "benchmark_config_file", f"Found {config_path.name}.")
    else:
        record(results, "fail", "benchmark_config_file", f"Missing {config_path.name}.")

    if summary is None:
        return runs, summary

    missing = [name for name in REQUIRED_MAIN_EXPERIMENTS if name not in summary["experiment_name"].tolist()]
    if missing:
        record(results, "fail", "benchmark_experiments_present", f"Missing main experiments in summary: {missing}.")
    else:
        record(results, "pass", "benchmark_experiments_present", "All three main experiments are present in the summary.")

    if "num_seeds" in summary.columns:
        too_few = summary.loc[summary["num_seeds"] < 3, ["experiment_name", "num_seeds"]]
        if len(too_few):
            record(results, "fail", "benchmark_num_seeds", f"Some benchmark experiments use fewer than 3 seeds: {too_few.to_dict(orient='records')}.")
        else:
            record(results, "pass", "benchmark_num_seeds", "All benchmark experiments report at least 3 seeds.")
    else:
        record(results, "fail", "benchmark_num_seeds", "Summary does not contain a num_seeds column.")

    if {"experiment_name", "test_accuracy_mean", "test_macro_f1_mean"}.issubset(summary.columns):
        indexed = summary.set_index("experiment_name")
        baseline_acc = float(indexed.loc["exp1_real_only", "test_accuracy_mean"])
        all_synth_acc = float(indexed.loc["exp2_real_plus_all_synthetic", "test_accuracy_mean"])
        selected_acc = float(indexed.loc["exp3_real_plus_selected_synthetic", "test_accuracy_mean"])
        baseline_f1 = float(indexed.loc["exp1_real_only", "test_macro_f1_mean"])
        all_synth_f1 = float(indexed.loc["exp2_real_plus_all_synthetic", "test_macro_f1_mean"])
        selected_f1 = float(indexed.loc["exp3_real_plus_selected_synthetic", "test_macro_f1_mean"])

        if selected_acc > baseline_acc and selected_acc > all_synth_acc:
            record(results, "pass", "selected_beats_main_baselines_accuracy", "Selected synthetic outperforms both baseline and all-synthetic in mean accuracy.")
        else:
            record(results, "fail", "selected_beats_main_baselines_accuracy", f"Mean accuracy does not show selected synthetic beating both controls: baseline={baseline_acc:.4f}, all={all_synth_acc:.4f}, selected={selected_acc:.4f}.")

        if selected_f1 > baseline_f1 and selected_f1 > all_synth_f1:
            record(results, "pass", "selected_beats_main_baselines_macro_f1", "Selected synthetic outperforms both baseline and all-synthetic in mean macro-F1.")
        else:
            record(results, "fail", "selected_beats_main_baselines_macro_f1", f"Mean macro-F1 does not show selected synthetic beating both controls: baseline={baseline_f1:.4f}, all={all_synth_f1:.4f}, selected={selected_f1:.4f}.")

    return runs, summary


def evaluate_ablation(tables_dir: Path, results: List[Dict[str, str]]) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Validate the ablation artifacts."""

    runs_path = tables_dir / "ablation_runs.csv"
    summary_path = tables_dir / "ablation_summary.csv"
    runs = load_csv_if_present(runs_path)
    summary = load_csv_if_present(summary_path)

    if exists(runs_path):
        record(results, "pass", "ablation_runs_file", f"Found {runs_path.name}.")
    else:
        record(results, "fail", "ablation_runs_file", f"Missing {runs_path.name}.")

    if exists(summary_path):
        record(results, "pass", "ablation_summary_file", f"Found {summary_path.name}.")
    else:
        record(results, "fail", "ablation_summary_file", f"Missing {summary_path.name}.")

    if summary is None:
        return runs, summary

    missing = [name for name in REQUIRED_ABLATIONS if name not in summary["experiment_name"].tolist()]
    if missing:
        record(results, "fail", "ablation_variants_present", f"Missing ablation variants in summary: {missing}.")
    else:
        record(results, "pass", "ablation_variants_present", "All expected ablation variants are present in the summary.")

    if "num_seeds" in summary.columns:
        too_few = summary.loc[summary["num_seeds"] < 3, ["experiment_name", "num_seeds"]]
        if len(too_few):
            record(results, "fail", "ablation_num_seeds", f"Some ablations use fewer than 3 seeds: {too_few.to_dict(orient='records')}.")
        else:
            record(results, "pass", "ablation_num_seeds", "All ablation variants report at least 3 seeds.")
    else:
        record(results, "warn", "ablation_num_seeds", "Ablation summary does not contain num_seeds.")

    score_column = "final_test_accuracy_mean" if "final_test_accuracy_mean" in summary.columns else None
    if score_column is not None:
        best_row = summary.sort_values(score_column, ascending=False).iloc[0]
        if str(best_row["experiment_name"]) == "ablation_full_attribute":
            record(results, "pass", "full_method_supported_by_ablation", "The full proposed method is the best ablation by mean accuracy.")
        else:
            record(results, "warn", "full_method_supported_by_ablation", f"The best ablation is {best_row['experiment_name']} rather than ablation_full_attribute.")

    return runs, summary


def evaluate_selection_stats(tables_dir: Path, results: List[Dict[str, str]]) -> None:
    """Validate the selection and pruning evidence."""

    accepted_path = tables_dir / "accepted_rejected_sample_statistics.csv"
    dino_path = tables_dir / "dino_pruning_stats.csv"
    base_paper_path = tables_dir / "base_paper_comparison.csv"

    accepted = load_csv_if_present(accepted_path)
    dino = load_csv_if_present(dino_path)

    if accepted is not None and len(accepted):
        row = accepted.iloc[0]
        record(results, "pass", "selection_stats_present", f"Selection stats found with {int(row['num_selected'])} selected of {int(row['num_candidates'])} candidates.")
        if float(row["confidence_threshold"]) < 0.1:
            record(results, "warn", "selection_threshold_strength", f"Saved selection confidence threshold is low ({float(row['confidence_threshold']):.4f}); justify or rerun with stronger settings.")
        if int(row["num_selected"]) == 0:
            record(results, "fail", "selection_nonempty", "No synthetic samples were selected.")
    else:
        record(results, "fail", "selection_stats_present", "Missing accepted_rejected_sample_statistics.csv.")

    if dino is not None and len(dino):
        row = dino.iloc[0]
        removed = int(row["num_removed"])
        if removed > 0:
            record(results, "pass", "dino_effect_observed", f"DINO pruning removed {removed} samples.")
        else:
            record(results, "warn", "dino_effect_observed", "DINO pruning removed 0 samples in the saved run, so its value is not demonstrated.")
    else:
        record(results, "fail", "dino_stats_present", "Missing dino_pruning_stats.csv.")

    if exists(base_paper_path):
        comparison = pd.read_csv(base_paper_path)
        if {"comparison_ready", "claim_allowed"}.issubset(comparison.columns) and len(comparison):
            row = comparison.iloc[0]
            if bool(row["comparison_ready"]) and bool(row["claim_allowed"]):
                record(results, "pass", "base_paper_comparison_ready", "Direct base-paper comparison file says the claim is allowed.")
            else:
                record(results, "warn", "base_paper_comparison_ready", "Base-paper comparison file exists but does not yet support an 'improves the base paper' claim.")
        else:
            record(results, "warn", "base_paper_comparison_ready", "base_paper_comparison.csv exists but does not use the expected schema.")
    else:
        record(results, "warn", "base_paper_comparison_ready", "Missing base_paper_comparison.csv, so you cannot honestly claim improvement over the base paper yet.")


def write_markdown_report(output_path: Path, findings: List[Dict[str, str]], ready: bool) -> None:
    """Write a human-readable audit report."""

    lines = [
        "# Publication Readiness Report",
        "",
        f"Overall status: {'READY' if ready else 'NOT READY'}",
        "",
        "## Findings",
        "",
    ]
    for item in findings:
        lines.append(f"- `{item['level'].upper()}` {item['check']}: {item['details']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Audit whether the saved outputs support publication-style claims."""

    parser = argparse.ArgumentParser(description="Audit publication readiness from saved benchmark outputs.")
    parser.add_argument("--tables-dir", type=Path, default=ROOT_DIR / "outputs" / "tables")
    parser.add_argument("--output-json", type=Path, default=ROOT_DIR / "outputs" / "tables" / "publication_readiness.json")
    parser.add_argument("--output-md", type=Path, default=ROOT_DIR / "paper" / "PUBLICATION_READINESS_REPORT.md")
    args = parser.parse_args()

    ensure_output_dirs()
    findings: List[Dict[str, str]] = []

    evaluate_main_benchmark(args.tables_dir, findings)
    evaluate_ablation(args.tables_dir, findings)
    evaluate_selection_stats(args.tables_dir, findings)

    ready = not any(item["level"] == "fail" for item in findings)
    payload = {
        "ready": ready,
        "num_fail": sum(item["level"] == "fail" for item in findings),
        "num_warn": sum(item["level"] == "warn" for item in findings),
        "num_pass": sum(item["level"] == "pass" for item in findings),
        "findings": findings,
    }
    save_json(payload, args.output_json)
    write_markdown_report(args.output_md, findings, ready)


if __name__ == "__main__":
    main()
