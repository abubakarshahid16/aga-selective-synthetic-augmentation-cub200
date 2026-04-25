# Publication Readiness Report

Overall status: NOT READY

## Findings

- `FAIL` benchmark_runs_file: Missing publication_benchmark_runs.csv.
- `FAIL` benchmark_summary_file: Missing publication_benchmark_summary.csv.
- `FAIL` benchmark_config_file: Missing publication_benchmark_config.json.
- `FAIL` ablation_runs_file: Missing ablation_runs.csv.
- `PASS` ablation_summary_file: Found ablation_summary.csv.
- `PASS` ablation_variants_present: All expected ablation variants are present in the summary.
- `WARN` ablation_num_seeds: Ablation summary does not contain num_seeds.
- `PASS` selection_stats_present: Selection stats found with 15 selected of 400 candidates.
- `WARN` selection_threshold_strength: Saved selection confidence threshold is low (0.0200); justify or rerun with stronger settings.
- `WARN` dino_effect_observed: DINO pruning removed 0 samples in the saved run, so its value is not demonstrated.
- `WARN` base_paper_comparison_ready: Base-paper comparison file exists but does not yet support an 'improves the base paper' claim.
