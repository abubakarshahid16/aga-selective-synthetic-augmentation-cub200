# IEEE Submission Checklist

## Experiments

- Run `src/run_publication_benchmark.py` with at least 3 seeds.
- Run `src/ablation.py` with the same backbone and enough epochs to converge.
- Confirm `outputs/tables/publication_benchmark_runs.csv` exists.
- Confirm `outputs/tables/publication_benchmark_summary.csv` exists.
- Confirm `outputs/tables/ablation_runs.csv` exists.
- Confirm `outputs/tables/ablation_summary.csv` exists.
- Confirm the histories for all final runs contain more than one epoch unless early stopping genuinely converged late.

## Reproducibility

- Record exact commands used.
- Record hardware and runtime metadata from the saved metrics JSON files.
- Keep seeds fixed and report them explicitly.
- Preserve train/val/test split generation details.

## Paper Claims

- Use "AGA-inspired" wording unless exact reproduction is demonstrated.
- Report mean and standard deviation across seeds.
- Prefer top-1 accuracy and macro-F1 as primary metrics.
- Include a limitations section that matches the actual implementation.

## Tables To Include

- Dataset split summary.
- Model/training configuration table.
- Main benchmark results table from `publication_benchmark_summary.csv`.
- Ablation table from `ablation_summary.csv`.
- Selection statistics table from accepted/rejected and DINO stats CSVs.
- Per-class gains/losses table.

## Figures To Include

- Baseline training curves.
- Main experiment comparison plot.
- Ablation comparison plot.
- Confusion matrices for the final chosen main experiments.
- Accepted vs rejected qualitative sample grids.

## Final Sanity Checks

- README commands match the actual experiment configuration used in the paper.
- Reported backbone matches the saved metrics JSON.
- Reported epochs and patience match the executed commands.
- No debug runs are mixed into final result tables.
- The conclusion does not overclaim beyond the tables.
