# Publication Requirements

This project is ready for a paper submission only when the evidence package below is complete.

## Required Experimental Evidence

- `outputs/tables/publication_benchmark_runs.csv` exists.
- `outputs/tables/publication_benchmark_summary.csv` exists.
- `outputs/tables/publication_benchmark_config.json` exists.
- The main benchmark includes all three experiments:
  - `exp1_real_only`
  - `exp2_real_plus_all_synthetic`
  - `exp3_real_plus_selected_synthetic`
- Each main benchmark experiment reports at least 3 seeds.
- The selected-synthetic method beats both controls in mean accuracy.
- The selected-synthetic method beats both controls in mean macro-F1.

## Required Ablation Evidence

- `outputs/tables/ablation_runs.csv` exists.
- `outputs/tables/ablation_summary.csv` exists.
- The ablation summary includes all five expected variants.
- Each ablation variant reports at least 3 seeds.
- If the title highlights CLIP, DINO, balance, or attribute coverage, the ablation table must support that claim.
- If the full method is not the best ablation, adjust the paper title and contributions accordingly.

## Required Selection Evidence

- `outputs/tables/accepted_rejected_sample_statistics.csv` exists.
- `outputs/tables/dino_pruning_stats.csv` exists.
- The final selected set is non-empty.
- If DINO removes zero samples, the paper must say DINO was not shown to help under the saved threshold.
- If the confidence threshold is unusually low, justify it or rerun the selection stage.

## Required Claim Discipline

- Call the method `AGA-inspired` unless the generation stage is matched exactly to the base paper.
- Do not claim improvement over the base paper unless `outputs/tables/base_paper_comparison.csv` is filled and supports that claim.
- Do not claim statistical significance unless you compute and report it.
- Do not treat one-seed or epoch-1 style pilot runs as final evidence.

## Required Reproducibility Package

- Final commands used for the paper are recorded.
- Seeds, backbone, image size, optimizer, patience, and augmentation settings are saved.
- Runtime metadata is preserved in metrics JSON files.
- Key figures and tables referenced in the manuscript exist in `outputs/`.

## Audit Command

Run:

```bash
python src/publication_readiness.py
```

This writes:

- `outputs/tables/publication_readiness.json`
- `paper/PUBLICATION_READINESS_REPORT.md`

Submission should wait until the report status is `READY`, or until any remaining warnings are explicitly reflected in the paper wording.
