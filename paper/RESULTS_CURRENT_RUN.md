# Current Results Draft

This file summarizes the results currently present in `outputs/tables/` as of April 24, 2026.

Important scope note: these are single-seed, single-run results from the saved repository outputs, not the multi-seed publication benchmark requested in the paper outline. Treat this file as a drafting aid, not as final evidence for the paper.

## Main Results

Backbone: `resnet18`

Number of seeds: `1`

Real-only baseline:
- Accuracy: `0.1388`
- Macro-F1: `0.1111`

Real + all synthetic:
- Accuracy: `0.1367`
- Macro-F1: `0.1025`

Real + selected synthetic:
- Accuracy: `0.1505`
- Macro-F1: `0.1206`

Best method by accuracy among the three main experiments: `exp3_real_plus_selected_synthetic`

Observed change versus real-only baseline:
- Real + all synthetic: `-0.0021` accuracy, `-0.0086` macro-F1
- Real + selected synthetic: `+0.0117` accuracy, `+0.0095` macro-F1

## Ablation Results

These values are also single-run results.

Confidence only:
- Accuracy: `0.1555`
- Macro-F1: `0.1239`

Confidence + CLIP:
- Accuracy: `0.1505`
- Macro-F1: `0.1206`

Confidence + CLIP + DINO:
- Accuracy: `0.1505`
- Macro-F1: `0.1206`

Confidence + CLIP + DINO + balance:
- Accuracy: `0.1505`
- Macro-F1: `0.1206`

Confidence + CLIP + DINO + balance + attribute coverage:
- Accuracy: `0.1505`
- Macro-F1: `0.1206`

Best ablation by accuracy in the current run: `ablation_confidence_only`

Interpretation for the current run:
- Selective augmentation outperformed both the real-only baseline and adding all synthetic samples.
- The extra CLIP, DINO, balance, and attribute-coverage stages did not outperform the confidence-only variant in this saved run.
- DINO pruning did not remove any candidates at the recorded threshold, so its contribution was not demonstrated here.

## Selection Statistics

Candidates generated: `400`

Candidates kept after final selection: `15`

Selection rate: `0.0375` (`3.75%`)

Candidates surviving confidence + CLIP + DINO before final balancing: `15`

DINO keep ratio: `1.0000` (`100%`)

Confidence threshold used in the saved selection statistics: `0.02`

CLIP threshold used in the saved selection statistics: `0.20`

Max synthetic-to-real ratio used in the saved selection statistics: `0.75`

Attribute-aware coverage enabled in the saved selection statistics: `True`

## Honest Conclusion Draft

In the current single-seed saved run, the proposed selective augmentation pipeline did improve top-1 accuracy over the real-only baseline on CUB-200-2011, and it also outperformed the naive real-plus-all-synthetic condition. The strongest evidence among the three main experiments came from `exp3_real_plus_selected_synthetic`, while the ablation results did not support an additional benefit from CLIP, DINO, class-balance, or attribute-aware coverage beyond confidence-only filtering in this run. Because the generation stage is AGA-inspired rather than an exact reproduction, and because multi-seed publication benchmarks are not yet present in the repository outputs, claims should remain limited to this implemented pipeline and this currently saved run.

## Source Files Used

- `outputs/tables/experiment_comparison.csv`
- `outputs/tables/ablation_summary.csv`
- `outputs/tables/accepted_rejected_sample_statistics.csv`
- `outputs/tables/dino_pruning_stats.csv`
- `outputs/tables/exp1_real_only_metrics.json`
- `outputs/tables/exp2_real_plus_all_synthetic_metrics.json`
- `outputs/tables/exp3_real_plus_selected_synthetic_metrics.json`
- `outputs/tables/exp1_real_only_classification_report.csv`
- `outputs/tables/exp2_real_plus_all_synthetic_classification_report.csv`
- `outputs/tables/exp3_real_plus_selected_synthetic_classification_report.csv`
- `outputs/tables/ablation_confidence_only_classification_report.csv`
- `outputs/tables/ablation_confidence_clip_classification_report.csv`
- `outputs/tables/ablation_confidence_clip_dino_classification_report.csv`
- `outputs/tables/ablation_confidence_clip_dino_balance_classification_report.csv`
- `outputs/tables/ablation_full_attribute_classification_report.csv`
