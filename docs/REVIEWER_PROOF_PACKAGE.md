# Reviewer Proof Package

This document is designed for a teacher, supervisor, or reviewer who wants to quickly verify that the repository is organized like a serious research submission.

## Best High-Level Description

Present the work as:

`A practical AGA-inspired selective synthetic augmentation framework for fine-grained bird classification on CUB-200-2011, evaluated as a reproducible empirical study with saved qualitative and quantitative evidence.`

## Why The Project Is Convincing

- it studies a real research question rather than a toy task
- it includes internal controls instead of only one headline experiment
- it preserves outputs, checkpoints, tables, plots, and qualitative evidence
- it includes ablation evidence and not just a single result screenshot
- it is connected to a published base paper and clearly states its own scope
- it is packaged with a manuscript, proof gallery, and supporting notes

## Fast Review Path

For the quickest convincing walkthrough, review these in order:

1. `README.md`
2. `docs/PROOF_GALLERY.md`
3. `paper/ieee_conference_paper.tex`
4. `outputs/tables/experiment_comparison.csv`
5. `outputs/tables/ablation_summary.csv`
6. `outputs/tables/accepted_rejected_sample_statistics.csv`

## Stage By Stage Proof

### Stage 1. Dataset Preparation

Purpose:
show that the project starts from a real fine-grained benchmark and a clean data split.

Evidence:

- `data/prepared/`
- `outputs/plots/real_train_distribution.png`

Key message:

`The project uses a structured CUB-200-2011 pipeline rather than ad hoc image folders.`

![Real train distribution](../outputs/plots/real_train_distribution.png)

### Stage 2. Synthetic Candidate Generation

Purpose:
show that the project generates a meaningful synthetic candidate pool before filtering.

Evidence:

- `outputs/samples/synthetic_candidates/`
- `outputs/samples/synthetic_candidate_grid.png`
- `outputs/samples/synthetic_candidates/synthetic_manifest.csv`

Key message:

`Synthetic data is produced systematically and tracked through a manifest-based workflow.`

![Synthetic candidate grid](../outputs/samples/synthetic_candidate_grid.png)

### Stage 3. Selection and Rejection Logic

Purpose:
show that the work is not just generating images, but making evidence-based decisions about which ones should be used.

Evidence:

- `outputs/tables/synthetic_classifier_scores.csv`
- `outputs/tables/synthetic_clip_scores.csv`
- `outputs/tables/synthetic_dino_pruned.csv`
- `outputs/tables/selected_synthetic_manifest.csv`
- `outputs/tables/accepted_rejected_sample_statistics.csv`
- `outputs/samples/accepted_synthetic_grid.png`
- `outputs/samples/rejected_synthetic_grid.png`
- `outputs/samples/real_vs_synthetic_vs_accepted_vs_rejected_grid.png`

Key message:

`The central contribution is selective admission, and the repository preserves direct proof of that decision process.`

![Accepted synthetic grid](../outputs/samples/accepted_synthetic_grid.png)

![Rejected synthetic grid](../outputs/samples/rejected_synthetic_grid.png)

![Real vs synthetic vs accepted vs rejected](../outputs/samples/real_vs_synthetic_vs_accepted_vs_rejected_grid.png)

### Stage 4. Main Experimental Comparison

Purpose:
show that the project compares multiple training conditions instead of reporting one isolated result.

Evidence:

- `outputs/tables/experiment_comparison.csv`
- `outputs/plots/experiment_comparison_accuracy.png`
- `outputs/plots/exp1_real_only_accuracy_curves.png`
- `outputs/plots/exp2_real_plus_all_synthetic_accuracy_curves.png`
- `paper/fig_selected_loss_curves_ai.png`

Key message:

`The saved outputs support the claim that selected synthetic augmentation performed better than both internal controls in the tested setup.`

![Experiment comparison](../outputs/plots/experiment_comparison_accuracy.png)

### Stage 5. Ablation Evidence

Purpose:
show that the project investigates which parts of the method matter.

Evidence:

- `outputs/tables/ablation_summary.csv`
- `outputs/plots/ablation_comparison_accuracy.png`

Key message:

`The ablations strengthen the paper by showing analytical depth, even when the results suggest that the broader framework is stronger than the case for every individual module.`

![Ablation comparison](../outputs/plots/ablation_comparison_accuracy.png)

### Stage 6. Qualitative Prediction Evidence

Purpose:
show that the work preserves evaluation artifacts beyond a single scalar metric.

Evidence:

- `outputs/confusion_matrices/exp1_real_only_confusion_matrix.png`
- `outputs/confusion_matrices/exp2_real_plus_all_synthetic_confusion_matrix.png`
- `outputs/confusion_matrices/exp3_real_plus_selected_synthetic_confusion_matrix.png`

Key message:

`The repository includes model-level evaluation outputs expected from a serious experimental project.`

![Selected synthetic confusion matrix](../outputs/confusion_matrices/exp3_real_plus_selected_synthetic_confusion_matrix.png)

## Base Paper Comparison Guidance

Use the base paper comparison in this way:

- cite the AGA paper as the conceptual foundation
- explain that this repository is AGA-inspired rather than an exact reproduction
- emphasize the selective admission contribution, saved artifact trail, and fine-grained classification focus
- do not claim direct superiority to the base paper unless a like-for-like matched comparison is completed

## Best Teacher Facing Summary

Use this wording in slides, viva, or discussions:

`This project is convincing because it does not stop at generating synthetic images. It builds a full reproducible pipeline for selecting, evaluating, and justifying synthetic augmentation, and it preserves the complete experimental trail through tables, plots, sample grids, checkpoints, and manuscript-ready figures.`

## Best Paper Facing Summary

Use this wording in the manuscript or project synopsis:

`The strongest saved evidence supports a scoped empirical claim: in the tested CUB-200-2011 setup, selectively admitted synthetic samples were more useful than either no synthetic augmentation or blind use of all generated samples.`
