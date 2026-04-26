# Proof Gallery

This document gathers the strongest visual and quantitative evidence from the repository in one place for fast review.

## Main Claim

The central claim supported by the saved outputs is:

`Selective synthetic augmentation was more useful than blind synthetic augmentation in the saved CUB-200-2011 setup.`

## Quantitative Headline

From `outputs/tables/experiment_comparison.csv`:

- `exp1_real_only`: `0.1388` accuracy
- `exp2_real_plus_all_synthetic`: `0.1367` accuracy
- `exp3_real_plus_selected_synthetic`: `0.1505` accuracy

From `outputs/tables/accepted_rejected_sample_statistics.csv`:

- generated candidates: `400`
- accepted synthetic samples: `15`
- rejected synthetic samples: `385`
- selection rate: `3.75%`

## Figure 1. Synthetic Candidate Generation

![Synthetic candidate grid](../outputs/samples/synthetic_candidate_grid.png)

This figure shows the candidate pool produced before filtering.

## Figure 2. Accepted and Rejected Sample Evidence

![Real vs synthetic vs accepted vs rejected](../outputs/samples/real_vs_synthetic_vs_accepted_vs_rejected_grid.png)

This figure illustrates the purpose of the selection module by contrasting retained and discarded samples.

## Figure 3. Main Experiment Comparison

![Main experiment comparison](../outputs/plots/experiment_comparison_accuracy.png)

This plot visually summarizes the key repository result: selected synthetic augmentation outperformed both internal controls.

## Figure 4. Ablation Comparison

![Ablation comparison](../outputs/plots/ablation_comparison_accuracy.png)

This plot shows that the selective framework is supported, while the current saved run does not justify claiming every extra module improved over confidence-only filtering.

## Figure 5. Selected Synthetic Training Evidence

![Selected synthetic loss curves](../paper/fig_selected_loss_curves_ai.png)

This figure gives a stronger paper-ready view of the saved learning behavior for the selected synthetic setting.

## Figure 6. Selected Synthetic Confusion Matrix

![Selected synthetic confusion matrix](../outputs/confusion_matrices/exp3_real_plus_selected_synthetic_confusion_matrix.png)

This figure provides additional qualitative evidence of final prediction behavior.

## Reviewer Summary

The repository is strongest when presented as:

- a reproducible AGA-inspired empirical study
- a selective sample-admission framework for synthetic augmentation
- a project with saved tables, plots, logs, checkpoints, and manuscript assets
- evidence that filtering synthetic samples mattered more than simply adding more synthetic data
