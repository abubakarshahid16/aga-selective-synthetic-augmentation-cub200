# IEEE Paper Outline

## Working Title

An AGA-Inspired Selective Synthetic Augmentation Framework for Fine-Grained Bird Classification on CUB-200-2011

## Abstract Draft

This paper studies whether synthetic augmentation can improve fine-grained bird classification on CUB-200-2011 when generated candidates are filtered before training rather than accepted blindly. We use an AGA-inspired foreground-preserving generation pipeline and evaluate a selective admission module based on classifier confidence, CLIP semantic consistency, DINO diversity pruning, class-balance constraints, and optional attribute-aware coverage. The central claim should be that selective filtering can outperform naive all-synthetic augmentation under the implemented setup if supported by the saved benchmark files. To avoid overstating evidence, all final quantitative claims should come from `outputs/tables/publication_benchmark_summary.csv` and `outputs/tables/ablation_summary.csv`, with mean and standard deviation reported across seeds.

## Suggested Section Structure

### I. Introduction

- Motivate fine-grained classification difficulty on CUB-200-2011.
- Explain why synthetic augmentation is attractive but noisy.
- State the paper contribution as selective synthetic-data acceptance rather than blind augmentation.
- Emphasize the novelty as a practical selective augmentation framework, not simply a larger synthetic dataset.
- Keep the AGA wording honest: "AGA-inspired" unless exact upstream generation is reproduced.

### II. Related Work

- Fine-grained bird classification on CUB-200-2011.
- Diffusion or generative augmentation for image classification.
- CLIP-based filtering and semantic validation.
- DINO or self-supervised embedding methods for diversity control.

### III. Method

#### A. Dataset and Splits

- Cite the official CUB-200-2011 split.
- Describe your train/validation split from the official training partition.
- Mention optional use of CUB image-level attributes as proxy metadata.

#### B. Synthetic Candidate Generation

- Describe the implemented foreground-preserving background-swap pipeline.
- State clearly that this is a practical approximation rather than an exact AGA reproduction.

#### C. Proposed Selection Module

- Classifier confidence matching.
- CLIP intended-class matching and thresholding.
- DINO diversity pruning.
- Class-balance quota.
- Optional attribute-aware coverage.
- Make clear that the full contribution is the overall selection framework even if not every component wins the ablation.

#### D. Training Protocol

- Backbone.
- Image size.
- Optimizer and learning-rate schedule.
- Epochs, patience, label smoothing.
- Number of seeds.
- Hardware/runtime note.

### IV. Experimental Setup

- Main experiments:
  - real only
  - real + all synthetic
  - real + selected synthetic
- Ablation experiments.
- Metrics:
  - top-1 accuracy
  - macro-F1
  - weighted-F1
  - mean +- std across seeds
- Minimum publishable evidence:
  - selected synthetic beats both controls in the main benchmark
  - at least one compact ablation explains which parts of the selection framework matter

### V. Results

#### A. Main Benchmark Table

Populate from `outputs/tables/publication_benchmark_summary.csv`.

#### B. Ablation Table

Populate from `outputs/tables/ablation_summary.csv`.

#### C. Selection Statistics

Use:
- `outputs/tables/accepted_rejected_sample_statistics.csv`
- `outputs/tables/dino_pruning_stats.csv`
- `outputs/tables/dino_pruning_per_class_stats.csv`

#### D. Qualitative Figures

Use:
- qualitative sample grids should be regenerated locally when needed

### VI. Discussion

- Explain whether selective filtering helps more than adding all synthetic images.
- Discuss whether CLIP/DINO/attribute coverage improve or merely regularize selection.
- If confidence-only is strongest, say so and redefine the contribution around the broader selective-admission idea rather than every individual module.
- Discuss failure cases and fragile classes using per-class tables.

### VII. Limitations

- Synthetic generation is AGA-inspired, not an exact reproduction.
- Attribute-aware coverage is based on parent-image proxy labels.
- Results depend on available compute and repeated runs.
- If DINO pruning removes little, state that honestly.

### VIII. Conclusion

- Keep conclusions proportional to evidence.
- Claim improvement only if supported by multi-seed summary tables.
- Claim improvement over the base paper only if a fair comparison file is completed and supports that statement.

## Non-Negotiable Honesty Rules

- Do not report one-seed or one-epoch debug runs as final evidence.
- Do not claim statistical significance unless you actually compute and report it.
- Do not call the generator "AGA" without the "inspired" qualifier unless you fully match the original method.
- Do not claim CLIP, DINO, or attribute coverage help if the ablation table does not support that claim.
