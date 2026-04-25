# Abstract And Claims For Current Evidence

## Recommended Title

**A Practical AGA-Inspired Selective Synthetic Augmentation Framework for Fine-Grained Bird Classification on CUB-200-2011**

## Manuscript-Safe Abstract

Fine-grained image classification is challenging because visually similar classes require discriminative training evidence, while naively adding synthetic data can introduce harmful noise. This paper studies a practical AGA-inspired synthetic augmentation pipeline for CUB-200-2011 in which generated bird images are not accepted blindly but filtered through a selective sample-admission process. The proposed framework combines class-consistency checking, semantic filtering, diversity control, and class-aware selection to retain only a small subset of candidate synthetic images for downstream training. In the saved experimental results, selected synthetic augmentation outperformed both a real-only baseline and a naive real-plus-all-synthetic setting, supporting the empirical claim that synthetic sample quality matters more than synthetic quantity in this setup. Additional ablation results support the broader selective-admission idea, while also showing that more complex filtering stages did not outperform confidence-only selection in the current saved run. Due to compute and environment constraints, the present evaluation should be interpreted as preliminary but reproducible evidence rather than a fully exhaustive benchmark.

## Safe Contribution List

1. A practical AGA-inspired synthetic augmentation workflow for fine-grained bird classification
2. A selective synthetic sample-admission pipeline instead of blind acceptance of all generated images
3. An empirical analysis showing that selected synthetic augmentation can outperform naive all-synthetic augmentation in the saved setup
4. A reproducible repository with outputs, manifests, plots, checkpoints, and reporting artifacts

## Safe Claims To Use

1. Selective synthetic augmentation outperformed both internal controls in the saved run.
2. Not all generated samples were beneficial for downstream recognition.
3. Synthetic selection quality mattered more than synthetic quantity in the saved experiment.
4. The repository presents a practical and reproducible framework for studying selective augmentation on CUB-200-2011.

## Claims To Avoid

1. The method consistently improves across seeds.
2. The method is statistically significant.
3. The method outperforms the base WACV 2025 paper.
4. CLIP, DINO, class-balance, and attribute coverage were each proven necessary.

## Recommended Limitations Paragraph

The current study has several limitations. First, the generation stage is AGA-inspired rather than an exact reproduction of the original method. Second, the strongest saved quantitative evidence currently comes from single-run outputs, while a complete multi-seed benchmark package remains future work. Third, the experimental scale was constrained by available compute and execution environment limits. Finally, the current ablations did not demonstrate a clear gain from all added filtering stages, and DINO pruning showed limited effect under the saved threshold settings.
