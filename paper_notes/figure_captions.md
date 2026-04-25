# Draft Figure Captions

## Dataset Overview

Figure X. Overview of the CUB-200-2011 fine-grained bird dataset used in all experiments, showing representative classes, the official train/test split, and the role of bounding boxes and optional attributes in the pipeline.

## Baseline Results

Figure X. Training and validation curves for the real-only baseline classifier on CUB-200-2011. The model is trained only on official real images using the prepared train/validation split derived from the official training partition.

## Synthetic Generation Examples

Figure X. Example synthetic candidates produced by the AGA-inspired foreground-preserving augmentation pipeline. Bird foregrounds are retained while background context and mild appearance factors are varied to generate candidate training images.

## Accepted vs Rejected Samples

Figure X. Examples of accepted and rejected synthetic candidates after the proposed selection module. Accepted samples satisfy class-consistency and semantic checks while contributing more useful diversity and balanced coverage.

## Confusion Matrices

Figure X. Confusion matrices for the real-only baseline, real plus all synthetic data, and real plus selected synthetic data. These plots illustrate how selective augmentation changes fine-grained class confusions.

## Experiment Comparison

Figure X. Comparison of overall test accuracy across the three main experiments: real-only baseline, real plus all synthetic candidates, and real plus selected synthetic candidates using the proposed CLIP-DINO guided filtering module.

## Ablation Charts

Figure X. Ablation study results for the proposed selection module. Each variant adds one component of the pipeline, allowing analysis of the contribution of classifier confidence, CLIP consistency, DINO diversity pruning, class-balance control, and attribute-aware coverage.
