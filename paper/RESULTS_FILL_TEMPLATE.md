# Results Fill Template

Use this file while drafting the paper. Replace every bracketed field after running the publication workflow.

## Main Results

Backbone: `[model_name]`

Number of seeds: `[n]`

Real-only baseline:
- Accuracy: `[mean] +- [std]`
- Macro-F1: `[mean] +- [std]`

Real + all synthetic:
- Accuracy: `[mean] +- [std]`
- Macro-F1: `[mean] +- [std]`

Real + selected synthetic:
- Accuracy: `[mean] +- [std]`
- Macro-F1: `[mean] +- [std]`

Best method by accuracy: `[experiment_name]`

## Ablation Results

Confidence only: `[mean] +- [std]`

Confidence + CLIP: `[mean] +- [std]`

Confidence + CLIP + DINO: `[mean] +- [std]`

Confidence + CLIP + DINO + balance: `[mean] +- [std]`

Confidence + CLIP + DINO + balance + attribute coverage: `[mean] +- [std]`

## Selection Statistics

Candidates generated: `[num_candidates]`

Candidates kept after final selection: `[num_selected]`

Selection rate: `[selection_rate]`

DINO keep ratio: `[keep_ratio]`

## Honest Conclusion Template

The proposed selective augmentation pipeline `[did / did not]` improve mean top-1 accuracy over the real-only baseline on CUB-200-2011. The strongest evidence came from `[experiment name]`, while ablation results `[supported / did not support]` the additional contribution of CLIP, DINO, and attribute-aware coverage. Because the generation stage is AGA-inspired rather than an exact reproduction, claims are limited to the implemented pipeline evaluated in this repository.
