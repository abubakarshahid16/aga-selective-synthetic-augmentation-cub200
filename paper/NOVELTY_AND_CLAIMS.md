# Novelty And Claims

Use this file to keep the manuscript aligned with evidence and novelty.

## Exact Novelty Statement

The paper novelty should be phrased as:

**We propose a practical AGA-inspired selective synthetic augmentation framework for fine-grained bird classification that filters generated samples using class-consistency, semantic alignment, diversity control, and class-aware selection rather than treating all synthetic images as equally useful.**

## What Is Actually New Here

The new part is not merely:

- generating bird images
- using CLIP by itself
- using DINO by itself

The new part is the **combined selective-admission pipeline** and the empirical finding that **selection quality matters more than synthetic quantity**.

## Safe Claims

These claims are safe if supported by the main benchmark:

- Selected synthetic augmentation outperformed the real-only baseline under our setup.
- Selected synthetic augmentation outperformed naive all-synthetic augmentation under our setup.
- Practical filtering is important for fine-grained synthetic augmentation.
- Not every generated image is beneficial for downstream recognition.

## Claims That Require Stronger Evidence

These need multi-seed summary support:

- The method consistently improves mean accuracy.
- The method consistently improves macro-F1.
- The method is robust across seeds.

## Claims That Require Direct Comparison

These require `base_paper_comparison.csv` to be completed fairly:

- We beat the base paper.
- We outperform AGA on CUB-200-2011.
- Our method is superior to the WACV 2025 method.

## If The Full Method Is Not The Best Ablation

If confidence-only wins, the paper can still be novel and publishable by reframing the contribution as:

- a selective synthetic-admission framework
- an empirical analysis of which filtering components matter most
- a practical finding that simpler filtering may outperform more complex selection under the tested setup

## Best Paper Framing

Best framing for speed, novelty, and publishability:

**A practical AGA-inspired selective augmentation framework for fine-grained bird classification, evaluated as an empirical study on CUB-200-2011.**

That framing supports:

- novelty
- project alignment
- honest writing
- a realistic IEEE-style submission strategy
