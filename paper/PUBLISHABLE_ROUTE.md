# Publishable Route

This file defines the fastest defensible path from the current repository to a publishable paper.

## Primary Goal

Produce a paper that is:

- novel enough to count as research
- aligned with the project requirements
- honest about evidence
- realistic for IEEE-style submission

## Recommended Paper Positioning

Use this positioning unless the final experiments provide stronger evidence:

**A practical AGA-inspired selective synthetic augmentation framework for fine-grained bird classification on CUB-200-2011.**

That framing is stronger and safer than claiming:

- exact reproduction of AGA
- guaranteed improvement over the WACV 2025 paper
- universal benefit from every CLIP/DINO/attribute component

## Core Novelty Claim

The main novelty should be stated as:

**Not all generated images are equally useful for fine-grained recognition, and selective acceptance based on class consistency, semantic alignment, diversity control, and class-aware filtering can outperform naive synthetic augmentation.**

This novelty is realistic because the repository already implements:

- classifier-confidence filtering
- CLIP semantic scoring
- DINO diversity pruning
- class-balance filtering
- optional attribute-aware coverage

## Fastest Defensible Experimental Package

The minimum experiment set for a credible paper is:

1. Real-only baseline
2. Real + all synthetic candidates
3. Real + selected synthetic candidates
4. One compact ablation table

If compute is limited, the paper can still be written as a preliminary empirical study, but the final wording must reflect that.

## Claim Tiers

### Tier 1: Safest claim

Allowed when the selected method beats both internal controls:

**Selective synthetic-data filtering improved performance over both the real-only baseline and naive all-synthetic augmentation under our experimental setup.**

### Tier 2: Stronger claim

Allowed when multi-seed summary files exist and the effect is consistent:

**The proposed selective augmentation framework improved mean top-1 accuracy and macro-F1 across multiple seeds on CUB-200-2011.**

### Tier 3: Hardest claim

Allowed only after fair direct comparison to the base paper:

**Our method outperformed the base paper under a fair like-for-like comparison.**

Do not use Tier 3 unless `outputs/tables/base_paper_comparison.csv` supports it.

## Venue Strategy

### Fastest IEEE-branded route

- IEEE Access

Why:

- continuous submission
- broad scope
- more realistic for an applied empirical paper than a top benchmark-heavy conference

### Also realistic

- workshop paper
- student research venue
- smaller applied AI / pattern-recognition venue

## If Results Are Mixed

If the full CLIP+DINO+attribute version is not the best ablation:

- keep the framework contribution
- narrow the claim
- describe the stronger result honestly, e.g. confidence-only filtering was most effective in the tested setup

That still leaves a publishable empirical story:

- synthetic selection matters
- naive augmentation is not always best
- some filtering components help more than others

## If Results Are Strong

If the main benchmark and ablations are favorable, write the paper around:

- practical AGA-inspired generation
- selective sample admission
- internal evidence that filtering beats naive augmentation
- compute-aware reproducibility

## If Results Do Not Beat the Base Paper

The paper can still be publishable if it claims:

- a practical, reproducible selective augmentation pipeline
- preliminary or empirical evidence under the implemented setup
- a contribution in methodology and analysis rather than benchmark supremacy

## Non-Negotiable Rule

Do not make **"beats the base paper"** the only success condition.

The fastest route to a publishable paper is:

- a clear novelty statement
- a reproducible method
- honest experiments
- scoped claims that match the evidence
