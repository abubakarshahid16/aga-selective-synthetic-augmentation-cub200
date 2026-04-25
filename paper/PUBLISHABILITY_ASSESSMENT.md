# Publishability Assessment

Last updated: April 25, 2026

## Executive Verdict

This project is **publishable in a scoped venue** if the manuscript is written as an honest empirical study and the claims are aligned to the currently saved evidence.

This project is **not yet ready for strong benchmark-style claims** under the repository's own publication standard because the multi-seed benchmark package is incomplete.

## Short Answer

- `Project requirements`: largely satisfied
- `Novelty`: satisfied
- `Reproducibility`: strong for a course or project paper
- `Evidence for a strong journal claim`: incomplete
- `Publishable right now`: yes, with narrow claims and a clear limitations section

## Best Publishable Paper Framing

Use the paper as:

**A practical AGA-inspired selective synthetic augmentation framework for fine-grained bird classification on CUB-200-2011: an empirical study under constrained compute.**

That framing is stronger and safer than presenting the work as:

- a definitive superiority claim over prior work
- an exact reproduction of AGA
- proof that every CLIP, DINO, balance, and attribute component improved performance

## What The Current Evidence Supports

The strongest saved evidence currently supports the following claims:

1. Selective synthetic augmentation outperformed the real-only baseline in the saved run.
2. Selective synthetic augmentation outperformed naive all-synthetic augmentation in the saved run.
3. The overall selective-admission idea is more defensible than a "more synthetic data is always better" story.
4. The repository contains a real, reproducible experimental pipeline with saved outputs, figures, logs, manifests, checkpoints, and result tables.

## What The Current Evidence Does Not Support

The current repository state does not support these claims yet:

1. Consistent improvement across multiple seeds
2. Statistical significance
3. Improvement over the base WACV 2025 paper
4. Strong evidence that CLIP, DINO, class-balance, and attribute-aware coverage each contributed positively

## Strongest Saved Evidence

### Main experiment results

From `outputs/tables/experiment_comparison.csv`:

- `exp1_real_only`: accuracy `0.1388`
- `exp2_real_plus_all_synthetic`: accuracy `0.1367`
- `exp3_real_plus_selected_synthetic`: accuracy `0.1505`

Interpretation:

- Adding all synthetic data did not help in the saved run.
- Adding selected synthetic data did help in the saved run.
- This is a publishable empirical story if written carefully.

### Ablation evidence

From `outputs/tables/ablation_summary.csv`:

- `ablation_confidence_only`: accuracy `0.1555`
- all more complex variants: accuracy `0.1505`

Interpretation:

- The selective-admission framework is supported.
- The current saved ablations do **not** support a strong claim that CLIP, DINO, balance, or attribute coverage improved over confidence-only selection.
- The paper should therefore emphasize the broader selective-filtering framework and empirical analysis rather than claiming every component helped.

### Selection statistics

From `outputs/tables/accepted_rejected_sample_statistics.csv`:

- candidates generated: `400`
- final selected: `15`
- selection rate: `3.75%`

Interpretation:

- The pipeline is highly selective.
- This directly supports the paper's central narrative that only a small fraction of generated images were retained as useful training additions.

## Requirement-by-Requirement Status

### Novelty

`PASS`

Why:

- The contribution is not generic image generation.
- The contribution is the selective synthetic-admission pipeline and the resulting empirical finding that selection quality matters more than synthetic quantity.

### Research workflow

`PASS`

Why:

- The repository includes baselines, an all-synthetic control, a selected-synthetic condition, ablations, qualitative outputs, checkpoints, logs, and reporting files.

### Reproducibility

`PASS`

Why:

- Commands, scripts, outputs, tables, plots, and checkpoints are organized clearly.
- The project can be audited and rerun.

### Evidence strength for publication

`PARTIAL`

Why:

- Single-run evidence is present and useful.
- Multi-seed publication benchmark outputs are still missing.
- `publication_benchmark_runs.csv`, `publication_benchmark_summary.csv`, `publication_benchmark_config.json`, and `ablation_runs.csv` are not yet present.

### Direct comparison to prior work

`NOT YET SUPPORTED`

Why:

- `outputs/tables/base_paper_comparison.csv` explicitly says a fair claim is not yet allowed.

## Required Writing Adjustments For Publishability

To maximize acceptance odds with the current evidence, the manuscript should do the following:

1. Use `AGA-inspired` wording throughout.
2. Present the paper as an empirical study under constrained compute.
3. Report the saved main results honestly as the current evidence base.
4. State clearly that broader multi-seed validation is future work.
5. Avoid claiming that the method beats the base paper.
6. Avoid claiming that every selection component improved the result.
7. Highlight that naive all-synthetic augmentation underperformed selective augmentation.

## Recommended Title

**A Practical AGA-Inspired Selective Synthetic Augmentation Framework for Fine-Grained Bird Classification on CUB-200-2011**

If you want the compute limitation visible in the title, use:

**A Practical AGA-Inspired Selective Synthetic Augmentation Framework for Fine-Grained Bird Classification: An Empirical Study Under Constrained Compute**

## Recommended Abstract Direction

Use an abstract built around these points:

1. Fine-grained classification is difficult and synthetic augmentation is attractive but noisy.
2. The paper studies selective acceptance of generated samples rather than blind use of all synthetic images.
3. The implemented pipeline uses classifier confidence, semantic filtering, diversity control, and class-aware selection.
4. In the saved experimental results, selected synthetic augmentation outperformed both the real-only baseline and naive all-synthetic augmentation.
5. The evaluation is limited by compute constraints and should be interpreted as preliminary but reproducible evidence.

## Required Limitations Section

The manuscript should include all of the following:

1. The generation pipeline is `AGA-inspired`, not an exact reproduction.
2. The current strongest evidence is based on saved single-run outputs rather than a completed multi-seed benchmark package.
3. Compute and environment constraints limited the scale of repeated experiments.
4. DINO pruning did not show a clear effect in the saved run.
5. The confidence threshold was low and should be justified or revisited in future work.

## Acceptance Risk

Acceptance is never guaranteed.

With the current evidence, the main risks are:

1. overclaiming beyond single-run evidence
2. using a title that promises more than the ablation supports
3. targeting a venue that expects stronger benchmarking

## Final Recommendation

This paper should be submitted only with **scoped, honest claims**.

Best current verdict:

- `Ready for project submission`
- `Conditionally ready for scoped paper submission`
- `Not ready for strong benchmark-paper claims`

The fastest path to a more credible submission is not to exaggerate the current evidence. It is to package the strongest existing evidence well, state the limitations clearly, and align the contribution around the selective-admission framework rather than full benchmark superiority.
