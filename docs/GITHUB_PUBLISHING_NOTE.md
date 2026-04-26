# GitHub Publishing Note

This repository has been prepared to be shared publicly as a **project and research artifact**, not as a raw dump of every training byproduct.

## Included Public Proof

The public repo intentionally keeps:

- the main project source code in `src/`
- the IEEE-style paper in `paper/`
- paper figures in `paper/fig_*.png`
- key saved output plots in `outputs/plots/`
- key qualitative sample grids in `outputs/samples/`
- key confusion matrices in `outputs/confusion_matrices/`
- key summary tables in `outputs/tables/`
- quick supplementary proof in `quick_results/`:
  - `training_log.csv`
  - `fig_loss_curves.png`
  - `fig_accuracy_curves.png`
- GitHub-ready screenshots in `docs/screenshots/`

## Intentionally Excluded

The public repo excludes bulky or redundant local artifacts such as:

- dataset image files
- local model folders
- quick-run checkpoints
- temporary screenshot staging folders
- local packaging folders created only for submission export

## Honest Presentation

The strongest honest public framing is:

> This repository presents a practical AGA-inspired selective synthetic augmentation framework for fine-grained bird classification on CUB-200-2011. In the saved repository run, selectively admitted synthetic samples outperformed both a real-only baseline and a naive all-synthetic control.

## Important Scope Note

The main paper results come from the saved repository run.  
The `quick_results/` files are supplementary constrained-compute proof artifacts used to demonstrate a real logged training run and curve generation.
