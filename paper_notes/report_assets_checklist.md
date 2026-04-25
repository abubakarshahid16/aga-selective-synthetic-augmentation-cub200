# Report Assets Checklist

Include the following in the final report and viva deck.

## Screenshots

- Folder structure of the project root after setup
- CUB dataset placement under `data/CUB_200_2011/`
- Example rows from `data/prepared/train.csv`
- Example rows from `outputs/tables/selected_synthetic_manifest.csv`
- Terminal screenshot of baseline training start with full command
- Terminal screenshot of synthetic generation run
- Terminal screenshot of CLIP scoring run
- Terminal screenshot of DINO pruning run
- Terminal screenshot of selected-dataset build run
- Terminal screenshot of final augmented training run

## Figures

- Dataset overview figure showing sample bird classes from CUB
- Train/validation loss curves for the real-only baseline
- Train/validation accuracy curves for the real-only baseline
- Synthetic generation examples from the AGA-inspired pipeline
- Accepted versus rejected synthetic sample grid
- Class-distribution chart before and after selection
- Confusion matrix for real-only baseline
- Confusion matrix for real + all synthetic
- Confusion matrix for real + selected synthetic
- Experiment comparison chart across the three main experiments
- Ablation comparison chart across selection variants

## Tables

- Dataset split summary
- Baseline metrics table
- Real + all synthetic metrics table
- Real + selected synthetic metrics table
- Ablation summary table
- Accepted versus rejected sample counts
- Per-class accuracy table for each main experiment
- Per-class gains/losses table comparing experiments
- Diversity and pruning statistics table

## Appendix Items

- Key CLI commands used for all experiments
- Important implementation limitations and approximation notes
- Description of the attribute-aware coverage approximation
