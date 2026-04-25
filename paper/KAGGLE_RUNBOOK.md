# Kaggle Runbook

## Reality Check

- No code can honestly guarantee that you will beat another paper in 12-13 hours.
- This repo is now optimized to maximize the chance of a strong run on Kaggle while avoiding common restart and OOM failures.
- If a Kaggle session stops, rerun the same command. The training scripts now save latest checkpoints every epoch and resume from them.

## Recommended Kaggle Hardware

- Best choice: `2x T4` if available in your environment.
- Safe choice: `1x T4` or `1x P100`.
- Keep internet enabled if CLIP or DINO weights need to download.

## Recommended Main Command

```bash
python src/run_kaggle_best_pipeline.py --model-name convnext_tiny --epochs 35 --patience 8 --batch-size 16 --grad-accum-steps 2 --num-workers 2 --image-size 224 --lr 2.5e-4 --weight-decay 5e-5 --seed 42
```

## Why These Defaults

- `convnext_tiny`: strongest backbone available in the repo without making Kaggle stability too risky.
- `batch-size 16` with `grad-accum-steps 2`: safer on Kaggle memory while keeping an effective batch of 32.
- `use_amp`: faster and usually stable on Kaggle GPUs.
- `eval_tta`: helps final evaluation slightly without changing training.
- `RandAugment + RandomErasing`: stronger regularization for better generalization.
- `resume`: every training command resumes from the latest checkpoint.
- `skip_oom_batches`: avoids total job failure from rare memory spikes.
- `warmup + curriculum` for selected synthetic training: safer optimization path than throwing all selected synthetic data in at once.

## Important Output Proofs

After the run, collect these:

- `outputs/tables/kaggle_best_pipeline_proof_bundle.json`
- `outputs/tables/kaggle_exp1_real_only_metrics.json`
- `outputs/tables/kaggle_exp2_real_plus_all_synthetic_metrics.json`
- `outputs/tables/kaggle_exp3_real_plus_selected_synthetic_metrics.json`
- `outputs/tables/kaggle_exp3_real_plus_selected_synthetic_kaggle_eval_evaluation_metrics.json`
- `outputs/tables/accepted_rejected_sample_statistics.csv`
- `outputs/tables/dino_pruning_stats.csv`
- `outputs/tables/selected_synthetic_manifest.csv`
- `outputs/plots/experiment_comparison_accuracy.png`

## If Kaggle Times Out

Run the exact same pipeline command again. It will:

- skip completed non-training steps if output files already exist
- resume unfinished training stages from `*_latest.pt`
- preserve previously generated tables and figures

## If You Still Hit OOM

Try this safer command:

```bash
python src/run_kaggle_best_pipeline.py --model-name convnext_tiny --epochs 35 --patience 8 --batch-size 8 --grad-accum-steps 4 --num-workers 2 --image-size 224 --lr 2.5e-4 --weight-decay 5e-5 --seed 42
```

## Publishing Guidance

- Report the method as "AGA-inspired" unless you replace the generation step with the exact original method.
- Use the saved proof bundle and metrics JSON files as the source of truth.
- Do not claim superiority over prior work unless your final saved numbers really show it.
