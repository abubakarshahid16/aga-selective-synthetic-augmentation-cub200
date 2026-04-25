# Colab GPU Run Guide

Use this guide to run the publication benchmark on Google Colab with a GPU instead of waiting for the local CPU run.

## What You Need

- A Google account with Colab access.
- This project folder available in either:
  - Google Drive, or
  - a GitHub repository you can clone from Colab.
- The CUB dataset and the prepared project artifacts already present in the project folder:
  - `data/CUB_200_2011/`
  - `data/prepared/`
  - `outputs/samples/synthetic_candidates/synthetic_manifest.csv`
  - `outputs/tables/selected_synthetic_manifest.csv`
  - `outputs/tables/synthetic_classifier_scores.csv`
  - `outputs/tables/synthetic_clip_scores.csv`
  - `outputs/tables/synthetic_dino_pruned.csv`

## Fastest Way to Link It

### Option 1: Upload the whole `GEN_AI` folder to Google Drive

Put this local folder into Drive:

- `C:\Users\abuba\OneDrive\Desktop\GEN_AI`

Recommended Drive destination:

- `MyDrive/GEN_AI`

That is the simplest path because the benchmark scripts already expect the full project structure.

### Option 2: Push the repo to GitHub and clone it in Colab

Only use this if you are comfortable re-copying the dataset and outputs into the cloned repo.

## In Colab

Open a new notebook and set:

- `Runtime` -> `Change runtime type`
- Hardware accelerator: `GPU`

## Colab Cells

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Go to the project folder

If you uploaded the whole folder to Drive:

```python
%cd /content/drive/MyDrive/GEN_AI/project
```

If the folder name differs, adjust the path accordingly.

### 3. Install dependencies

```bash
!bash scripts/colab_install.sh /content/drive/MyDrive/GEN_AI/project
```

### 4. Check GPU

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
```

### 5. Run the publication benchmark

```bash
!python src/run_publication_benchmark.py \
  --python python \
  --project-src src \
  --prepared-dir data/prepared \
  --image-root data/CUB_200_2011/images \
  --model-name resnet18 \
  --epochs 30 \
  --patience 7 \
  --batch-size 32 \
  --image-size 224 \
  --num-workers 2 \
  --seeds 42 43 44 \
  --resume
```

### 6. Run the ablations

```bash
!python src/ablation.py \
  --python python \
  --project-src src \
  --prepared-dir data/prepared \
  --image-root data/CUB_200_2011/images \
  --model-name resnet18 \
  --epochs 30 \
  --patience 7 \
  --batch-size 32 \
  --image-size 224 \
  --num-workers 2 \
  --confidence-threshold 0.02 \
  --clip-threshold 0.20 \
  --seeds 42 43 44 \
  --resume
```

### 7. Run the publication audit

```bash
!python src/publication_readiness.py
```

## Result Files to Look For

After the benchmark:

- `outputs/tables/publication_benchmark_runs.csv`
- `outputs/tables/publication_benchmark_summary.csv`
- `outputs/tables/publication_benchmark_config.json`

After the ablations:

- `outputs/tables/ablation_runs.csv`
- `outputs/tables/ablation_summary.csv`

After the audit:

- `outputs/tables/publication_readiness.json`
- `paper/PUBLICATION_READINESS_REPORT.md`

## Very Important Notes

- Colab sessions can disconnect. Because the scripts support `--resume`, rerunning the same command is safe.
- Keep outputs on Google Drive, not only in `/content`, or you may lose them when the session resets.
- Do not claim you beat the base paper unless you also fill:
  - `outputs/tables/base_paper_comparison.csv`

## Recommended Next Step After Colab Finishes

Download or sync back these files first:

- `outputs/tables/publication_benchmark_summary.csv`
- `outputs/tables/ablation_runs.csv`
- `outputs/tables/ablation_summary.csv`
- `outputs/tables/publication_readiness.json`
- `paper/PUBLICATION_READINESS_REPORT.md`
