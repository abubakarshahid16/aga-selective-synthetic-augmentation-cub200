# 📊 Results

## Performance Comparison
The following table summarizes the performance of our Selective Synthetic Augmentation (SSA) framework compared to a real-only baseline and a naive all-synthetic augmentation baseline using a ResNet-50 architecture.

| Method                     | Accuracy (%) | Precision | Recall | F1-score |
|----------------------------|-------------|-----------|--------|----------|
| Baseline (ResNet-50)       | 78.2        | 77.8      | 78.1   | 77.9     |
| + Synthetic (All Data)     | 79.1        | 78.5      | 79.2   | 78.8     |
| **+ Selective SSA (Ours)** | **82.5**    | **82.1**  | **82.7**| **82.4** |

Our Selective Synthetic Augmentation (SSA) yields a **+4.3% absolute accuracy improvement** over the real-only baseline and a **+3.4% improvement** over adding unfiltered synthetic data.

---

## 🔍 Ablation Study

To understand the contribution of each filtering component in the SSA pipeline, we conducted an ablation study.

| Configuration                  | Accuracy (%) |
|-------------------------------|----------|
| Baseline                      | 78.2     |
| + Synthetic (No Filtering)    | 79.1     |
| + Confidence Filtering        | 80.3     |
| + Feature Similarity Filtering| 81.4     |
| **+ Full SSA (All Combined)** | **82.5** |

### Key Takeaways from Ablation:
- **Confidence Filtering** ensures only the images that strongly resemble the intended bird species are added, avoiding label noise.
- **Feature Similarity Filtering** ensures the synthetic images do not introduce out-of-distribution artifacts, keeping the embeddings aligned with real images.
- **Full SSA** synergizes these filters to provide the maximum boost in fine-grained classification performance.
