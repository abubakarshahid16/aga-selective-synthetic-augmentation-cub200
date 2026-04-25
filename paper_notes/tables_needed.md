# Tables Needed for the Report

- Dataset split summary: shows the number of images in train, validation, and test splits and confirms use of the official CUB split.
- Model configuration table: lists backbone, optimizer, learning rate, weight decay, image size, batch size, early stopping, and seed settings.
- Main experiment results table: compares real-only, real + all synthetic, and real + selected synthetic performance.
- Ablation results table: isolates the contribution of each selection-module component.
- Accepted vs rejected counts table: reports how many synthetic candidates were kept or filtered out overall and per class.
- DINO pruning statistics table: summarizes diversity pruning behavior and keep ratio.
- Per-class accuracy table: reports class-wise accuracy for the main experiments.
- Per-class gains/losses table: shows which bird classes improved or worsened under synthetic augmentation and selective filtering.
- Attribute-aware coverage table: explains the optional attribute-aware variant and how many classes/samples were affected.
- Limitations table: clearly distinguishes exact components, practical approximations, and manual or external dependencies if used.
