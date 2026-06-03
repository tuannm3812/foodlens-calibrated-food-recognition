# 1. Project Instructions

## 1. Objective

This project builds and evaluates a **101-class Food-101 image classifier**.
The objective is to move from baseline model comparison to a champion model
that can support practical food-recognition workflows.

The project tracks four questions:

1. Which pretrained backbone gives the strongest baseline?
2. How much does ResNet50 improve with deeper fine-tuning and a better recipe?
3. Do modern compact backbones beat the refined ResNet50 champion?
4. Can calibrated confidence support product-style prediction decisions?

## 2. Dataset

Dataset: **Food-101**.

| Property | Value |
| --- | ---: |
| Images | 101,000 |
| Classes | 101 |
| Images per class | 1,000 |
| Image type | RGB food photographs |
| Split | Stratified train / validation / test |

Food-101 includes noisy real-world food photography, varied lighting,
presentation differences, and label ambiguity. This makes it a useful
fine-grained classification benchmark rather than a clean toy dataset.

## 3. Modeling Scope

The project is organized into controlled notebook stages:

| Stage | Notebook | Purpose |
| --- | --- | --- |
| Baseline | `01_food101_baseline_transfer_finetuning.ipynb` | compare frozen pretrained backbones and fine-tune ResNet50 |
| Refinement | `02_resnet50_training_refinements.ipynb` | improve ResNet50 with a stronger training recipe |
| Backbone comparison | `03_modern_backbone_comparison.ipynb` | test EfficientNet-B0 and ConvNeXt-Tiny against the champion |
| Calibration and inference | `04_resnet50_error_calibration_inference.ipynb` | calibrate confidence and prepare deterministic inference |
| Accuracy phase | `09` to `11` notebooks | test controlled Food-101 accuracy improvements beyond ResNet50 FT-V2 |

The current product champion is **ResNet50 FT-V2**. The current accuracy leader
is **A3 ConvNeXt-Tiny**, pending decision-layer recalibration.

## 4. Evaluation Contract

Every major notebook should preserve the same evaluation contract:

- stratified train, validation, and test splits;
- top-1 and top-5 accuracy;
- per-class metrics and hard-class reporting;
- repeated confusion-pair analysis;
- qualitative high-confidence error examples where relevant;
- parameter count, model size, and inference latency;
- exported CSV artifacts for downstream analysis.

This keeps each experiment comparable and prevents changes in reporting from
being mistaken for model improvement.

## 5. Artifact Policy

Generated artifacts should not be committed to git:

- Food-101 images or archives;
- Kaggle working directories;
- `.pth` model checkpoints;
- generated prediction CSVs;
- generated figures;
- notebook checkpoints and cache folders.

Only lightweight notebooks, documentation, and project metadata belong in this
repository.

## 6. Current Direction

The project has already shown that:

- ResNet50 is the strongest model family tested so far.
- ResNet50 FT-V2 improves held-out test top-1 to **78.28%**.
- Test top-5 accuracy reaches **92.65%**, supporting ranked suggestions.
- Temperature scaling improves test ECE from **0.0432** to **0.0265**.
- The next useful step is a confidence-based decision layer, not another broad
  architecture search.
