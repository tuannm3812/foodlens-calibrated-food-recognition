# 1. Project Instructions

## 1. Objective

This project builds and evaluates calibrated food-recognition models. The
primary benchmark remains the **101-class Food-101 classifier**, and the
expanded-taxonomy track now tests whether the same modeling approach can cover
more than 100 food classes.

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
| Calibration and inference | `04_resnet50_error_calibration_inference.ipynb` | calibrate confidence and prepare deterministic inference |
| Decision layer | `05_confidence_decision_layer.ipynb` | convert calibrated confidence into product actions |
| Demo inference | `06_food_recognition_demo_inference.ipynb` | export demo predictions and app artifacts |
| Multi-food pipeline | `08_detection_to_foodlens_pipeline.ipynb` | classify detected regions with the FoodLens classifier |
| Accuracy phase | `11_food101_accuracy_phase1_a3b_convnext_tiny_continued.ipynb` plus archived `09`/`10` records | test controlled Food-101 accuracy improvements beyond ResNet50 FT-V2 |
| Taxonomy expansion | `12_food_taxonomy_expansion_audit.ipynb` | audit external food labels before training beyond 101 classes |
| Expanded baseline | `13_expanded_taxonomy_v1_baseline.ipynb` | train the first conservative 130-class classifier head |

Baseline, refinement, frozen-head comparison, detector exploration, and
superseded accuracy notebooks are preserved under `notebooks/archive/`.

The current product champion is **ResNet50 FT-V2**. The current accuracy leader
is **A3b ConvNeXt-Tiny**, pending decision-layer recalibration.

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

- ResNet50 FT-V2 remains the product champion because it has the strongest
  calibrated decision-layer evidence.
- ResNet50 FT-V2 improves held-out test top-1 to **78.28%**.
- Test top-5 accuracy reaches **92.65%**, supporting ranked suggestions.
- Temperature scaling improves test ECE from **0.0432** to **0.0265**.
- A3b ConvNeXt-Tiny is the current 101-class accuracy leader with **83.90%**
  test top-1 and **95.78%** test top-5.
- The first expanded-taxonomy baseline reaches **86.10%** test top-1 and
  **96.88%** test top-5 across 130 classes.
- The next useful step is decision-layer recalibration for A3b and controlled
  expanded-taxonomy fine-tuning, not another broad architecture search.
