# Multi-Class Food Recognition

<img src="https://www.meatdistrictco.com.au/wp-content/uploads/2024/08/0O2A0384-1700x660.jpg" alt="Food recognition project banner" width="100%">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-455A64?style=flat-square)
![Status](https://img.shields.io/badge/Status-Champion%20Model-2E7D32?style=flat-square)

**Food-101 image classification** project focused on building a reliable
baseline, improving it with controlled experiments, and translating model
performance into practical food-recognition decisions.

## 1. Project Overview

Food recognition is a **fine-grained computer vision** problem. Many dishes
share similar colors, ingredients, textures, and plating styles, so the model
needs to be evaluated beyond a single accuracy score.

This project classifies Food-101 images into **101 food categories**. The work
is structured around **reproducible notebooks** because the project is
exploratory: each notebook records the reasoning, code, metrics, errors, and
model artifacts for one experiment stage.

| Item | Value |
| --- | ---: |
| Dataset | Food-101 |
| Images | 101,000 |
| Classes | 101 |
| Images per class | 1,000 |
| Split strategy | Stratified train / validation / test |
| Framework | PyTorch |

## 2. Task And Goal

The task is **multi-class food image classification**: given one food image,
predict the correct class from **101 possible labels**.

The goal is to build a **defensible model** that can support practical
food-recognition workflows. That means the project tracks:

1. **Held-out test accuracy**, not validation accuracy only.
2. **Top-1 and top-5 accuracy** for both strict prediction and ranked suggestions.
3. **Model size, parameter count, and inference latency**.
4. **Hard classes, repeated confusion pairs, and qualitative failure examples**.
5. Whether a new experiment changes the decision, not only whether it adds
   complexity.

## 3. Key Metrics

The current champion is **ResNet50 FT-V2**, a refined ResNet50 model trained
with longer fine-tuning, AdamW, learning-rate scheduling, stronger
augmentation, and label smoothing.

| Metric | Champion result |
| --- | ---: |
| Validation top-1 accuracy | 77.90% |
| Validation top-5 accuracy | 92.36% |
| Test top-1 accuracy | 78.28% |
| Test top-5 accuracy | 92.65% |
| Test calibration ECE | 0.0265 |
| Auto-accept coverage | 58.02% |
| Auto-accept top-1 accuracy | 96.47% |
| Suggestion-band top-5 containment | 100.00% |
| Parameters | 24.7M |
| Model size | 94.48 MB |
| T4 latency | 5.35 ms/image |

## 4. Model Progress

| Stage | Result |
| --- | ---: |
| Frozen ResNet50 transfer learning | 59.49% validation top-1 |
| Baseline fine-tuned ResNet50 `layer3 + layer4` | 73.64% test top-1 |
| Refined ResNet50 FT-V2 | 78.28% test top-1 |
| Refined ResNet50 FT-V2 | 92.65% test top-5 |
| Calibrated decision layer | 58.02% auto-accept coverage at 96.47% top-1 |

The largest gain came from improving the **ResNet50 training recipe**. The
FT-V2 model improved held-out test top-1 by **4.63 percentage points** over the
first fine-tuned ResNet50 baseline.

## 5. Model Comparison

| Model | Stage | Test top-1 | Test top-5 | Parameters | Model size | T4 latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ResNet50 FT-V2 | current champion | 78.28% | 92.65% | 24.7M | 94.48 MB | 5.35 ms/image |
| ConvNeXt-Tiny | frozen-head challenger | 70.92% | 90.24% | 28.4M | 108.23 MB | 7.17 ms/image |
| EfficientNet-B0 | frozen-head challenger | 52.13% | 77.02% | 4.8M | 18.55 MB | 7.44 ms/image |

**ConvNeXt-Tiny** was the strongest modern-backbone challenger, but it was less
accurate, larger, and slower than **ResNet50 FT-V2**. **EfficientNet-B0** was
much smaller, but its accuracy was not competitive for the current target.

## 6. Technical Findings

- **ResNet50** remains the strongest model family tested so far.
- **Training-recipe refinement** delivered more value than switching to a modern
  backbone.
- **Top-5 accuracy above 92%** shows the model is much stronger as a ranked
  suggestion engine than as a single hard-label system.
- **Temperature scaling** improves confidence quality without changing the
  model ranking, reducing test ECE from 0.0432 to **0.0265**.
- **Decision thresholds** convert calibrated confidence into practical actions:
  auto-accept easy predictions, show suggestions for ambiguous classes, and
  reserve confirmation or review for risky cases.
- **Hard classes** cluster around visually similar foods: steak-like dishes,
  tartare or ceviche dishes, pastry-like desserts, and chocolate desserts.

## 7. Business Implications

- A food-recognition product should present **ranked suggestions** rather than
  only one label, because top-5 performance is materially stronger than top-1.
- The model is promising for **user-assisted tagging, menu enrichment, search**,
  or food diary workflows where the user can confirm one of several likely
  predictions.
- The hardest classes are realistic **business edge cases**: visually similar
  dishes may need user confirmation, richer metadata, or manual review.
- **Confidence scores should not be exposed directly without calibration**,
  because the model can be very confident on semantically plausible but wrong
  classes.
- The decision layer supports a more usable product experience: easy dishes can
  be accepted automatically, ambiguous correct cases can become ranked
  suggestions, hard classes can request confirmation, and known confusion pairs
  can be flagged for review.
- **ResNet50 FT-V2 is the best current trade-off**: the tested modern
  alternatives did not improve accuracy or efficiency enough to justify
  replacing it.

## 8. Experiment Workflow

| Notebook | Purpose |
| --- | --- |
| `01_food101_baseline_transfer_finetuning.ipynb` | Builds the baseline with data ingestion, transfer-learning comparison, ResNet50 fine-tuning, held-out test evaluation, confusion analysis, qualitative errors, and efficiency reporting. |
| `02_resnet50_training_refinements.ipynb` | Improves the selected ResNet50 checkpoint with longer fine-tuning, AdamW, LR scheduling, stronger augmentation, and label smoothing. |
| `03_modern_backbone_comparison.ipynb` | Compares EfficientNet-B0 and ConvNeXt-Tiny against ResNet50 FT-V2 using the same split, metrics, and artifact exports. |
| `04_resnet50_error_calibration_inference.ipynb` | Analyzes the champion with calibration metrics, hard-class reports, high-confidence errors, and deterministic single-image inference. |
| `05_confidence_decision_layer.ipynb` | Converts calibrated predictions into product actions: auto-accept, show suggestions, request confirmation, or review. |
| `06_food_recognition_demo_inference.ipynb` | Demonstrates the final single-image workflow with top-k predictions, calibrated confidence, decision action, and CSV demo exports. |

## 9. Product Direction: FoodLens

The next product direction is **FoodLens**, an image-first food-recognition
assistant. FoodLens should use the final model workflow to identify dishes from
uploaded images, show ranked predictions, and choose the right product action:
**auto-accept**, **suggest**, **confirm**, or **review**.

The recommended MVP is an **image upload app** backed by a lightweight inference
API. Video recognition should come later through frame sampling once the image
workflow is reliable.

The first static frontend concept is available at
[`app/frontend/index.html`](app/frontend/index.html). It uses mocked predictions
for now and will connect to a real inference API after the model artifacts are
placed outside git.

Detailed approach, result notes, and next steps are indexed in
[docs/README.md](docs/README.md) and maintained in
[docs/03_modeling_approach.md](docs/03_modeling_approach.md),
[docs/04_model_results.md](docs/04_model_results.md), and
[docs/05_next_steps.md](docs/05_next_steps.md). The FoodLens product concept
is documented in [docs/06_foodlens_app_concept.md](docs/06_foodlens_app_concept.md).

Banner image source:
[`meatdistrictco.com.au`](https://www.meatdistrictco.com.au/wp-content/uploads/2024/08/0O2A0384-1700x660.jpg)
