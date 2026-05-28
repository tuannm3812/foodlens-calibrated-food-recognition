# 5. Next Steps

## 1. Current Position

The current workflow has a clear baseline and a credible champion:

| Stage | Best result |
| --- | ---: |
| Frozen ResNet50 transfer learning | 59.49% validation top-1 |
| Fine-tuned ResNet50 `layer3 + layer4` | 72.86% validation top-1 |
| Baseline fine-tuned ResNet50 `layer3 + layer4` | 73.64% test top-1 |
| Refined ResNet50 FT-V2 | 78.28% test top-1 |
| Refined ResNet50 FT-V2 | 92.65% test top-5 |

Notebook 1 should remain the baseline and evaluation reference. Notebook 2 is
now the current ResNet50 champion because it improves held-out test top-1 by
4.63 percentage points without increasing model size or parameter count.

## 2. Baseline Notebook Refinements

The baseline notebook now includes the evaluation layer:

1. Final held-out test evaluation for the selected checkpoint.
2. Top-1 and top-5 accuracy.
3. Normalized confusion matrix focused on the hardest classes.
4. Model histories, predictions, metrics, and per-class reports exported to
   CSV.
5. Qualitative error-analysis panel for the final model.
6. Model size and single-image inference latency.
7. Artifact-backed inference mode for faster reruns.

These changes make the result easier to defend because they show whether the
72.86% validation result transfers to unseen test images and which classes
still need targeted attention.

## 3. Model Improvement Plan

Notebook 2 has validated that the training recipe matters. It keeps ResNet50
fixed and improves the recipe:

| Experiment | Change | Reason |
| --- | --- | --- |
| ResNet50 FT-V2 | longer fine-tuning, AdamW, scheduler, augmentation, label smoothing | new champion |
| ResNet50 FT-V3 | calibration and threshold analysis | reduces overconfident wrong predictions |
| ResNet50 FT-V4 | targeted augmentation for hard class clusters | tests error-driven improvement |

Recommended additions:

- `ReduceLROnPlateau` or cosine scheduling.
- Early stopping based on validation accuracy or validation loss.
- `RandomResizedCrop`, `ColorJitter`, and mild affine transforms.
- Optional label smoothing for noisy Food-101 labels.

## 4. Architecture Comparison Outcome

Notebook 3 has now tested EfficientNet-B0 and ConvNeXt-Tiny as frozen-head
modern-backbone challengers. ResNet50 FT-V2 remains the champion:

- ConvNeXt-Tiny reached **70.92% test top-1** and **90.24% test top-5**, but it
  is larger, slower, and 7.36 percentage points below ResNet50 FT-V2.
- EfficientNet-B0 reached **52.13% test top-1** and **77.02% test top-5**. Its
  small size is useful for deployment exploration, but accuracy is currently
  too low to promote.
- The current evidence does not justify replacing ResNet50 FT-V2.

## 5. Scope Expansion

Scale the project in three directions:

1. **Error-driven improvement:** inspect confusion pairs for classes such as
   `steak`, `pork_chop`, `filet_mignon`, `ravioli`, and `chocolate_mousse`,
   then tune augmentation or sampling based on observed failure modes.
2. **Confidence calibration:** add temperature scaling or calibration metrics
   so high-confidence errors are easier to interpret in an application.
3. **Deployment readiness:** export the selected model, add deterministic
   single-image inference, and document expected Kaggle artifact paths.

## 6. Recommended Next Task

The next implementation task should be:

> Build the next notebook around ResNet50 FT-V2 error analysis and calibration
> rather than another broad architecture search.

This keeps the current champion stable and focuses effort on the real failure
modes seen across notebooks 1 and 2: visually similar classes and
overconfident wrong predictions.

A new architecture should be promoted only if it improves accuracy
meaningfully, improves inference efficiency, or reduces the same hard-class
confusion patterns.

Keep `resnet50_ft_v2_best.pth` as the current champion artifact and use its
Kaggle Model path as the reference input for future comparison notebooks.
