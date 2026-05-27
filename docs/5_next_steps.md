# 5. Next Steps

## 1. Current Position

The current workflow has a clear baseline and a credible champion:

| Stage | Best result |
| --- | ---: |
| Frozen ResNet50 transfer learning | 59.49% validation top-1 |
| Fine-tuned ResNet50 `layer3 + layer4` | 72.86% validation top-1 |
| Fine-tuned ResNet50 `layer3 + layer4` | 73.64% test top-1 |
| Fine-tuned ResNet50 `layer3 + layer4` | 91.18% test top-5 |

This is a strong baseline for the personal Food-101 project. The current
notebook should remain the baseline and evaluation notebook.

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

Notebook 2 now keeps ResNet50 and changes only the training recipe:

| Experiment | Change | Reason |
| --- | --- | --- |
| ResNet50 FT-V2 | longer fine-tuning with early stopping | tests whether the current run is undertrained |
| ResNet50 FT-V3 | learning-rate scheduler | stabilizes deeper fine-tuning |
| ResNet50 FT-V4 | stronger augmentation | targets presentation and lighting variation |

Recommended additions:

- `ReduceLROnPlateau` or cosine scheduling.
- Early stopping based on validation accuracy or validation loss.
- `RandomResizedCrop`, `ColorJitter`, and mild affine transforms.
- Optional label smoothing for noisy Food-101 labels.

## 4. Scope Expansion

After the evaluation layer is reliable, scale the project in three directions:

1. **Architecture comparison:** add EfficientNet-B0 or ConvNeXt-Tiny as a
   modern baseline and compare accuracy, parameter count, and inference time.
2. **Deployment readiness:** export the selected model, add deterministic
   single-image inference, and document expected Kaggle artifact paths.
3. **Error-driven improvement:** inspect confusion pairs for classes such as
   `steak`, `pork_chop`, `filet_mignon`, `ravioli`, and `chocolate_mousse`,
   then tune augmentation or sampling based on observed failure modes.

## 5. Recommended Next Task

The next implementation task should be:

> Run `02_resnet50_training_refinements.ipynb` using the uploaded baseline
> checkpoint artifact, then compare its final validation/test metrics against
> notebook 1.

This keeps the baseline notebook stable and makes any improvement easier to
attribute to the training recipe rather than evaluation changes.

Latest training signal: notebook 2 reached **77.90% validation top-1**, a
**+5.04 percentage point** gain over the notebook 1 baseline. The immediate
next step is to run notebook 2's final evaluation section against
`resnet50_ft_v2_best.pth` and record held-out test top-1/top-5.

If evaluating in a fresh Kaggle session, upload `resnet50_ft_v2_best.pth` as a
Kaggle Model artifact and point `CFG.REFINED_ARTIFACT_DIR` at that mounted
model directory. If evaluating in the same session that trained notebook 2, the
checkpoint under `/kaggle/working/results/resnet50_refinements` is enough.
