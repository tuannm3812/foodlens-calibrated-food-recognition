# 3. Modeling Approach And Notebook Reasoning Flow

## 1. Purpose

This document explains the **logic behind the notebook sequence**. The project
is not organized as isolated experiments; each notebook answers one modeling
question and passes a clearer decision to the next stage.

The core reasoning is:

1. Establish a stable **Food-101 baseline**.
2. Improve the strongest baseline with **controlled fine-tuning**.
3. Test whether a **modern backbone** justifies replacing the refined champion.
4. Use **error analysis and calibration** to make the champion easier to trust.
5. Convert calibrated predictions into **product decision bands**.
6. Run controlled accuracy-improvement experiments against the champion.

## 2. Why Notebook-First

This project uses a **notebook-first workflow** because the work is
experimental and evidence-driven. For this type of computer vision project,
notebooks make the reasoning easier to audit:

- **Dataset checks, visual inspection, model code, metrics, plots**, and written
  interpretation stay together.
- Each run can show exactly which **configuration, checkpoint, and result** led
  to the next decision.
- Kaggle provides the required **GPU runtime** without turning the repository
  into an infrastructure project.
- Markdown cells document **assumptions and conclusions** immediately beside
  the code that produced them.

The standard is still professional: **imports are centralized**,
**configuration is explicit**, constants use uppercase names, outputs are
cleared when notebook code changes, and model artifacts are kept outside git.

## 3. Notebook 1: Build The Baseline

Notebook:
[`../notebooks/01_food101_baseline_transfer_finetuning.ipynb`](../notebooks/01_food101_baseline_transfer_finetuning.ipynb)

Question:

> What is a reliable baseline for Food-101, and which pretrained backbone
> should be fine-tuned first?

Key points:

1. Load the Food-101 image folders and create a reproducible manifest.
2. Validate class balance and image availability.
3. Create stratified train, validation, and test splits.
4. Compare pretrained GoogLeNet, ResNet50, and MobileNetV3 as frozen feature
   extractors.
5. Select the strongest frozen-backbone candidate.
6. Fine-tune ResNet50 with controlled unfreezing strategies.
7. Evaluate the selected checkpoint on validation and held-out test data.
8. Export histories, predictions, per-class reports, confusion pairs,
   qualitative errors, and efficiency metrics.

Decision produced:

| Candidate | Outcome |
| --- | ---: |
| Frozen ResNet50 | 59.49% validation top-1 |
| Fine-tuned ResNet50 `layer3 + layer4` | 72.86% validation top-1 |
| Fine-tuned ResNet50 `layer3 + layer4` | 73.64% test top-1 |

Notebook 1 establishes **ResNet50 as the baseline family** and proves that
**domain-specific fine-tuning is necessary**.

## 4. Notebook 2: Improve The Training Recipe

Notebook:
[`../notebooks/02_resnet50_training_refinements.ipynb`](../notebooks/02_resnet50_training_refinements.ipynb)

Question:

> Can the selected ResNet50 improve through a better training recipe without
> changing the architecture?

Key points:

1. Start from the best baseline ResNet50 checkpoint.
2. Keep the architecture fixed so gains can be attributed to training changes.
3. Add longer fine-tuning, AdamW, scheduler-based learning-rate control,
   stronger augmentation, and label smoothing.
4. Evaluate the refined checkpoint with the same validation/test protocol used
   in Notebook 1.
5. Compare top-1, top-5, hard classes, confusion pairs, model size, and latency
   against the baseline.

Decision produced:

| Model | Test top-1 | Test top-5 | Change vs baseline |
| --- | ---: | ---: | ---: |
| Baseline fine-tuned ResNet50 | 73.64% | 91.18% | reference |
| ResNet50 FT-V2 | 78.28% | 92.65% | +4.63 pp top-1 |

Notebook 2 becomes the **current champion** because it improves generalization
without increasing model size or changing the architecture.

## 5. Notebook 3: Test Modern Backbones

Notebook:
[`../notebooks/03_modern_backbone_comparison.ipynb`](../notebooks/03_modern_backbone_comparison.ipynb)

Question:

> Do EfficientNet-B0 or ConvNeXt-Tiny offer a better accuracy, size, and
> latency trade-off than ResNet50 FT-V2?

Key points:

1. Keep the same Food-101 split and evaluation contract.
2. Train frozen-head challengers for EfficientNet-B0 and ConvNeXt-Tiny.
3. Export the same comparison artifacts used by the earlier notebooks.
4. Compare top-1, top-5, parameter count, model size, and latency against the
   ResNet50 FT-V2 reference.
5. Promote a challenger only if it improves accuracy meaningfully or gives a
   practical efficiency advantage.

Decision produced:

| Model | Test top-1 | Test top-5 | Decision |
| --- | ---: | ---: | --- |
| ResNet50 FT-V2 | 78.28% | 92.65% | keep as champion |
| ConvNeXt-Tiny | 70.92% | 90.24% | do not promote |
| EfficientNet-B0 | 52.13% | 77.02% | do not promote |

Notebook 3 confirms that the current evidence supports **improving ResNet50
FT-V2** rather than replacing it with the tested backbones.

## 6. Notebook 4: Calibrate And Operationalize The Champion

Notebook:
[`../notebooks/04_resnet50_error_calibration_inference.ipynb`](../notebooks/04_resnet50_error_calibration_inference.ipynb)

Question:

> Can ResNet50 FT-V2 become easier to trust in a practical food-recognition
> workflow?

Key points:

1. Load the **ResNet50 FT-V2** checkpoint from the champion artifact.
2. Reuse the same Food-101 validation/test split.
3. Collect logits once for reproducible calibration and error analysis.
4. Fit temperature scaling on validation logits.
5. Compare calibration quality before and after temperature scaling.
6. Export hard classes, confusion pairs, high-confidence errors, and
   calibrated prediction tables.
7. Provide deterministic single-image inference for deployment-style checks.
8. Bundle calibration artifacts into one zip file for downstream notebooks.

Decision expected:

| Area | Decision signal |
| --- | --- |
| Calibration | lower expected calibration error after temperature scaling |
| Error analysis | clear hard-class clusters for targeted review |
| Inference | stable top-k output for one image at a time |

Notebook 4 moves the project from model comparison toward **trust, usability,
and deployment readiness**.

## 7. Notebook 5: Define Product Decision Bands

Notebook:
[`../notebooks/05_confidence_decision_layer.ipynb`](../notebooks/05_confidence_decision_layer.ipynb)

Question:

> How should the product act on each calibrated prediction?

Key points:

1. Load calibrated prediction outputs from Notebook 4.
   Inputs can come from the linked Notebook output directory or from the
   `resnet50_error_calibration_artifacts.zip` bundle.
2. Engineer decision features such as top-1 confidence, top-1/top-2 margin,
   hard-class flags, and frequent confusion-pair flags.
3. Search interpretable thresholds for auto-accept, suggest, confirm, and
   review bands.
4. Report coverage and accuracy for each decision band.
5. Export a decision policy and representative examples.
6. Provide decision-band examples for the final demo notebook.

Decision expected:

| Band | Meaning |
| --- | --- |
| Auto-accept | prediction is confident enough to accept directly |
| Suggest | top-k ranking should be shown to the user |
| Confirm | user confirmation is needed before accepting |
| Review | known hard/confusing pattern should be inspected carefully |

Notebook 5 turns calibrated model output into **product behavior**.

## 8. Notebook 6: Demonstrate Final Inference

Notebook:
[`../notebooks/06_food_recognition_demo_inference.ipynb`](../notebooks/06_food_recognition_demo_inference.ipynb)

Question:

> What does the final food-recognition experience look like for one image?

Key points:

1. Load the ResNet50 FT-V2 checkpoint.
2. Load calibrated temperature and decision thresholds.
3. Load hard-class and confusion-pair context when available.
4. Predict calibrated top-k classes for one image.
5. Return a decision band and recommended user-facing action.
6. Export demo prediction and decision-summary CSV files.

Decision expected:

| Output | Meaning |
| --- | --- |
| Top-k predictions | ranked class suggestions with calibrated confidence |
| Decision band | auto-accept, suggest, confirm, or review |
| Recommended action | user-facing behavior for the prediction |
| Demo CSVs | auditable output for documentation or presentation |

Notebook 6 is the demo layer for the project.

## 9. Notebook 7: Explore Multi-Food Detection

Notebook:
[`../notebooks/07_multi_food_detection_exploration.ipynb`](../notebooks/07_multi_food_detection_exploration.ipynb)

Question:

> Can a pretrained detector propose useful food regions for images or sampled
> video frames?

Key points:

1. Load a pretrained detector such as YOLO.
2. Process one image or sampled video frames.
3. Export bounding boxes, detector labels, confidence scores, crops, and
   annotated figures.
4. Review whether the detected regions are usable enough for crop
   classification.

Notebook 7 is an exploration layer. It does not replace the FoodLens classifier.

## 10. Notebook 8: Classify Detected Crops

Notebook:
[`../notebooks/08_detection_to_foodlens_pipeline.ipynb`](../notebooks/08_detection_to_foodlens_pipeline.ipynb)

Question:

> Can detected regions be classified with the existing ResNet50 FT-V2 Food-101
> model?

Key points:

1. Load detections and crops from Notebook 7.
2. Load the FoodLens ResNet50 FT-V2 classifier and class names.
3. Predict top-k Food-101 labels for each crop.
4. Apply simple decision bands per detected region.
5. Export `multi_food_predictions.csv` for later API and frontend integration.

Notebook 8 is the bridge from detection exploration to a multi-food FoodLens
workflow.

## 11. Notebook 9: Accuracy Phase 1 A1

Notebook:
[`../notebooks/09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb`](../notebooks/09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb)

Question:

> Does continuing from ResNet50 FT-V2 with full-backbone fine-tuning improve
> Food-101 accuracy?

Key points:

1. Load the ResNet50 FT-V2 champion checkpoint.
2. Unfreeze all ResNet50 parameters.
3. Fine-tune with a lower backbone learning rate and a higher classifier-head
   learning rate.
4. Reuse the project-standard Food-101 split and evaluation contract.
5. Fit temperature scaling on validation logits.
6. Compare test top-1, top-5, calibrated ECE, latency, and model size against
   the champion.

Decision produced:

| Model | Test top-1 | Test top-5 | Test ECE |
| --- | ---: | ---: | ---: |
| ResNet50 FT-V2 champion | 78.28% | 92.65% | 0.0265 |
| A1 ResNet50 FT-V3 full backbone | 78.23% | 92.45% | 0.0229 |

Notebook 9 does **not** promote a new model. It rules out a simple
full-backbone ResNet50 continuation at 224px and points the next accuracy
experiment toward `A3`: fully fine-tuned ConvNeXt-Tiny.

## 12. Cross-Notebook Evaluation Contract

All notebooks should preserve the same **comparison contract**:

| Area | Standard |
| --- | --- |
| Data split | stratified train / validation / test |
| Accuracy metrics | top-1 and top-5 |
| Error analysis | hardest classes, confusion pairs, qualitative examples |
| Efficiency | parameter count, model size, single-image latency |
| Artifacts | checkpoints, histories, predictions, class reports |
| Documentation | markdown conclusions tied to the executed result |
| Calibration | confidence quality for the champion model |
| Decision layer | action bands, coverage, and accuracy by band |
| Demo inference | top-k predictions and user-facing action for sample images |
| Detection | bounding boxes, crop exports, and per-region predictions |

This keeps model changes interpretable. A new experiment should explain **what
changed**, **why it changed**, and whether the result is strong enough to alter
the project direction.

## 13. Current Reasoning Conclusion

The current champion is **ResNet50 FT-V2**. The project has moved from general
model search to **targeted improvement and decision design**:

1. **Calibrate confidence scores**.
2. Study repeated **hard-class confusion pairs**.
3. Improve **deterministic single-image inference**.
4. Define **product decision bands** from calibrated predictions.
5. Demonstrate the final **user-facing prediction workflow**.
6. Extend toward **multi-food detection** through detector crops.
7. Run controlled **accuracy Phase 1** experiments before using external
   datasets.
8. Revisit compact models only if **deployment constraints** become more
   important than accuracy.
