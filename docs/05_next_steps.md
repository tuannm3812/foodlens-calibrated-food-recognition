# 5. Next Steps

## 1. Current Position

The project now has a clear champion and a trustworthy evaluation layer.

| Stage | Best result |
| --- | ---: |
| Frozen ResNet50 transfer learning | 59.49% validation top-1 |
| Baseline fine-tuned ResNet50 `layer3 + layer4` | 73.64% test top-1 |
| Refined ResNet50 FT-V2 | 78.28% test top-1 |
| Refined ResNet50 FT-V2 | 92.65% test top-5 |
| Calibrated ResNet50 FT-V2 | 0.0265 test ECE |
| A3 ConvNeXt-Tiny full fine-tune | 83.41% test top-1 |
| A3 ConvNeXt-Tiny full fine-tune | 95.73% test top-5 |
| A3b ConvNeXt-Tiny continued fine-tune | 83.90% test top-1 |
| A3b ConvNeXt-Tiny continued fine-tune | 95.78% test top-5 |
| E1 expanded taxonomy baseline | 86.10% test top-1 across 130 classes |
| E1 expanded taxonomy baseline | 96.88% test top-5 across 130 classes |
| Decision layer | 58.02% auto-accept coverage at 96.47% top-1 |

The current production reference remains **ResNet50 FT-V2** because its
calibrated decision layer is stronger. A3b ConvNeXt-Tiny is now the accuracy
leader, but it needs decision-layer recalibration before product promotion.

The active model-improvement direction is now:

> Keep ResNet50 FT-V2 as the product baseline, continue the ConvNeXt-Tiny
> accuracy phase, recalibrate the decision layer before promotion, and improve
> the expanded 130-class classifier with controlled fine-tuning.

The detailed execution plan is maintained in
[`08_model_accuracy_improvement_plan.md`](08_model_accuracy_improvement_plan.md).

## 2. What The Latest Output Means

Notebooks 4 to 6 changed the project from pure model evaluation to
product-oriented decision design.

Key output:

- **Accuracy stayed stable:** 78.28% test top-1 and 92.65% test top-5.
- **Calibration improved:** test ECE moved from 0.0432 to 0.0265.
- **Hard classes persisted:** `chocolate_mousse`, `steak`, `pork_chop`,
  `bread_pudding`, and `tuna_tartare` remain difficult.
- **Decision routing works:** the system can auto-accept easy cases and route
  hard classes to suggestions or confirmation.
- **Single-image inference works:** the final demo returns top-k predictions,
  calibrated confidence, a decision band, and a recommended action.

Interpretation:

- The model is strong enough for **ranked suggestions**.
- The model should not expose raw confidence without the calibrated
  temperature.
- The next accuracy work should continue from A3b and E1 before broad
  architecture search or direct dataset mixing.
- Product hardening remains important, but the next model step is Phase 1 of
  the accuracy plan.

## 3. Active Accuracy Phase

The first expanded-taxonomy baseline is now complete:

| Run | Model | Why this comes first |
| --- | --- | --- |
| `A1` | ResNet50 FT-V3 full-backbone 224 | did not beat ResNet50 FT-V2 |
| `A3` | ConvNeXt-Tiny full fine-tune 224 | best accuracy result so far, but calibration is weaker |
| `A3b` | ConvNeXt-Tiny continued fine-tune 224 | best accuracy result so far, but calibration is weaker |
| `A4` | ConvNeXt-Tiny full fine-tune 320 | prepared; next step if keeping ConvNeXt backbone |
| `T1` | Expanded taxonomy audit | found 36 raw candidate new classes and a conservative 130-class target |
| `E1` | Expanded taxonomy v1 baseline | completed head-only 130-class run from the A3b checkpoint |
| `E2` | Expanded taxonomy fine-tune | completed partial-ConvNeXt fine-tune from E1 checkpoint |

Promotion criteria:

- improve held-out test top-1 by at least 2 percentage points;
- preserve or improve top-5 accuracy;
- recalibrate and keep ECE near or below the current 0.0265 test result;
- inspect hard-class F1 and repeated confusion pairs;
- report latency and model size before replacing the champion.

External datasets should remain separated by purpose. Use audited labels for
the expanded 130-class taxonomy, and keep other external sources focused on
pretraining, detector training, or crop robustness rather than direct
101-class label mixing.

## 4. Implemented Decision-Layer Task

Notebook 5 now implements a confidence-based decision layer around calibrated
predictions.

Suggested decision bands:

| Band | Condition | Product action |
| --- | --- | --- |
| Auto-accept | high calibrated confidence and not a known hard class | accept the top-1 label |
| Suggest | medium confidence or visually similar class group | show top-5 suggestions |
| Confirm | low confidence, hard class, or small top-1/top-2 margin | ask the user to confirm |
| Review | repeated business-critical confusion pair | flag for manual or rule-based review |

The exact thresholds are learned from each notebook's corresponding outputs
rather than chosen manually. The flow analyzes confidence, correctness, top-1/top-2
margin (when available), hard-class membership, and frequent confusion patterns.

For the 130-class path, Notebook 15 uses the same workflow on E2 outputs.

## 5. Notebook 5

File:

```text
notebooks/05_confidence_decision_layer.ipynb
```

Purpose:

> Convert calibrated model outputs into product-ready prediction decisions.

Implemented sections:

1. Load `test_predictions_calibrated.csv` from Notebook 4 output.
2. Compute top-1 confidence, top-2 margin, and hard-class flags.
3. Search candidate thresholds for auto-accept, suggest, and confirm bands.
4. Report coverage and accuracy for each band.
5. Export a decision-policy table.
6. Add an inference wrapper that returns both predictions and recommended
   action.

Required Notebook 5 input:

```text
/kaggle/input/notebooks/tuannm3823/resnet50-ft-v2-error-analysis-calibration/results/resnet50_error_calibration
```

That attached Notebook output should contain the calibrated CSV files from
Notebook 4, especially:

- `test_predictions_calibrated.csv`
- `hard_classes_calibrated.csv`
- `top_confusion_pairs_calibrated.csv`

Notebook 4 also creates a single artifact bundle:

```text
/kaggle/working/results/resnet50_error_calibration_artifacts.zip
```

If you prefer a Kaggle Dataset upload, upload that zip once. Notebook 5 can
extract `resnet50_error_calibration_artifacts.zip` automatically and read the
same CSV files from it.

## 5b. Notebook 15 (Expanded Taxonomy Decision Layer)

File:

```text
notebooks/archive/15_expanded_taxonomy_v2_decision_layer.ipynb
```

Purpose:

> Convert E2 expanded-taxonomy outputs into a reusable product policy.

Required Notebook 15 input:

```text
/kaggle/input/notebooks/tuannm3823/foodlens-expanded-taxonomy-v2-finetune/results/expanded_taxonomy/e2_expanded_taxonomy_v2_finetune_224
```

Expected inputs:

- `test_predictions.csv`
- `weak_class_report.csv`
- `expanded_metrics.csv`
- optional confusion pairs (or built from the predictions)

Expected outputs:

- `decision_policy.csv`
- `decision_policy_search.csv`
- `decision_band_metrics.csv`
- `decision_examples_auto_accept.csv`
- `decision_examples_suggest.csv`
- `decision_examples_confirm.csv`
- `decision_examples_review.csv`
- `weak_classes.json`
- `top_confusion_pairs.json`
- `expanded_taxonomy_v2_decision_layer_artifacts.zip`

## 6. Next After Decision-Layer Work

Notebook 5 has produced the selected decision thresholds and band metrics:

| Decision band | Coverage | Key signal |
| --- | ---: | --- |
| Auto-accept | 58.02% | 96.47% top-1 accuracy |
| Suggest | 20.78% | 100.00% top-5 containment |
| Confirm | 18.99% | low top-1 accuracy, user input needed |
| Review | 2.21% | known hard-confusion cases |

Notebook 6 now implements the final demo workflow and exports:

- `demo_predictions.csv`
- `demo_decision_summary.csv`

The latest demo covers all four decision bands:

| Decision band | Demo example | Meaning |
| --- | --- | --- |
| Auto-accept | `miso_soup -> miso_soup` | high-confidence, low-risk prediction |
| Suggest | `ice_cream -> ice_cream` | correct but visually close to another class |
| Confirm | `grilled_salmon -> grilled_salmon` | correct hard-case prediction requiring user check |
| Review | `steak -> filet_mignon` | known meat-dish confusion requiring inspection |

Next action:

> Package the final project story: champion model, calibration, decision layer,
> and four-band demo behavior.

## 7. Multi-Food Detection Phase

The next product capability is **multi-food detection**. The current FoodLens
classifier predicts one label for a whole image or sampled video frame. It does
not locate multiple foods.

Recommended next implementation:

1. Run a pretrained detector such as YOLO or RT-DETR to propose food/object
   regions.
2. Export bounding boxes and crop images.
3. Classify each crop with the existing ResNet50 FT-V2 Food-101 classifier.
4. Apply the calibrated decision layer per crop.
5. Return one FoodLens prediction object per detected food region.

New notebooks:

- `archive/07_multi_food_detection_exploration.ipynb`
- `08_detection_to_foodlens_pipeline.ipynb`

This should stay in the same repo because it is a direct extension of the
FoodLens product workflow.

## 8. Secondary Improvements

After the decision layer is in place, the next improvements should be scoped
and evidence-driven:

1. **Decision-layer recalibration:** recalibrate around the A3b ConvNeXt
   checkpoint before product promotion.
2. **Expanded taxonomy E2:** completed partial ConvNeXt fine-tune from the E1
   classifier. Current 130-class checkpoint now uses partial-unfreezing with lower
   calibrated ECE and targeted weak-class review for `kaathi_rolls`,
   `masala_dosa`, `dosa`, `butter_naan`, and `dal_makhani`.
3. **Final reporting:** turn the model, calibration, decision-layer, and demo
   results into a concise project summary or presentation.
4. **Demo stress testing:** expand the sample set beyond one example per
   decision band and include non-Food-101 images.
5. **Hard-class review:** build class-group reports for meat dishes, tartare
   dishes, pastry desserts, and chocolate desserts.
6. **Inference packaging:** create a small reusable inference function or demo
   notebook for one-image prediction.
7. **Artifact documentation:** document the final champion checkpoint,
   calibrated temperature, and expected input preprocessing.
8. **Compact model revisit:** revisit EfficientNet-B0 or another small model
   only if deployment size becomes more important than accuracy.

## 9. Stop Conditions

Avoid expanding the project endlessly. A next experiment should be skipped if
it does not improve at least one of these:

- held-out accuracy;
- calibrated confidence quality;
- decision coverage at acceptable accuracy;
- inference usability;
- model size or latency under a real deployment constraint.
