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
| Decision layer | 58.02% auto-accept coverage at 96.47% top-1 |

The current model direction is settled: **keep ResNet50 FT-V2 as the
champion**. Modern backbone replacement is not justified by the current
evidence, and temperature scaling has made confidence scores more reliable.

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
- The next useful improvement is **demo hardening and product validation**, not
  another architecture search.

## 3. Implemented Next Task

Notebook 5 now implements a confidence-based decision layer around calibrated
predictions.

Suggested decision bands:

| Band | Condition | Product action |
| --- | --- | --- |
| Auto-accept | high calibrated confidence and not a known hard class | accept the top-1 label |
| Suggest | medium confidence or visually similar class group | show top-5 suggestions |
| Confirm | low confidence, hard class, or small top-1/top-2 margin | ask the user to confirm |
| Review | repeated business-critical confusion pair | flag for manual or rule-based review |

The exact thresholds are learned from Notebook 4 outputs rather than chosen
manually. The notebook analyzes calibrated confidence, correctness,
top-1/top-2 margin, and hard-class membership together.

## 4. Notebook 5

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

Expected outputs:

- `decision_policy.csv`
- `decision_band_metrics.csv`
- `decision_examples_auto_accept.csv`
- `decision_examples_suggest.csv`
- `decision_examples_confirm.csv`
- `decision_examples_review.csv`

## 5. Next After Notebook 5

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

The latest demo correctly predicts six sample images and routes distinctive
classes to **auto-accept** while routing known hard classes such as `steak`,
`tuna_tartare`, and `chocolate_mousse` to **suggest**.

Next action:

> Rerun Notebook 6 after the export refinement, then use the exported demo CSVs
> to update final reporting or build a lightweight presentation/demo page.

## 6. Secondary Improvements

After the decision layer is in place, the next improvements should be scoped
and evidence-driven:

1. **Demo stress testing:** include examples that trigger all four decision
   bands: auto-accept, suggest, confirm, and review.
2. **Hard-class review:** build class-group reports for meat dishes, tartare
   dishes, pastry desserts, and chocolate desserts.
3. **Inference packaging:** create a small reusable inference function or demo
   notebook for one-image prediction.
4. **Artifact documentation:** document the final champion checkpoint,
   calibrated temperature, and expected input preprocessing.
5. **Compact model revisit:** revisit EfficientNet-B0 or another small model
   only if deployment size becomes more important than accuracy.

## 7. Stop Conditions

Avoid expanding the project endlessly. A next experiment should be skipped if
it does not improve at least one of these:

- held-out accuracy;
- calibrated confidence quality;
- decision coverage at acceptable accuracy;
- inference usability;
- model size or latency under a real deployment constraint.
