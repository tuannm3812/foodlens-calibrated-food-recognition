# 4. Model Results

## 1. Transfer Learning Results

Part A freezes the pretrained convolutional features and trains only the
custom 3-layer classifier heads.

Latest Kaggle run:

| Model | Best validation accuracy | Checkpoint |
| --- | ---: | --- |
| GoogLeNet | 42.03% | `best_model_googlenet.pth` |
| ResNet50 | 59.49% | `best_model_resnet50.pth` |
| MobileNetV3 Large | 54.60% | `best_model_mobilenetv3.pth` |

ResNet50 is the strongest Part A model in the saved output and is selected for
fine-tuning.

## 2. Fine-Tuning Results

Part B starts from the selected ResNet50 checkpoint and tests two unfreezing
depths.

| Experiment | Trainable backbone scope | Learning rate | Best validation accuracy |
| --- | --- | ---: | ---: |
| Exp 1 | `layer4` | 1e-5 | 69.23% |
| Exp 2 | `layer3` + `layer4` | 1e-5 | 72.86% |

The strongest baseline fine-tuning run is **Exp 2: ResNet50 with `layer3` and
`layer4` fine-tuned**, reaching **72.86% validation top-1 accuracy**.

Final evaluation:

| Split | Top-1 accuracy | Top-5 accuracy |
| --- | ---: | ---: |
| Validation | 72.86% | 90.99% |
| Test | 73.64% | 91.18% |

## 3. Error Analysis

Per-class F1 scores show which categories are most visually separable and which
remain difficult after fine-tuning.

Top classes after frozen ResNet50 transfer learning:

| Class | F1 score |
| --- | ---: |
| `edamame` | 0.958 |
| `seaweed_salad` | 0.896 |
| `bibimbap` | 0.893 |
| `oysters` | 0.891 |
| `pho` | 0.891 |
| `macarons` | 0.886 |
| `takoyaki` | 0.882 |
| `sashimi` | 0.876 |
| `spaghetti_carbonara` | 0.873 |
| `frozen_yogurt` | 0.872 |

Hardest classes after final fine-tuning on validation and test remain visually
ambiguous. The latest test run highlights:

| Class | F1 score |
| --- | ---: |
| `ceviche` | 0.549 |
| `tuna_tartare` | 0.549 |
| `foie_gras` | 0.538 |
| `scallops` | 0.537 |
| `filet_mignon` | 0.521 |
| `bread_pudding` | 0.521 |
| `chocolate_mousse` | 0.498 |
| `pork_chop` | 0.495 |
| `ravioli` | 0.473 |
| `steak` | 0.450 |

## 4. Interpretation

Fine-tuning deeper ResNet50 blocks provides a clear gain over using the network
as a frozen feature extractor. The improvement is likely because Food-101 needs
texture and presentation-specific features that differ from generic ImageNet
objects.

The hardest classes are mostly foods with ambiguous plating, similar color and
texture, or overlapping visual cues. Additional augmentation, longer
fine-tuning, label-noise inspection, and top-k reporting would be reasonable
next experiments.

## 5. Key Insights

The saved run supports four practical conclusions:

1. **Backbone depth matters for Food-101.** ResNet50 outperformed GoogLeNet and
   MobileNetV3 in the frozen-feature comparison, suggesting the residual
   backbone provides a stronger starting representation for fine-grained food
   categories.
2. **Domain adaptation is necessary.** Fine-tuning `layer3` and `layer4`
   improved validation accuracy from 59.49% to 72.86%, a gain of 13.37
   percentage points over the frozen ResNet50 baseline.
3. **The model handles distinctive dishes well.** Classes such as `edamame`,
   `seaweed_salad`, `bibimbap`, and `pho` likely benefit from distinctive
   color, shape, or plating cues.
4. **Remaining errors are semantic and visual, not just capacity-related.**
   Classes such as `steak`, `chocolate_mousse`, `ravioli`, and `pork_chop`
   often overlap with nearby categories in texture, color, and composition.
   These errors should be inspected with confusion matrices and image examples
   before assuming that a larger model is the right fix.
5. **Top-5 accuracy changes the product story.** The model reaches 91.18%
   test top-5 accuracy, which is valuable for food-recognition interfaces that
   can show ranked suggestions rather than a single hard prediction.
6. **Confidence calibration needs attention.** High-confidence errors such as
   `sashimi -> sushi`, `ramen -> pho`, `gyoza -> dumplings`, and
   `frozen_yogurt -> ice_cream` are semantically reasonable, but near-100%
   confidence on wrong classes suggests future calibration work.

## 6. Notebook Refinements Added

The notebook has been refined with stronger evaluation coverage:

- Held-out test evaluation for the selected checkpoint.
- Top-1 and top-5 accuracy reporting.
- Normalized confusion matrix for the hardest classes.
- CSV exports for histories, predictions, metrics, and per-class reports.
- Qualitative high-confidence error examples.
- Model-size and single-image inference-latency reporting.
- Top confusion-pair export to identify repeated substitutions.

Latest efficiency result:

| Metric | Value |
| --- | ---: |
| Parameters | 24,714,405 |
| Model size | 94.48 MB |
| T4 latency | 5.82 ms/image |

## 7. Model Improvement Direction

The baseline notebook is a stable personal-project baseline because it exceeds
70% validation accuracy and has held-out test reporting. Later notebooks build
on it in controlled steps:

1. Use notebook 1 as the fixed baseline and artifact-backed evaluation
   workflow.
2. Run the second controlled ResNet50 refinement notebook with longer
   fine-tuning,
   learning-rate scheduling, and early stopping.
3. Compare stronger modern backbones such as EfficientNet-B0 and ConvNeXt-Tiny
   against ResNet50 under the same split and reporting protocol.

This order keeps the scope defensible: it separates real generalization gains
from changes that only improve validation accuracy by chance.

## 8. ResNet50 Refinement Results

Notebook 2 is now the strongest evaluated ResNet50 run. Starting from the
baseline fine-tuned checkpoint, the refined recipe used stronger augmentation,
AdamW, label smoothing, learning-rate scheduling, and longer fine-tuning.

Final evaluation for `resnet50_ft_v2_best.pth`:

| Split | Baseline top-1 | Refined top-1 | Change | Refined top-5 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 72.86% | 77.90% | +5.04 pp | 92.36% |
| Test | 73.64% | 78.28% | +4.63 pp | 92.65% |

Efficiency remains effectively unchanged because the architecture is still
ResNet50:

| Metric | Baseline | Refined |
| --- | ---: | ---: |
| Parameters | 24,714,405 | 24,714,405 |
| Model size | 94.48 MB | 94.48 MB |
| T4 latency | 5.82 ms/image | 5.35 ms/image |

The result is a real generalization improvement, not only a validation gain.
The refined model improves held-out test top-1 by **4.63 percentage points**
while preserving the same deployment footprint.

Remaining error patterns are concentrated in visually similar food families:

- `steak`, `filet_mignon`, and `pork_chop` remain difficult because they share
  grilled textures and overlapping plating styles.
- `tuna_tartare`, `beef_tartare`, and `ceviche` are confused because they are
  often presented as small raw-food portions with similar colors.
- `bread_pudding`, `apple_pie`, and `french_toast` overlap through browned
  pastry-like textures.
- `chocolate_cake`, `chocolate_mousse`, and related desserts remain a
  high-error cluster.

The high-confidence error examples also show a calibration issue: several
incorrect predictions have confidence above 0.98. The next model should
therefore be judged on accuracy, top-5 accuracy, confusion behavior, and
confidence quality rather than top-1 accuracy alone.

## 9. Modern Backbone Comparison Results

Notebook 3 compared frozen-head EfficientNet-B0 and ConvNeXt-Tiny against the
refined ResNet50 FT-V2 champion. Neither challenger beat the current champion.

| Model | Stage | Test top-1 | Test top-5 | Parameters | Model size | T4 latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ResNet50 FT-V2 | reference | 78.28% | 92.65% | 24,714,405 | 94.48 MB | 5.35 ms/image |
| ConvNeXt-Tiny | frozen head | 70.92% | 90.24% | 28,371,141 | 108.23 MB | 7.17 ms/image |
| EfficientNet-B0 | frozen head | 52.13% | 77.02% | 4,820,705 | 18.55 MB | 7.44 ms/image |

ConvNeXt-Tiny is the best modern-backbone challenger, but it is still **7.36
percentage points below** ResNet50 FT-V2 on test top-1 accuracy. It is also
larger and slower in the current setup, so it should not replace the ResNet50
champion.

EfficientNet-B0 is much smaller, but its frozen-head accuracy is too low for
the current product-quality target. It remains interesting only if deployment
size becomes more important than accuracy.

The architecture comparison confirms that the strongest improvement so far
came from the ResNet50 training recipe, not from switching backbones. The next
technical step should focus on error-driven refinement and calibration before
adding more architectures.

## 10. Calibration And Inference Results

Notebook 4 evaluated the current champion, `resnet50_ft_v2_best.pth`, with
temperature scaling and deterministic single-image inference.

Calibration improved without changing the model's top-k ranking:

| Split | Temperature | ECE before | ECE after | Change |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.958 | 0.0419 | 0.0251 | -0.0168 |
| Test | 0.958 | 0.0432 | 0.0265 | -0.0167 |

The calibrated model keeps the same held-out test accuracy:

| Metric | Value |
| --- | ---: |
| Test top-1 accuracy | 78.28% |
| Test top-5 accuracy | 92.65% |

The main value of Notebook 4 is not higher accuracy. It makes the model more
usable by improving confidence quality and exporting product-oriented
diagnostics.

Hardest test classes after calibrated evaluation:

| Class | F1 score |
| --- | ---: |
| `chocolate_mousse` | 0.529 |
| `steak` | 0.560 |
| `pork_chop` | 0.589 |
| `bread_pudding` | 0.611 |
| `tuna_tartare` | 0.617 |
| `foie_gras` | 0.625 |
| `scallops` | 0.644 |
| `apple_pie` | 0.645 |
| `crab_cakes` | 0.650 |
| `ravioli` | 0.652 |

Top repeated confusion pairs still reflect visually similar food families:

- `tuna_tartare -> beef_tartare`
- `steak -> filet_mignon`
- `bread_pudding -> apple_pie`
- `filet_mignon -> steak`
- `chocolate_cake -> chocolate_mousse`
- `gyoza -> dumplings`
- `donuts -> beignets`

Single-image inference is now available through a deterministic helper. The
sample Kaggle image `miso_soup/1014272.jpg` was correctly predicted as
`miso_soup` with 0.992 calibrated confidence.

Interpretation:

- Temperature scaling should be kept for product-facing confidence values.
- Accuracy is already stable; the main remaining risk is class ambiguity.
- Next work should convert calibrated confidence into decision thresholds such
  as auto-accept, show top-k suggestions, or ask for user confirmation.

## 11. Confidence Decision Layer Results

Notebook 5 converts calibrated predictions into practical product actions. The
selected policy is intentionally simple and auditable:

| Threshold | Value |
| --- | ---: |
| Auto-accept confidence | 0.70 |
| Suggest confidence | 0.35 |
| Top-1/top-2 margin | 0.40 |

Decision-band results on the held-out test predictions:

| Decision band | Coverage | Top-1 accuracy | Top-5 contains actual | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Auto-accept | 58.02% | 96.47% | 98.84% | safe enough for direct labeling |
| Suggest | 20.78% | 79.94% | 100.00% | strong candidate for ranked user choice |
| Confirm | 18.99% | 29.98% | 66.16% | user confirmation is required |
| Review | 2.21% | 0.00% | 88.79% | known confusion cases need inspection |

Technical insight:

- The model can automatically accept more than half of test predictions while
  keeping auto-accept top-1 accuracy above **96%**.
- The suggestion band is especially valuable because the correct answer is in
  the top five for **100%** of those cases.
- The confirm and review bands correctly isolate risky examples; they should
  not be treated as model failure alone, but as evidence that product behavior
  should change by confidence and ambiguity.

Business insight:

- The project is now closer to a usable assisted-labeling workflow than a raw
  classifier. A product can accept easy dishes, show choices for ambiguous
  dishes, and avoid overconfident automation on known hard classes.

## 12. Final Demo Inference Results

Notebook 6 validates the end-to-end user-facing workflow. It loads:

- the **ResNet50 FT-V2** checkpoint;
- calibrated temperature **0.958111**;
- decision policy **0.70 / 0.35 / 0.40**;
- **15 hard classes** and **30 repeated confusion pairs**.

Latest demo output:

| Image class | Predicted class | Confidence | Margin | Decision |
| --- | --- | ---: | ---: | --- |
| `miso_soup` | `miso_soup` | 0.9917 | 0.9909 | auto-accept |
| `ice_cream` | `ice_cream` | 0.5476 | 0.1221 | suggest |
| `grilled_salmon` | `grilled_salmon` | 0.6251 | 0.5521 | confirm |
| `steak` | `filet_mignon` | 0.5765 | 0.1863 | review |

Interpretation:

- The latest demo intentionally covers all four action bands:
  **auto-accept**, **suggest**, **confirm**, and **review**.
- `miso_soup` is a clear high-confidence prediction and is correctly routed to
  **auto-accept**.
- `ice_cream` is correct but close to `frozen_yogurt`, so the system routes it
  to **suggest** instead of over-automating an ambiguous dessert case.
- `grilled_salmon` is correct but marked as a hard-case prediction, so the
  system asks the user to **confirm**.
- `steak -> filet_mignon` is an incorrect but semantically close meat-dish
  confusion. The **review** band catches this known risk pattern.
- Notebook 6 now exports `demo_predictions.csv` and
  `demo_decision_summary.csv` so the final demo can be reviewed outside the
  notebook.

Final project conclusion:

- The champion model remains **ResNet50 FT-V2**.
- The best product use case is **assisted food tagging with ranked
  suggestions**, not fully automatic labeling for every image.
- The main limitation is **visual ambiguity between similar dishes**, especially
  meat dishes, tartare-style dishes, and dessert families.
- The final workflow is now more mature than a raw classifier because it
  combines **top-k prediction**, **calibrated confidence**, and a
  **decision action** for each image.

## 13. Accuracy Phase 1 A1 Result

Notebook 9 tests whether continuing from the ResNet50 FT-V2 champion with
full-backbone fine-tuning improves Food-101 accuracy.

Kaggle kernels:

- completed script run:
  `https://www.kaggle.com/code/tuannm3823/foodlens-accuracy-phase-1-a1-resnet50-ft-v3`
- notebook-backed run:
  `https://www.kaggle.com/code/tuannm3823/foodlens-a1-resnet50-ft-v3-notebook`

Result:

| Model | Test top-1 | Test top-5 | Test ECE calibrated | Latency |
| --- | ---: | ---: | ---: | ---: |
| ResNet50 FT-V2 champion | 78.28% | 92.65% | 0.0265 | 5.35 ms/image |
| A1 ResNet50 FT-V3 full backbone | 78.23% | 92.45% | 0.0229 | 5.53 ms/image |

Decision:

- Do not promote A1.
- Full-backbone ResNet50 continuation at 224px did not improve held-out
  accuracy.
- The slight calibration improvement is useful but not enough to offset the
  accuracy, top-5, and latency results.
- Continue Phase 1 with `A3`: full fine-tuning ConvNeXt-Tiny under the same
  split and reporting contract.
