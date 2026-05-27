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

The current champion is **Exp 2: ResNet50 with `layer3` and `layer4`
fine-tuned**, reaching **72.86% validation top-1 accuracy**.

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

The current champion is a strong personal-project baseline because it exceeds
70% validation accuracy. The next model improvement should be measured, not
speculative:

1. Use notebook 1 as the fixed baseline and artifact-backed evaluation
   workflow.
2. Run the second controlled ResNet50 refinement notebook with longer
   fine-tuning,
   learning-rate scheduling, and early stopping.
3. After that, compare a stronger modern backbone such as EfficientNet-B0
   or ConvNeXt-Tiny against ResNet50 under the same split and reporting
   protocol.

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
