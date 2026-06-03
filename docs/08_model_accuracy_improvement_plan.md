# 8. Model Accuracy Improvement Plan

## 1. Objective

Improve FoodLens classifier accuracy while preserving the project's existing
evaluation discipline:

- reuse the same Food-101 train / validation / test split contract;
- compare against `resnet50_ft_v2_best.pth` as the champion baseline;
- report validation and held-out test top-1 accuracy;
- report top-5 accuracy, calibration ECE, class-level failures, model size, and
  latency;
- promote a model only when it improves product value, not only notebook
  complexity.

Current champion:

| Model | Test top-1 | Test top-5 | Test ECE | T4 latency |
| --- | ---: | ---: | ---: | ---: |
| ResNet50 FT-V2 | 78.28% | 92.65% | 0.0265 | 5.35 ms/image |

Target for the next accuracy phase:

| Target | Threshold |
| --- | ---: |
| Near-term test top-1 | 82%+ |
| Strong result test top-1 | 85%+ |
| Test top-5 | keep above 92.65% |
| Test ECE after calibration | keep near or below 0.0265 |
| Product decision layer | preserve or improve auto-accept accuracy |

## 2. Guardrails

Do not tune on the test set. The test split remains a final comparison only.

Do not mix external datasets directly into the 101-class target unless the
taxonomy mapping is audited. Use external datasets first for pretraining,
representation learning, detector training, or crop robustness.

Do not promote a larger or slower model from accuracy alone. Promotion requires
an explicit trade-off statement covering accuracy, calibration, latency, model
size, and FoodLens product behavior.

Every candidate must export:

- training history;
- validation metrics;
- test metrics for final candidates;
- test predictions;
- per-class report;
- confusion-pair report;
- calibration summary;
- model-size and latency report;
- decision-layer metrics after recalibration.

## 3. Phase 1: Stronger Food-101 Training

Phase 1 keeps Food-101 as the only supervised classification dataset. The goal
is to determine whether better training alone can beat the current champion.

### Experiment Matrix

| Run ID | Model | Training scope | Input size | Main change | Promotion signal |
| --- | --- | --- | ---: | --- | --- |
| `A1` | ResNet50 FT-V3 | full backbone | 224 | lower LR full fine-tune from FT-V2 | establishes whether more unfreezing helps |
| `A2` | ResNet50 FT-V4 | full backbone | 320 | higher resolution + current recipe | improves hard fine-grained classes |
| `A3` | ConvNeXt-Tiny FT | full backbone | 224 | fair full fine-tune challenger | beats frozen-head result by a wide margin |
| `A4` | ConvNeXt-Tiny FT-HR | full backbone | 320 | higher resolution ConvNeXt | challenges ResNet50 with modern features |
| `A5` | EfficientNetV2-S FT | full backbone | 300 | compact but stronger EfficientNet family | accuracy / latency trade-off |
| `A6` | Swin-Tiny or ViT-S FT | full backbone | 224 or 384 | transformer-style challenger | only continue if validation is strong |

### Training Recipe

Start conservatively and change one major variable per run:

- pretrained ImageNet weights;
- AdamW optimizer;
- cosine or one-cycle learning-rate schedule;
- label smoothing between `0.05` and `0.10`;
- early stopping on validation top-1 with patience;
- checkpoint best validation top-1 model;
- run final held-out test evaluation only for the best candidates.

Augmentation candidates:

- random resized crop;
- horizontal flip;
- color jitter;
- RandAugment or TrivialAugment;
- random erasing;
- MixUp and CutMix only after a clean non-mixed baseline exists.

### Hard-Class Focus

Track these class groups separately:

- meat: `steak`, `filet_mignon`, `pork_chop`;
- raw seafood / tartare: `tuna_tartare`, `beef_tartare`, `ceviche`;
- browned pastry: `bread_pudding`, `apple_pie`, `french_toast`;
- chocolate desserts: `chocolate_cake`, `chocolate_mousse`;
- dumpling-like classes: `gyoza`, `dumplings`;
- soup / noodle classes: `ramen`, `pho`, `miso_soup`.

Each candidate should report whether gains come from broad improvement or only
from easy classes.

## 4. Phase 2: Noise-Aware Food-101 Refinement

Food-101 training labels are not perfectly clean, so the next layer is
noise-aware training.

Candidate techniques:

| Technique | Purpose |
| --- | --- |
| high-loss sample audit | find suspicious or ambiguous training images |
| generalized cross entropy or symmetric CE | reduce label-noise sensitivity |
| class-balanced sampling | avoid weak classes being washed out by easy classes |
| hard-class oversampling | spend more updates on persistent failure clusters |
| top-k consistency review | flag samples where the model repeatedly disagrees with the label |

Promotion rule:

> Continue only if noise-aware training improves validation top-1 or hard-class
> F1 without harming top-5 or calibration.

## 5. Phase 3: External Dataset Pretraining

External data should be used after Phase 1 establishes a stronger Food-101-only
baseline.

Recommended order:

| Dataset | Role | Reason |
| --- | --- | --- |
| FoodX-251 | classifier pretraining | larger fine-grained food taxonomy |
| VireoFood-172 | classifier pretraining / ingredient context | food and ingredient diversity |
| FoodSeg103 | segmentation or crop robustness | improves region understanding |
| AIcrowd / MyFoodRepo food benchmark | detector or segmentation training | real multi-food annotations |
| Nutrition5k | later nutrition-oriented product work | useful after recognition is stable |

Do not directly combine these labels with Food-101 labels until class mapping is
audited. Preferred workflow:

1. Pretrain on the external food dataset.
2. Replace the classification head.
3. Fine-tune on the project Food-101 split.
4. Recalibrate on validation logits.
5. Evaluate on the unchanged Food-101 test split.

## 6. Phase 4: Product Accuracy For FoodLens

FoodLens accuracy depends on more than Food-101 classification. The app must
detect the right crop before classifying it.

Separate product improvements:

- train or fine-tune a food detector / segmenter;
- classify detector crops with the champion classifier;
- reject non-food, containers, and context objects;
- combine detector confidence, classifier confidence, crop size, and known
  confusion pairs;
- evaluate crop-level accuracy separately from whole-image Food-101 accuracy.

Recommended detector-first datasets:

- AIcrowd / MyFoodRepo food recognition benchmark;
- FoodSeg103;
- any internally curated FoodLens crop review set.

## 7. Promotion Criteria

A model can replace ResNet50 FT-V2 only if it satisfies one of these:

1. improves held-out test top-1 by at least 2 percentage points without harming
   top-5, ECE, or latency materially;
2. improves hard-class F1 materially while holding overall accuracy stable;
3. reduces model size or latency substantially while keeping accuracy close to
   the champion;
4. improves FoodLens decision-layer coverage at the same or better
   auto-accept accuracy.

If a model has higher top-1 but worse calibration, it must be recalibrated and
re-evaluated before promotion.

## 8. First Execution Slice

Start with two runs:

| Run | Reason |
| --- | --- |
| `A1` ResNet50 FT-V3 full-backbone 224 | lowest-risk continuation from champion |
| `A3` ConvNeXt-Tiny full fine-tune 224 | fair test of the strongest modern challenger |

Required outputs:

```text
/kaggle/working/results/accuracy_phase1/
|-- a1_resnet50_ft_v3/
|   |-- best_model.pth
|   |-- validation_metrics.csv
|   |-- test_metrics.csv
|   |-- calibration_summary.csv
|   |-- class_report.csv
|   `-- confusion_pairs.csv
`-- a3_convnext_tiny_ft/
    |-- best_model.pth
    |-- validation_metrics.csv
    |-- test_metrics.csv
    |-- calibration_summary.csv
    |-- class_report.csv
    `-- confusion_pairs.csv
```

After those two runs, decide whether to continue high-resolution training or
move to noise-aware refinement.

## 9. Kaggle Execution

Completed A1 script kernel:

```text
https://www.kaggle.com/code/tuannm3823/foodlens-accuracy-phase-1-a1-resnet50-ft-v3
```

Notebook-backed A1 kernel:

```text
https://www.kaggle.com/code/tuannm3823/foodlens-a1-resnet50-ft-v3-notebook
```

Local kernel package:

```text
kaggle/accuracy_phase1/
|-- kernel-metadata.json
`-- foodlens_accuracy_phase1_a1.py
```

The script starts from the attached `resnet50_ft_v2_best.pth` artifact and
trains all ResNet50 parameters with lower backbone learning rate and higher
classifier-head learning rate. It also detects Kaggle P100 assignments whose
CUDA architecture is not supported by the default PyTorch image and installs a
compatible PyTorch / torchvision build before training.

Notebook entry point:

```text
notebooks/09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb
```

Kaggle upload package:

```text
kaggle/accuracy_phase1/
|-- kernel-metadata.json
|-- 09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb
`-- foodlens_accuracy_phase1_a1.py
```

Active A3 notebook entry point:

```text
notebooks/10_food101_accuracy_phase1_a3_convnext_tiny.ipynb
```

Active A3 Kaggle upload package:

```text
kaggle/accuracy_phase1_a3/
|-- kernel-metadata.json
|-- 10_food101_accuracy_phase1_a3_convnext_tiny.ipynb
`-- foodlens_accuracy_phase1_a3.py
```

### A3 Key Points

- Start from the ConvNeXt-Tiny frozen-head checkpoint when the Kaggle model
  artifact is attached.
- Fall back to ImageNet weights if that checkpoint is unavailable, while
  recording the source in the run metadata.
- Unfreeze the full backbone and train with separate learning rates for
  pretrained features and the classifier head.
- Keep the Food-101 split, metrics, calibration, latency, and class-error
  reporting consistent with the champion contract.
- Promote only if A3 closes the frozen-head gap and creates a better product
  trade-off than ResNet50 FT-V2.

The notebook keeps each section focused on the key decisions behind the training, evaluation, and promotion contract.

## 10. A1 Result

A1 completed on Kaggle. It should not replace the champion.

| Model | Test top-1 | Test top-5 | Test ECE calibrated | Latency |
| --- | ---: | ---: | ---: | ---: |
| ResNet50 FT-V2 champion | 78.28% | 92.65% | 0.0265 | 5.35 ms/image |
| A1 ResNet50 FT-V3 full backbone | 78.23% | 92.45% | 0.0229 | 5.53 ms/image |

Interpretation:

- full-backbone continuation did not improve held-out accuracy;
- top-5 accuracy dropped slightly;
- calibration improved slightly after temperature scaling;
- latency increased slightly;
- the next Phase 1 run should be `A3` ConvNeXt-Tiny full fine-tuning rather
  than more ResNet50 continuation at 224px.
