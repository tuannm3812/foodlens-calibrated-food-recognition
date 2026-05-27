# 3. Notebook: Food-101 Transfer Learning And Fine-Tuning

## 1. File

Notebook:
[`../notebooks/01_food101_baseline_transfer_finetuning.ipynb`](../notebooks/01_food101_baseline_transfer_finetuning.ipynb)

This notebook is maintained as the Kaggle source of truth for the personal
Food-101 project.

## 2. Purpose

This notebook is the source of truth for the Food-101 project workflow. It
builds the dataset manifest, audits the image data, trains transfer-learning
baselines, fine-tunes the selected winner, and evaluates the final checkpoint
with metrics, diagnostics, qualitative errors, and efficiency reporting.

## 3. Sections

| Section | Role |
| --- | --- |
| 1. Project Summary | project context, dataset challenge, and experiment stages |
| 2. Runtime, Imports, And Configuration | centralized imports, uppercase `CFG` constants, seed setup, and device selection |
| 3. Data Ingestion And Audit | fixed Kaggle dataset path, manifest creation, class balance, and image-shape sampling |
| 4. Preprocessing, Splits, And Dataloaders | transforms, stratified split, custom `FoodDataset`, and dataloaders |
| 5. Model Construction | reusable builders for GoogLeNet, ResNet50, and MobileNetV3 classifier heads |
| 6. Training And Evaluation Utilities | shared training, validation, checkpointing, and plotting functions |
| 7. Part A: Transfer Learning Comparison | frozen-backbone benchmark and per-class error analysis |
| 8. Part B: ResNet50 Fine-Tuning | selective unfreezing experiments for `layer4` and `layer3 + layer4` |
| 9. Final Model Evaluation And Inference | selected checkpoint loading, test metrics, hard-class confusion, qualitative errors, and latency |

Follow-up notebooks:

| Notebook | Role |
| --- | --- |
| [`../notebooks/02_resnet50_training_refinements.ipynb`](../notebooks/02_resnet50_training_refinements.ipynb) | tests longer ResNet50 fine-tuning, early stopping, scheduling, stronger augmentation, and label smoothing |
| [`../notebooks/03_modern_backbone_comparison.ipynb`](../notebooks/03_modern_backbone_comparison.ipynb) | compares EfficientNet-B0 and ConvNeXt-Tiny against the ResNet50 baseline |

## 4. Configuration

Key configuration values from the notebook:

| Setting | Value |
| --- | ---: |
| Random seed | 42 |
| Batch size | 32 |
| Image size | 224 x 224 |
| Initial learning rate | 1e-3 |
| Mode | `CFG.MODE = "train"` or `CFG.MODE = "inference"` |
| Part A epochs | 5 |
| Fine-tuning epochs | 5 |
| Fine-tuning learning rate | 1e-5 |
| Artifact directory | `/kaggle/input/models/tuannm3823/food101-baseline-artifacts/pytorch/default/1/food101-baseline-artifacts` |

## 5. Generated Outputs

The notebook writes model artifacts under:

```text
/kaggle/working/results
```

Expected checkpoint names include:

- `best_model_googlenet.pth`
- `best_model_resnet50.pth`
- `best_model_mobilenetv3.pth`
- `finetuned_exp_1_layer4.pth`
- `finetuned_exp_2_layer3_4.pth`

Evaluation artifacts include:

- `history_*.csv`
- `transfer_val_predictions.csv`
- `transfer_val_metrics.csv`
- `transfer_val_class_report.csv`
- `val_predictions.csv`
- `val_metrics.csv`
- `val_class_report.csv`
- `test_predictions.csv`
- `test_metrics.csv`
- `test_class_report.csv`
- `test_confusion_pairs.csv`
- `final_model_efficiency.csv`
- `qualitative_error_examples.csv`
- `figures/test_hard_class_confusion.png`
- `figures/qualitative_error_examples.png`

These files are Kaggle outputs and should not be committed to git.

## 6. Current Limitations

The current notebook now has a stronger evaluation layer. Future iterations
should focus on training improvements and architecture comparison in a second
notebook so the baseline remains stable.
