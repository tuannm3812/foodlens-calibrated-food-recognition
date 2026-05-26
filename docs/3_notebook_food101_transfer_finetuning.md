# 3. Notebook: Food-101 Transfer Learning And Fine-Tuning

## 1. File

Notebook:
[`../notebooks/1_food101_transfer_finetuning.ipynb`](../notebooks/1_food101_transfer_finetuning.ipynb)

Source notebook copied from the assignment workspace:

```text
Assignment2-25739083.ipynb
```

## 2. Purpose

This notebook is the source of truth for the Food-101 assignment workflow. It
builds the dataset manifest, audits the image data, trains transfer-learning
baselines, fine-tunes the selected winner, and demonstrates top-3 inference on
random images.

## 3. Sections

| Section | Role |
| --- | --- |
| 1. Project Summary | assignment context, dataset challenge, and experiment stages |
| 2. Runtime, Imports, And Configuration | centralized imports, uppercase `CFG` constants, seed setup, and device selection |
| 3. Data Ingestion And Audit | fixed Kaggle dataset path, manifest creation, class balance, and image-shape sampling |
| 4. Preprocessing, Splits, And Dataloaders | transforms, stratified split, custom `FoodDataset`, and dataloaders |
| 5. Model Construction | reusable builders for GoogLeNet, ResNet50, and MobileNetV3 classifier heads |
| 6. Training And Evaluation Utilities | shared training, validation, checkpointing, and plotting functions |
| 7. Part A: Transfer Learning Comparison | frozen-backbone benchmark and per-class error analysis |
| 8. Part B: ResNet50 Fine-Tuning | selective unfreezing experiments for `layer4` and `layer3 + layer4` |
| 9. Final Model Evaluation And Inference | selected checkpoint loading and top-3 image prediction |

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

These files are Kaggle outputs and should not be committed to git.

## 6. Current Limitations

The current notebook is strong enough for the assignment workflow, but future
iterations could improve:

- test-set evaluation after final model selection;
- confusion matrix and top-k accuracy reporting;
- model size and inference latency comparison;
- history export to CSV or JSON for easier documentation updates.
