# Multi-Class Food Recognition

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Runtime-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-Assignment%20Workflow-2E7D32?style=flat-square)

Notebook-first Food-101 image classification project for 94691 Deep Learning
Assignment 2. The workflow compares transfer learning across GoogLeNet,
ResNet50, and MobileNetV3, then fine-tunes the strongest backbone for
101-class food recognition.

## 1. Project Overview

The task is fine-grained visual categorization over the Food-101 dataset:
101,000 RGB food images across 101 balanced classes. The challenge is high
intra-class variance, strong visual similarity between some dishes, and noisy
real-world photography conditions.

This repository is intentionally small because training is run on Kaggle. The
Kaggle notebook is the executable source of truth; docs capture the assignment
structure, experiment decisions, and current model results.

Expected Kaggle dataset path used by the notebook:

```text
/kaggle/input/datasets/kmader/food41
```

The notebook reads images from:

```text
/kaggle/input/datasets/kmader/food41/images
```

Notebook outputs are written under:

```text
/kaggle/working/results
```

## 2. Repository Structure

```text
.
|-- README.md
|-- docs/
|   |-- 1_instructions.md
|   |-- 2_coding_standards.md
|   |-- 3_notebook_food101_transfer_finetuning.md
|   `-- 4_model_results.md
`-- notebooks/
    `-- 1_food101_transfer_finetuning.ipynb
```

## 3. Notebook Workflow

| Notebook | Purpose |
| --- | --- |
| `1_food101_transfer_finetuning.ipynb` | Food-101 EDA, PyTorch dataloaders, transfer learning comparison, ResNet50 fine-tuning, and top-3 image inference. |

## 4. Modeling Approach

Part A benchmarks three ImageNet-pretrained CNN backbones with frozen feature
extractors and a required 3-layer fully connected head:

| Backbone | Role |
| --- | --- |
| GoogLeNet | Multi-scale Inception-style baseline |
| ResNet50 | Deep residual baseline and selected fine-tuning candidate |
| MobileNetV3 Large | Efficient mobile-oriented baseline |

Part B fine-tunes ResNet50 with two unfreezing strategies:

| Experiment | Trainable scope | Best validation accuracy |
| --- | --- | ---: |
| Exp 1 | `layer4` + classifier head | 69.52% |
| Exp 2 | `layer3`, `layer4` + classifier head | 72.75% |

The current champion is **ResNet50 fine-tuned on layer3 + layer4**, reaching
**72.75% validation accuracy** in the saved notebook output.

## 5. Key Findings

- Food-101 is balanced with 101 classes, but visual difficulty varies sharply.
- Frozen ResNet50 was the strongest Part A backbone in the saved run.
- Fine-tuning deeper ResNet50 blocks improved validation accuracy from the
  frozen-feature baseline to above the 70% target.
- Hard classes include visually similar or ambiguous dishes such as `steak`,
  `chocolate_mousse`, `ravioli`, `pork_chop`, and `ceviche`.
- Easier classes include distinctive dishes such as `edamame`, `seaweed_salad`,
  `bibimbap`, `oysters`, and `pho`.

Detailed results: [docs/4_model_results.md](docs/4_model_results.md).

## 6. Kaggle Usage

Open [notebooks/1_food101_transfer_finetuning.ipynb](notebooks/1_food101_transfer_finetuning.ipynb)
on Kaggle and attach the Food-101 dataset. The notebook handles:

1. Dataset manifest creation.
2. Stratified train/validation/test split.
3. ImageNet normalization and augmentation.
4. Transfer learning model construction.
5. Training, validation, checkpointing, and error analysis.
6. Fine-tuning comparison and final top-3 inference.

No local dependency setup is required for this repository.

## 7. Documentation

- [Assignment instructions and approach](docs/1_instructions.md)
- [Coding standards](docs/2_coding_standards.md)
- [Notebook notes](docs/3_notebook_food101_transfer_finetuning.md)
- [Model results](docs/4_model_results.md)
