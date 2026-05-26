# 1. Assignment Instructions And Approach

## 1. Objective

This project builds a 101-class food image classifier for the Food-101 dataset
as part of 94691 Deep Learning Assignment 2 on convolutional neural networks.

The primary objective is to compare transfer learning and fine-tuning
strategies for fine-grained food recognition. The assignment notebook focuses
on:

- dataset ingestion and exploratory data analysis;
- image preprocessing, augmentation, and PyTorch dataloaders;
- transfer learning with three pretrained CNN architectures;
- fine-tuning experiments on the best transfer-learning candidate;
- final inference that returns top-3 class predictions for individual images.

## 2. Dataset

Dataset: Food-101.

Expected Kaggle dataset path:

```text
/kaggle/input/datasets/kmader/food41
```

The notebook reads images from:

```text
/kaggle/input/datasets/kmader/food41/images
```

Dataset characteristics used by the notebook:

| Property | Value |
| --- | ---: |
| Images | 101,000 |
| Classes | 101 |
| Images per class | 1,000 |
| Image type | RGB food photographs |

Food-101 intentionally includes noisy mobile-style imagery, varied lighting,
glare, presentation differences, and occasional label ambiguity. This makes it
a good transfer-learning benchmark rather than a clean toy classification task.

## 3. Modeling Requirements

Part A compares three pretrained CNN backbones as frozen feature extractors:

1. GoogLeNet.
2. ResNet50.
3. MobileNetV3 Large.

Each model receives a 3-layer fully connected classifier head for the 101 food
classes. The pretrained convolutional layers are frozen during this comparison.

Part B selects the strongest Part A candidate and fine-tunes different depths
of the backbone. The current notebook selects ResNet50.

## 4. Evaluation

The notebook uses a stratified split:

| Split | Share |
| --- | ---: |
| Train | 80% |
| Validation | 10% |
| Test | 10% |

The primary metric is top-1 validation accuracy. Error analysis also uses
per-class F1 scores to identify the easiest and hardest food categories.

## 5. Kaggle Execution Contract

Training is expected to run on Kaggle, not locally. The notebook writes
checkpoints and generated artifacts under:

```text
/kaggle/working/results
```

Generated checkpoints should not be committed to this repository. Kaggle is the
trusted execution environment for rerunning the notebook and regenerating
outputs.

## 6. Current Solution

The saved notebook output shows:

- ResNet50 led the frozen transfer-learning comparison.
- Fine-tuning ResNet50 `layer3` and `layer4` produced the best validation
  result.
- The final inference helper loads the fine-tuned checkpoint and reports top-3
  class probabilities for random Food-101 images.
