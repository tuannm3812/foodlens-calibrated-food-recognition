# Multi-Class Food Recognition

<img src="https://www.meatdistrictco.com.au/wp-content/uploads/2024/08/0O2A0384-1700x660.jpg" alt="Food recognition project banner" width="100%">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Runtime-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-ResNet50%20Champion-2E7D32?style=flat-square)

Notebook-first Food-101 image classification project. The workflow builds a
defensible baseline, improves the selected ResNet50 model, and checks whether
modern compact backbones can beat the current champion under the same
evaluation protocol.

## Current Result

The strongest evaluated model is **ResNet50 FT-V2**, trained with longer
fine-tuning, AdamW, learning-rate scheduling, stronger augmentation, and label
smoothing.

| Model | Stage | Test top-1 | Test top-5 | Parameters | Model size | T4 latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ResNet50 FT-V2 | current champion | 78.28% | 92.65% | 24.7M | 94.48 MB | 5.35 ms/image |
| ConvNeXt-Tiny | frozen-head challenger | 70.92% | 90.24% | 28.4M | 108.23 MB | 7.17 ms/image |
| EfficientNet-B0 | frozen-head challenger | 52.13% | 77.02% | 4.8M | 18.55 MB | 7.44 ms/image |

The architecture comparison did not produce a replacement for ResNet50 FT-V2.
The next project step is error-driven refinement and confidence calibration.

## Project Overview

Food-101 contains **101,000 RGB food images** across **101 balanced classes**.
The task is challenging because many categories share similar ingredients,
colors, textures, and plating styles. This project focuses on practical model
comparison rather than local infrastructure: training and evaluation run on
Kaggle, while this repository keeps the notebooks and documentation clean.

Expected Kaggle dataset path:

```text
/kaggle/input/datasets/kmader/food41
```

Image directory used by the notebooks:

```text
/kaggle/input/datasets/kmader/food41/images
```

Notebook outputs are written under:

```text
/kaggle/working/results
```

## Repository Structure

```text
.
|-- README.md
|-- docs/
|   |-- 1_instructions.md
|   |-- 2_coding_standards.md
|   |-- 3_notebook_food101_transfer_finetuning.md
|   |-- 4_model_results.md
|   `-- 5_next_steps.md
`-- notebooks/
    |-- 01_food101_baseline_transfer_finetuning.ipynb
    |-- 02_resnet50_training_refinements.ipynb
    `-- 03_modern_backbone_comparison.ipynb
```

## Notebook Workflow

| Notebook | Purpose |
| --- | --- |
| `01_food101_baseline_transfer_finetuning.ipynb` | Builds the Food-101 baseline: data ingestion, transfer-learning comparison, ResNet50 fine-tuning, held-out test evaluation, hard-class confusion analysis, qualitative errors, and efficiency reporting. |
| `02_resnet50_training_refinements.ipynb` | Improves the selected ResNet50 checkpoint with a stronger training recipe and evaluates the final FT-V2 artifact. |
| `03_modern_backbone_comparison.ipynb` | Compares EfficientNet-B0 and ConvNeXt-Tiny against ResNet50 FT-V2 using the same split, metrics, and artifact exports. |

## Modeling Approach

The project progresses through three controlled stages:

1. **Transfer learning baseline:** compare GoogLeNet, ResNet50, and
   MobileNetV3 with frozen ImageNet-pretrained feature extractors.
2. **ResNet50 fine-tuning:** selectively unfreeze deeper ResNet50 blocks and
   establish a stable held-out test baseline.
3. **Training recipe and backbone comparison:** improve the ResNet50 recipe,
   then compare modern backbones against the refined champion.

Best baseline and refinement results:

| Stage | Best result |
| --- | ---: |
| Frozen ResNet50 transfer learning | 59.49% validation top-1 |
| Baseline fine-tuned ResNet50 `layer3 + layer4` | 73.64% test top-1 |
| Refined ResNet50 FT-V2 | 78.28% test top-1 |
| Refined ResNet50 FT-V2 | 92.65% test top-5 |

## Key Findings

- Fine-tuning is more important than swapping architectures at this stage.
- ResNet50 FT-V2 improves held-out test top-1 by **4.63 percentage points**
  over the first fine-tuned ResNet50 baseline.
- ConvNeXt-Tiny is the strongest modern-backbone challenger, but it is still
  behind ResNet50 FT-V2 while also being larger and slower in the current run.
- EfficientNet-B0 is compact, but frozen-head accuracy is too low for the
  current accuracy target.
- Remaining errors are concentrated in visually similar food families such as
  steak-like dishes, tartare or ceviche dishes, pastry-like desserts, and
  chocolate desserts.
- High-confidence wrong predictions suggest calibration should be part of the
  next evaluation layer.

Detailed results: [docs/4_model_results.md](docs/4_model_results.md).

## Kaggle Usage

Open the notebooks on Kaggle and attach the Food-101 dataset. Start with:

```text
notebooks/01_food101_baseline_transfer_finetuning.ipynb
```

Then run:

```text
notebooks/02_resnet50_training_refinements.ipynb
notebooks/03_modern_backbone_comparison.ipynb
```

For faster reruns, upload trained `.pth` checkpoints as Kaggle Model artifacts
and switch notebook modes from training to inference or evaluation.

Important artifact paths used by the current notebooks:

```text
/kaggle/input/models/tuannm3823/food101-baseline-artifacts/pytorch/default/1/food101-baseline-artifacts
/kaggle/input/models/tuannm3823/food101-resnet50-refinements/pytorch/default/1
```

No local dependency setup is required for this repository.

## Documentation

- [Project instructions and approach](docs/1_instructions.md)
- [Coding standards](docs/2_coding_standards.md)
- [Notebook notes](docs/3_notebook_food101_transfer_finetuning.md)
- [Model results](docs/4_model_results.md)
- [Next steps](docs/5_next_steps.md)

Banner image source:
[`meatdistrictco.com.au`](https://www.meatdistrictco.com.au/wp-content/uploads/2024/08/0O2A0384-1700x660.jpg)
