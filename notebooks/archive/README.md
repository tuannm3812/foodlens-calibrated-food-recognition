# Archived Notebooks

These notebooks are retained as historical experiment records. They are no
longer part of the active notebook path, but they document the baseline,
intermediate model comparisons, detector exploration, and superseded accuracy
runs that informed the current direction.

| Notebook | Reason archived |
| --- | --- |
| `01_food101_baseline_transfer_finetuning.ipynb` | Superseded by the refined ResNet50 champion and later accuracy experiments. |
| `02_resnet50_training_refinements.ipynb` | Produced the ResNet50 FT-V2 champion, now consumed by downstream calibration and demo notebooks. |
| `03_modern_backbone_comparison.ipynb` | Historical frozen-head backbone comparison before full ConvNeXt fine-tuning. |
| `07_multi_food_detection_exploration.ipynb` | Detector exploration precursor to the current detector-to-classifier pipeline. |
| `09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb` | Non-promotion ResNet50 continuation result. |
| `10_food101_accuracy_phase1_a3_convnext_tiny.ipynb` | Intermediate ConvNeXt result superseded by A3b continuation. |
| `11_food101_accuracy_phase1_a3b_convnext_tiny_continued.ipynb` | Transitioned from frozen-head ConvNeXt to continued full fine-tuning and was superseded by A4 full fine-tune at 320px. |
| `12_food_taxonomy_expansion_audit.ipynb` | Historical taxonomy expansion audit baseline; no longer directly active after v2. |
| `13_expanded_taxonomy_v1_baseline.ipynb` | Version-1 expanded taxonomy baseline before v2 refinements. |
| `14_expanded_taxonomy_v2_finetune.ipynb` | ConvNeXt v2 finetune predecessor to later calibration/decision-layer runs. |
| `15_expanded_taxonomy_v2_decision_layer.ipynb` | Decision-layer tuning stage before confidence-optimized demo inference path. |
