# Documentation Index

This folder keeps the detailed project reasoning, standards, results, and next
steps. The root `README.md` stays focused on the high-level project story.

| File | Purpose |
| --- | --- |
| [`01_project_instructions.md`](01_project_instructions.md) | project objective, dataset scope, evaluation contract, and artifact policy |
| [`02_coding_standards.md`](02_coding_standards.md) | repository, notebook, code, documentation, and git standards |
| [`03_modeling_approach.md`](03_modeling_approach.md) | notebook-by-notebook reasoning flow and modeling decisions |
| [`04_model_results.md`](04_model_results.md) | baseline, refinement, backbone, calibration, and inference results |
| [`05_next_steps.md`](05_next_steps.md) | recommended next work, demo validation, and decision-layer plan |
| [`06_foodlens_app_concept.md`](06_foodlens_app_concept.md) | FoodLens product concept, MVP scope, app architecture, and roadmap |
| [`07_multi_food_detection_plan.md`](07_multi_food_detection_plan.md) | detector-plus-classifier plan for multi-food image and video recognition |
| [`08_model_accuracy_improvement_plan.md`](08_model_accuracy_improvement_plan.md) | phased plan for improving Food-101 accuracy, calibration, and product-level model quality |

Active notebook sequence:

1. `04_resnet50_error_calibration_inference.ipynb`
2. `05_confidence_decision_layer.ipynb`
3. `06_food_recognition_demo_inference.ipynb`
4. `08_detection_to_foodlens_pipeline.ipynb`
5. `11_food101_accuracy_phase1_a3b_convnext_tiny_continued.ipynb`
6. `12_food_taxonomy_expansion_audit.ipynb`
7. `13_expanded_taxonomy_v1_baseline.ipynb`
8. `14_expanded_taxonomy_v2_finetune.ipynb`

Archived notebook records:

- `../notebooks/archive/01_food101_baseline_transfer_finetuning.ipynb`
- `../notebooks/archive/02_resnet50_training_refinements.ipynb`
- `../notebooks/archive/03_modern_backbone_comparison.ipynb`
- `../notebooks/archive/07_multi_food_detection_exploration.ipynb`
- `../notebooks/archive/09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb`
- `../notebooks/archive/10_food101_accuracy_phase1_a3_convnext_tiny.ipynb`

Current accuracy-improvement execution plan:

- start with Food-101-only full fine-tuning experiments;
- treat A1 ResNet50 FT-V3 as a non-promotion result because it stayed just
  below the ResNet50 FT-V2 champion;
- treat A3b ConvNeXt-Tiny as the current accuracy leader, with product promotion
  blocked until calibration and decision-layer behavior are rechecked;
- audit candidate external datasets before training any classifier beyond the
  current 101 classes;
- train the first expanded-taxonomy baseline with a conservative 130-class
  label set, which has now reached **86.10%** test top-1 and **96.88%** test
  top-5 across 130 classes;
- use external food datasets later for pretraining, detector training, or crop
  robustness instead of directly mixing labels into the 101-class classifier.
