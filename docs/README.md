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

Current notebook sequence:

1. `01_food101_baseline_transfer_finetuning.ipynb`
2. `02_resnet50_training_refinements.ipynb`
3. `03_modern_backbone_comparison.ipynb`
4. `04_resnet50_error_calibration_inference.ipynb`
5. `05_confidence_decision_layer.ipynb`
6. `06_food_recognition_demo_inference.ipynb`
7. `07_multi_food_detection_exploration.ipynb`
8. `08_detection_to_foodlens_pipeline.ipynb`
9. `09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb`
10. `10_food101_accuracy_phase1_a3_convnext_tiny.ipynb`
11. `11_food101_accuracy_phase1_a3b_convnext_tiny_continued.ipynb`

Current accuracy-improvement execution plan:

- start with Food-101-only full fine-tuning experiments;
- treat A1 ResNet50 FT-V3 as a non-promotion result because it stayed just
  below the ResNet50 FT-V2 champion;
- treat A3 ConvNeXt-Tiny as the current accuracy leader, with product promotion
  blocked until calibration and decision-layer behavior are rechecked;
- run A3b ConvNeXt-Tiny continuation because A3 validation accuracy was still
  improving at the final configured epoch;
- use external food datasets later for pretraining, detector training, or crop
  robustness instead of directly mixing labels into the 101-class classifier.
