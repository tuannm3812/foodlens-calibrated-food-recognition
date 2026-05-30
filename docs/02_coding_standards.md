# 2. Coding Standards

## 1. Repository Scope

This repository uses a notebook-led experiment workflow. Kaggle notebooks are
the executable source of truth, while `docs/` captures project instructions,
modeling results, and project decisions.

Keep the root small:

- `notebooks/` for Kaggle notebooks.
- `docs/` for standards, instructions, modeling approach, and result summaries.
- `README.md` for the high-level project overview.

Avoid adding local-only folders such as `data/`, `models/`, `outputs/`,
`configs/`, or `scripts/` unless the project moves away from Kaggle execution.
Food-101 data, Kaggle working directories, and model checkpoints should not be
committed.

## 2. Notebook Naming

Use numbered, stable notebook names that match the workflow:

1. `01_food101_baseline_transfer_finetuning.ipynb`
2. `02_resnet50_training_refinements.ipynb`
3. `03_modern_backbone_comparison.ipynb`
4. `04_resnet50_error_calibration_inference.ipynb`
5. `05_confidence_decision_layer.ipynb`
6. `06_food_recognition_demo_inference.ipynb`
7. `07_multi_food_detection_exploration.ipynb`
8. `08_detection_to_foodlens_pipeline.ipynb`

Notebook 1 is the stable baseline and evaluation workflow. Later notebooks
should isolate experiment families, such as ResNet50 recipe changes, modern
backbone comparisons, calibration analysis, product decision layers, or final
demo inference. Detection notebooks should separate detector exploration from
classifier-per-crop integration so results remain attributable.

Use numbered documentation filenames so the reading order is obvious:

1. `01_project_instructions.md`
2. `02_coding_standards.md`
3. `03_modeling_approach.md`
4. `04_model_results.md`
5. `05_next_steps.md`
6. `06_foodlens_app_concept.md`
7. `07_multi_food_detection_plan.md`

## 3. Code Style

Follow PEP 8 for Python code:

- Use 4 spaces for indentation.
- Keep lines to 79 characters or fewer where practical.
- Prefer f-strings, list comprehensions, and small helper functions when they
  improve readability.
- Add type hints for reusable functions when the type is clear.
- Use `UPPER_SNAKE_CASE` for constants, fixed paths, and configuration fields
  inside `CFG`. Use `snake_case` for ordinary variables, function arguments,
  DataFrames, model instances, dataloaders, and metric outputs.
- Group imports in this order:
  1. Standard library
  2. Third-party libraries
  3. Local modules, if the project later adds them
- Separate import groups with a blank line.

Use Google-style docstrings for reusable functions and classes:

```python
def predict_food(
    image_path: str,
    model: torch.nn.Module,
    class_names: list[str],
    device: torch.device,
) -> None:
    """Display top-3 class predictions for one food image.

    Args:
        image_path: Path to a Food-101 image.
        model: Trained classification model.
        class_names: Class names ordered by model index.
        device: PyTorch execution device.
    """
```

Add short inline comments only when they explain why a decision was made.
Avoid comments that restate what the code already says.

## 4. Notebook Style

Each notebook should include:

- a clear title and short purpose statement;
- numbered Markdown sections;
- a configuration section for tunable values such as seed, batch size, image
  size, learning rate, and epochs;
- Kaggle path constants near the top;
- deterministic seed setup for reproducibility;
- Markdown insight cells after important plots or metrics;
- artifact-writing cells for checkpoints, histories, figures, and inference
  outputs.

Prefer readable, self-contained notebook code over imports from local project
modules. Kaggle should be able to run the notebook after attaching the required
dataset.

When notebook code changes, clear outputs before committing. When only Markdown,
documentation, or result commentary changes, existing notebook outputs may be
kept if they are intentionally preserved as evidence. Kaggle remains the trusted
execution record for regenerated outputs.

## 5. Deep Learning Standards

Modeling code should make the experimental contract explicit:

- freeze pretrained layers during transfer-learning comparison;
- replace each classifier with the required 3-layer head;
- use consistent image preprocessing across models;
- record training and validation loss and accuracy per epoch;
- save the best checkpoint per model or fine-tuning experiment;
- keep fine-tuning learning rates lower than classifier-head training rates;
- load saved weights before final error analysis or inference.

Avoid leakage and accidental evaluation drift:

- keep train, validation, and test splits stratified;
- fit label mappings from the training dataset contract and reuse them;
- use validation transforms for validation, test, and inference;
- do not report test performance unless the notebook explicitly evaluates the
  held-out test split.

## 6. Plot Style

Use clear comparison plots for model selection:

- line charts for validation accuracy and loss across epochs;
- compact tables for architecture and result comparisons;
- per-class F1 summaries for error analysis;
- readable titles that state the analytical purpose.

Prefer Viridis or other accessible palettes when adding new charts. Avoid
decorative plots that do not support a decision.

## 7. Documentation Style

Documentation should be written for a future reviewer or teammate who wants the
reasoning quickly:

- use numbered sections;
- lead with findings and implications;
- include exact metrics when available;
- link notebooks and docs with relative paths;
- keep broad narrative in the root `README.md`;
- keep detailed evidence in focused docs.

## 8. Git Hygiene

Do not commit:

- raw Food-101 images or archives;
- Kaggle working directories;
- model checkpoints such as `.pth` files;
- local cache folders;
- notebook checkpoints;
- ad hoc experiment dumps.

Commit lightweight documentation and cleared notebooks only.
