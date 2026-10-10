# FoodLens A4 ConvNeXt-Tiny Full Fine-tune (320)

This folder contains the A4 experiment scaffold.

- `kernel-metadata.json`: Kaggle kernel metadata for `tuannm3823`.
- `foodlens_accuracy_phase1_a4.py`: Script-backed run entrypoint.
- `foodlens_accuracy_phase1_a4.ipynb`: Notebook-backed run entrypoint.
- Bootstrap installation in this run is intentionally lightweight: no forced `pip
  install` of torch is performed at startup.

## Intended run
- Run ID: `a4_convnext_tiny_full_finetune_320`
- Input size: `320 x 320`
- Purpose: Isolate resolution gain for ConvNeXt-Tiny while keeping training and
  evaluation protocol identical to A3.

## Notebook run
- Open `foodlens_accuracy_phase1_a4.ipynb` in Kaggle or any local Jupyter
  environment with the same dependencies.
- Run cells top-to-bottom; outputs are written to
  `/kaggle/working/results/accuracy_phase1/a4_convnext_tiny_full_finetune_320/` on
  Kaggle, or `results/accuracy_phase1/a4_convnext_tiny_full_finetune_320/` in local
  runs by default.

## Local run requirements
- Provide a Food-101 folder with class subdirectories as:
  `label_name/*.jpg` via:
  - `FOODLENS_DATA_DIR=/path/to/food-101`.
- Optionally set:
  - `FOODLENS_RESULTS_ROOT=/path/for/run/artifacts`
  - `FOODLENS_CHALLENGER_ARTIFACT_DIR=/path/to/challenger/checkpoints`

## Local validation done
- JSON metadata validates.
- Python source compiles successfully.

## Push to Kaggle
FoodLens kernels are owned by the Kaggle account `tuannm3823`, but the
default `~/.kaggle/kaggle.json` on this machine belongs to `tuannm3812`, so
`KAGGLE_CONFIG_DIR` must be set explicitly on every push -- otherwise the
push silently authenticates as the wrong account. Put the `tuannm3823`
credentials in a durable directory (not `/tmp`, which is cleared on reboot)
as a file named exactly `kaggle.json` (the Kaggle CLI requires that
filename), then point `KAGGLE_CONFIG_DIR` at that directory:

```bash
source .venv/bin/activate
KAGGLE_CONFIG_DIR=/path/to/your/kaggle-credentials kaggle kernels push -p kaggle/accuracy_phase1_a4
```

Do not commit the credentials file itself; only the directory path is
referenced here.

If you receive `Maximum batch GPU session count of 2 reached`, wait until another
GPU session finishes and re-run the same push command.
