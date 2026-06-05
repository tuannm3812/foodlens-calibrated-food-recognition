# FoodLens A4 ConvNeXt-Tiny Full Fine-tune (320)

This folder contains the A4 experiment scaffold.

- `kernel-metadata.json`: Kaggle kernel metadata for `tuannm3823`.
- `foodlens_accuracy_phase1_a4.py`: Script-backed run entrypoint.

## Intended run
- Run ID: `a4_convnext_tiny_full_finetune_320`
- Input size: `320 x 320`
- Purpose: Isolate resolution gain for ConvNeXt-Tiny while keeping training and
  evaluation protocol identical to A3.

## Local validation done
- JSON metadata validates.
- Python source compiles successfully.

## Push to Kaggle
Use the command below when your account has available GPU batch slots:

```bash
source .venv/bin/activate
KAGGLE_CONFIG_DIR=/tmp/kaggle-cred kaggle kernels push -p kaggle/accuracy_phase1_a4
```

If you receive `Maximum batch GPU session count of 2 reached`, wait until another
GPU session finishes and re-run the same push command.
