import json
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


def ensure_cuda_compatible_torch() -> None:
    """Log runtime context to help diagnose startup issues."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        result = None

    compute_capability = result.stdout.strip().splitlines()[0] if result else ""
    if compute_capability:
        print(f"CUDA compute capability: {compute_capability}")

    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")


ensure_cuda_compatible_torch()


@dataclass(frozen=True)
class CFG:
    """A4: full-backbone ConvNeXt-Tiny fine-tuning with high-resolution input (320x320)."""

    RUN_ID: str = "a4_convnext_tiny_full_finetune_320"
    SEED: int = 42
    NUM_CLASSES: int = 101
    IMAGE_SIZE: tuple[int, int] = (320, 320)
    BATCH_SIZE: int = 20
    NUM_WORKERS: int = 2
    MAX_EPOCHS: int = 8
    PATIENCE: int = 2
    BACKBONE_LR: float = 6e-6
    HEAD_LR: float = 2.5e-5
    WEIGHT_DECAY: float = 5e-2
    LABEL_SMOOTHING: float = 0.1
    TOP_K: int = 5
    ECE_BINS: int = 15
    LATENCY_WARMUP_RUNS: int = 10
    LATENCY_BENCHMARK_RUNS: int = 50
    DATA_DIR: Path = Path("/kaggle/input/datasets/kmader/food41")
    CHALLENGER_ARTIFACT_DIR: Path = Path(
        "/kaggle/input/models/tuannm3823/food101-modern-backbones/"
        "pytorch/default/1"
    )
    FROZEN_HEAD_CHECKPOINT_NAME: str = "convnext_tiny_frozen_head_best.pth"
    RESULTS_ROOT: Path = Path("/kaggle/working/results/accuracy_phase1")
    CHAMPION_TEST_TOP_1: float = 78.28
    CHAMPION_TEST_TOP_5: float = 92.65
    CHAMPION_TEST_ECE: float = 0.0265
    CHAMPION_LATENCY_MS: float = 5.35
    FROZEN_HEAD_TEST_TOP_1: float = 70.92
    FROZEN_HEAD_TEST_TOP_5: float = 90.24


RESULTS_DIR = CFG.RESULTS_ROOT / CFG.RUN_ID
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    """Make the experiment repeatable inside the Kaggle runtime."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(CFG.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Run: {CFG.RUN_ID}")
print(f"Device: {device}")


def resolve_image_dir(data_dir: Path) -> Path:
    """Resolve the Food-101 image directory from a Kaggle dataset mount."""
    candidate_dirs = [data_dir / "images", data_dir]
    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue
        class_dirs = [path for path in candidate_dir.iterdir() if path.is_dir()]
        has_images = any(
            image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for class_dir in class_dirs
            for image_path in class_dir.iterdir()
        )
        if has_images:
            return candidate_dir

    raise FileNotFoundError(
        "Food-101 class image folders were not found under "
        f"{data_dir} or {data_dir / 'images'}."
    )


def create_data_manifest(image_dir: Path) -> pd.DataFrame:
    """Create an image-path and label manifest from class folders."""
    records: list[dict[str, str]] = []
    for class_dir in sorted(path for path in image_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append({"path": str(image_path), "label": class_dir.name})

    if not records:
        raise ValueError(f"No images found under {image_dir}.")
    return pd.DataFrame.from_records(records)


def split_manifest(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the project-standard stratified train, validation, and test split."""
    train_df, temp_df = train_test_split(
        manifest,
        test_size=0.2,
        random_state=CFG.SEED,
        stratify=manifest["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=CFG.SEED,
        stratify=temp_df["label"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


IMAGE_DIR = resolve_image_dir(CFG.DATA_DIR)
manifest_df = create_data_manifest(IMAGE_DIR)
train_df, val_df, test_df = split_manifest(manifest_df)
class_names = sorted(manifest_df["label"].unique())
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

print(f"Food-101 root: {CFG.DATA_DIR}")
print(f"Image directory: {IMAGE_DIR}")
print(f"Images: {len(manifest_df):,}")
print(f"Classes: {len(class_names):,}")
print(f"Train/val/test: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")

(RESULTS_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2))
train_df.to_csv(RESULTS_DIR / "train_manifest.csv", index=False)
val_df.to_csv(RESULTS_DIR / "val_manifest.csv", index=False)
test_df.to_csv(RESULTS_DIR / "test_manifest.csv", index=False)

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.RandomResizedCrop(CFG.IMAGE_SIZE, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
        transforms.RandomErasing(p=0.15),
    ]
)

EVAL_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(CFG.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]
)


class FoodDataset(Dataset):
    """PyTorch dataset wrapper for Food-101."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        class_to_idx: dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return image, label


def create_loader(
    dataframe: pd.DataFrame,
    transform: transforms.Compose,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        FoodDataset(dataframe, class_to_idx, transform),
        batch_size=CFG.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )


train_loader = create_loader(train_df, TRAIN_TRANSFORMS, shuffle=True)
val_loader = create_loader(val_df, EVAL_TRANSFORMS, shuffle=False)
test_loader = create_loader(test_df, EVAL_TRANSFORMS, shuffle=False)


def make_classifier_head(in_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, CFG.NUM_CLASSES),
    )


def build_convnext_tiny(pretrained: bool = False) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = make_classifier_head(in_features)
    return model


def resolve_frozen_head_checkpoint() -> Optional[Path]:
    candidates = [
        CFG.CHALLENGER_ARTIFACT_DIR / CFG.FROZEN_HEAD_CHECKPOINT_NAME,
        CFG.CHALLENGER_ARTIFACT_DIR
        / "backbone_comparison"
        / CFG.FROZEN_HEAD_CHECKPOINT_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if CFG.CHALLENGER_ARTIFACT_DIR.exists():
        matches = sorted(CFG.CHALLENGER_ARTIFACT_DIR.rglob(CFG.FROZEN_HEAD_CHECKPOINT_NAME))
        if matches:
            return matches[0]
    return None


def configure_optimizer(model: nn.Module) -> optim.Optimizer:
    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = True
        if name.startswith("classifier."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    return optim.AdamW(
        [
            {"params": backbone_params, "lr": CFG.BACKBONE_LR},
            {"params": head_params, "lr": CFG.HEAD_LR},
        ],
        weight_decay=CFG.WEIGHT_DECAY,
    )


checkpoint_path = resolve_frozen_head_checkpoint()
if checkpoint_path is not None:
    model = build_convnext_tiny(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded ConvNeXt frozen-head checkpoint: {checkpoint_path}")
else:
    model = build_convnext_tiny(pretrained=True)
    print("Frozen-head checkpoint not found. Starting from ImageNet weights.")

model = model.to(device)
optimizer = configure_optimizer(model)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
criterion = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)
best_checkpoint_path = RESULTS_DIR / "convnext_tiny_full_finetune_best.pth"

print(
    "Trainable parameters:",
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer] = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.set_grad_enabled(is_train):
        progress = tqdm(loader, leave=False, desc="train" if is_train else "eval")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += logits.argmax(dim=1).eq(labels).sum().item()
            total_count += batch_size
            progress.set_postfix(
                loss=total_loss / total_count,
                acc=100.0 * total_correct / total_count,
            )

    return {
        "loss": total_loss / total_count,
        "top_1_accuracy": 100.0 * total_correct / total_count,
    }


history = []
best_val_top_1 = -np.inf
patience_count = 0

for epoch in range(1, CFG.MAX_EPOCHS + 1):
    started_at = time.time()
    train_metrics = run_epoch(model, train_loader, optimizer)
    val_metrics = run_epoch(model, val_loader)
    scheduler.step(val_metrics["top_1_accuracy"])
    elapsed = time.time() - started_at

    row = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_top_1_accuracy": train_metrics["top_1_accuracy"],
        "val_loss": val_metrics["loss"],
        "val_top_1_accuracy": val_metrics["top_1_accuracy"],
        "backbone_lr": optimizer.param_groups[0]["lr"],
        "head_lr": optimizer.param_groups[1]["lr"],
        "elapsed_seconds": elapsed,
    }
    history.append(row)
    pd.DataFrame(history).to_csv(RESULTS_DIR / "training_history.csv", index=False)
    print(json.dumps(row, indent=2))

    if val_metrics["top_1_accuracy"] > best_val_top_1:
        best_val_top_1 = val_metrics["top_1_accuracy"]
        patience_count = 0
        torch.save(model.state_dict(), best_checkpoint_path)
        print(f"Saved new best checkpoint: {best_checkpoint_path}")
    else:
        patience_count += 1
        if patience_count >= CFG.PATIENCE:
            print("Early stopping triggered.")
            break


def load_best_model() -> nn.Module:
    eval_model = build_convnext_tiny(pretrained=False)
    eval_model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    eval_model = eval_model.to(device)
    eval_model.eval()
    return eval_model


def collect_logits_and_predictions(
    model: nn.Module,
    loader: DataLoader,
    split_name: str,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    model.eval()
    logits_parts = []
    labels_parts = []
    rows = []
    manifest = loader.dataset.df
    seen = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"collect {split_name}"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probabilities, CFG.TOP_K, dim=1)

            logits_cpu = logits.cpu()
            labels_cpu = labels.cpu()
            logits_parts.append(logits_cpu)
            labels_parts.append(labels_cpu)

            for row_index, true_idx in enumerate(labels_cpu.tolist()):
                pred_idx = top_indices[row_index, 0].cpu().item()
                manifest_row = manifest.iloc[seen + row_index]
                rows.append(
                    {
                        "path": manifest_row["path"],
                        "true_label": class_names[true_idx],
                        "pred_label": class_names[pred_idx],
                        "confidence": top_probs[row_index, 0].cpu().item(),
                        "is_correct": pred_idx == true_idx,
                        "top_5": "|".join(
                            class_names[idx]
                            for idx in top_indices[row_index].cpu().tolist()
                        ),
                    }
                )
            seen += labels_cpu.size(0)

    return (
        torch.cat(logits_parts),
        torch.cat(labels_parts),
        pd.DataFrame(rows),
    )


def top_k_accuracy(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> float:
    top_indices = torch.topk(probabilities, k, dim=1).indices
    return top_indices.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = CFG.ECE_BINS,
) -> float:
    confidences, predictions = probabilities.max(dim=1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)

    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        proportion = in_bin.float().mean()
        if proportion.item() > 0:
            accuracy = accuracies[in_bin].float().mean()
            confidence = confidences[in_bin].mean()
            ece += torch.abs(confidence - accuracy) * proportion
    return ece.item()


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temperature = torch.ones(1, requires_grad=True)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature.clamp_min(1e-4), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp_min(1e-4).item())


def classification_report_frame(prediction_df: pd.DataFrame) -> pd.DataFrame:
    report = classification_report(
        prediction_df["true_label"],
        prediction_df["pred_label"],
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().loc[class_names].sort_values(
        "f1-score",
        ascending=False,
    )


def confusion_pair_frame(prediction_df: pd.DataFrame) -> pd.DataFrame:
    errors = prediction_df[~prediction_df["is_correct"]]
    if errors.empty:
        return pd.DataFrame(
            columns=["true_label", "pred_label", "count", "mean_confidence"]
        )
    return (
        errors.groupby(["true_label", "pred_label"], as_index=False)
        .agg(count=("path", "size"), mean_confidence=("confidence", "mean"))
        .sort_values(["count", "mean_confidence"], ascending=False)
        .reset_index(drop=True)
    )


def export_split_reports(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prediction_df: pd.DataFrame,
    split_name: str,
    temperature: float,
) -> dict[str, float]:
    raw_probs = F.softmax(logits, dim=1)
    calibrated_probs = F.softmax(logits / temperature, dim=1)
    metrics = {
        "split": split_name,
        "top_1_accuracy": 100.0 * top_k_accuracy(raw_probs, labels, 1),
        "top_5_accuracy": 100.0 * top_k_accuracy(raw_probs, labels, CFG.TOP_K),
        "ece_raw": expected_calibration_error(raw_probs, labels),
        "ece_calibrated": expected_calibration_error(calibrated_probs, labels),
    }

    prediction_df.to_csv(RESULTS_DIR / f"{split_name}_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / f"{split_name}_metrics.csv",
        index=False,
    )
    classification_report_frame(prediction_df).to_csv(
        RESULTS_DIR / f"{split_name}_class_report.csv"
    )
    confusion_pair_frame(prediction_df).to_csv(
        RESULTS_DIR / f"{split_name}_confusion_pairs.csv",
        index=False,
    )
    return metrics


def model_efficiency_summary(model: nn.Module) -> pd.DataFrame:
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    buffer_count = sum(buffer.numel() for buffer in model.buffers())
    model_size_mb = 4 * (parameter_count + buffer_count) / 1024**2
    sample = torch.randn(1, 3, *CFG.IMAGE_SIZE).to(device)

    with torch.no_grad():
        for _ in range(CFG.LATENCY_WARMUP_RUNS):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started_at = time.perf_counter()
        for _ in range(CFG.LATENCY_BENCHMARK_RUNS):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started_at

    return pd.DataFrame(
        [
            {
                "parameters": parameter_count,
                "model_size_mb": model_size_mb,
                "device": str(device),
                "latency_ms_per_image": 1000 * elapsed / CFG.LATENCY_BENCHMARK_RUNS,
            }
        ]
    )


best_model = load_best_model()
val_logits, val_labels, val_predictions = collect_logits_and_predictions(
    best_model,
    val_loader,
    "val",
)
test_logits, test_labels, test_predictions = collect_logits_and_predictions(
    best_model,
    test_loader,
    "test",
)

temperature = fit_temperature(val_logits, val_labels)
calibration = {"temperature": temperature}
(RESULTS_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2))

val_metrics = export_split_reports(
    val_logits,
    val_labels,
    val_predictions,
    "val",
    temperature,
)
test_metrics = export_split_reports(
    test_logits,
    test_labels,
    test_predictions,
    "test",
    temperature,
)
calibration_summary = pd.DataFrame(
    [
        {
            "split": "val",
            "temperature": temperature,
            "ece_raw": val_metrics["ece_raw"],
            "ece_calibrated": val_metrics["ece_calibrated"],
        },
        {
            "split": "test",
            "temperature": temperature,
            "ece_raw": test_metrics["ece_raw"],
            "ece_calibrated": test_metrics["ece_calibrated"],
        },
    ]
)
calibration_summary.to_csv(RESULTS_DIR / "calibration_summary.csv", index=False)

efficiency_df = model_efficiency_summary(best_model)
efficiency_df.to_csv(RESULTS_DIR / "model_efficiency.csv", index=False)

comparison_df = pd.DataFrame(
    [
        {
            "model": "ResNet50 FT-V2 champion",
            "test_top_1_accuracy": CFG.CHAMPION_TEST_TOP_1,
            "test_top_5_accuracy": CFG.CHAMPION_TEST_TOP_5,
            "test_ece_calibrated": CFG.CHAMPION_TEST_ECE,
            "latency_ms_per_image": CFG.CHAMPION_LATENCY_MS,
        },
        {
            "model": "ConvNeXt-Tiny frozen-head challenger",
            "test_top_1_accuracy": CFG.FROZEN_HEAD_TEST_TOP_1,
            "test_top_5_accuracy": CFG.FROZEN_HEAD_TEST_TOP_5,
            "test_ece_calibrated": np.nan,
            "latency_ms_per_image": 7.17,
        },
        {
            "model": "A4 ConvNeXt-Tiny full fine-tune (320)",
            "test_top_1_accuracy": test_metrics["top_1_accuracy"],
            "test_top_5_accuracy": test_metrics["top_5_accuracy"],
            "test_ece_calibrated": test_metrics["ece_calibrated"],
            "latency_ms_per_image": efficiency_df.loc[0, "latency_ms_per_image"],
        },
    ]
)
comparison_df["test_top_1_delta_vs_champion"] = (
    comparison_df["test_top_1_accuracy"] - CFG.CHAMPION_TEST_TOP_1
)
comparison_df.to_csv(RESULTS_DIR / "champion_comparison.csv", index=False)

print("Validation metrics")
print(json.dumps(val_metrics, indent=2))
print("Test metrics")
print(json.dumps(test_metrics, indent=2))
print("Calibration")
print(calibration_summary)
print("Efficiency")
print(efficiency_df)
print("Champion comparison")
print(comparison_df)
print(f"Outputs written to: {RESULTS_DIR}")
