"""Inference helpers for the FoodLens API."""

from io import BytesIO
import json
from pathlib import Path
from typing import Any, Optional

from .schemas import Decision, Prediction, PredictionResponse


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_NAME = "resnet50_ft_v2"
TEMPERATURE = 0.958111
IMAGE_SIZE = (224, 224)
DEFAULT_POLICY = {
    "auto_confidence": 0.70,
    "suggest_confidence": 0.35,
    "margin_threshold": 0.40,
}
DEFAULT_HARD_CLASSES = {
    "chocolate_mousse",
    "steak",
    "pork_chop",
    "bread_pudding",
    "tuna_tartare",
}

MOCK_IMAGE_PREDICTIONS: tuple[tuple[str, float], ...] = (
    ("steak", 0.7838),
    ("filet_mignon", 0.1543),
    ("prime_rib", 0.0223),
    ("baby_back_ribs", 0.0102),
    ("pork_chop", 0.0092),
)

MOCK_VIDEO_PREDICTIONS: tuple[tuple[str, float], ...] = (
    ("sushi", 0.6842),
    ("sashimi", 0.2015),
    ("ceviche", 0.0511),
    ("tuna_tartare", 0.0394),
    ("miso_soup", 0.0128),
)

_RUNTIME: Optional[dict[str, Any]] = None


def artifact_status() -> str:
    """Return whether real model artifacts are currently available."""
    checkpoint_path = ARTIFACT_DIR / "resnet50_ft_v2_best.pth"
    class_names_path = ARTIFACT_DIR / "class_names.json"
    return "ready" if checkpoint_path.exists() and class_names_path.exists() else "mock"


def read_json(path: Path, default: Any) -> Any:
    """Read a JSON artifact when available."""
    if not path.exists():
        return default
    return json.loads(path.read_text())


def read_temperature() -> float:
    """Read the calibrated temperature artifact when available."""
    calibration = read_json(ARTIFACT_DIR / "calibration.json", {})
    return float(calibration.get("temperature", TEMPERATURE))


def read_policy() -> dict[str, float]:
    """Read decision thresholds when available."""
    policy = read_json(ARTIFACT_DIR / "decision_policy.json", DEFAULT_POLICY)
    return {
        "auto_confidence": float(policy.get("auto_confidence", DEFAULT_POLICY["auto_confidence"])),
        "suggest_confidence": float(
            policy.get("suggest_confidence", DEFAULT_POLICY["suggest_confidence"])
        ),
        "margin_threshold": float(policy.get("margin_threshold", DEFAULT_POLICY["margin_threshold"])),
    }


def read_hard_classes() -> set[str]:
    """Read hard classes when available."""
    hard_classes = read_json(ARTIFACT_DIR / "hard_classes.json", list(DEFAULT_HARD_CLASSES))
    return set(hard_classes)


def read_confusion_pairs() -> set[tuple[str, str]]:
    """Read known confusion pairs when available."""
    raw_pairs = read_json(ARTIFACT_DIR / "confusion_pairs.json", [])
    return {
        (str(pair[0]), str(pair[1]))
        for pair in raw_pairs
        if isinstance(pair, (list, tuple)) and len(pair) >= 2
    }


def make_classifier_head(torch_nn: Any, in_features: int) -> Any:
    """Create the project-standard Food-101 classifier head."""
    return torch_nn.Sequential(
        torch_nn.Linear(in_features, 512),
        torch_nn.ReLU(),
        torch_nn.Linear(512, 256),
        torch_nn.ReLU(),
        torch_nn.Linear(256, 101),
    )


def load_runtime() -> dict[str, Any]:
    """Load model and metadata once when real artifacts are present."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    if artifact_status() != "ready":
        raise FileNotFoundError(
            "Missing real inference artifacts. Expected resnet50_ft_v2_best.pth "
            "and class_names.json under app/artifacts."
        )

    try:
        import torch
        import torch.nn.functional as functional
        from PIL import Image
        from torch import nn
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Real inference requires torch, torchvision, and Pillow."
        ) from exc

    class_names = read_json(ARTIFACT_DIR / "class_names.json", [])
    if len(class_names) != 101:
        raise ValueError("class_names.json must contain 101 ordered Food-101 class names.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = make_classifier_head(nn, model.fc.in_features)
    model.load_state_dict(
        torch.load(ARTIFACT_DIR / "resnet50_ft_v2_best.pth", map_location=device)
    )
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    _RUNTIME = {
        "torch": torch,
        "functional": functional,
        "image_class": Image,
        "device": device,
        "model": model,
        "transform": transform,
        "class_names": class_names,
        "temperature": read_temperature(),
        "policy": read_policy(),
        "hard_classes": read_hard_classes(),
        "confusion_pairs": read_confusion_pairs(),
    }
    return _RUNTIME


def build_predictions(raw_predictions: tuple[tuple[str, float], ...]) -> list[Prediction]:
    """Convert raw label-score tuples to API prediction objects."""
    return [
        Prediction(rank=index + 1, class_name=class_name, confidence=confidence)
        for index, (class_name, confidence) in enumerate(raw_predictions)
    ]


def build_decision(
    mode: str,
    predictions: list[Prediction],
    policy: Optional[dict[str, float]] = None,
    hard_classes: Optional[set[str]] = None,
    confusion_pairs: Optional[set[tuple[str, str]]] = None,
) -> Decision:
    """Build a Notebook 6-style decision output."""
    policy = policy or DEFAULT_POLICY
    hard_classes = hard_classes or DEFAULT_HARD_CLASSES
    confusion_pairs = confusion_pairs or set()
    top_1 = predictions[0]
    top_2 = predictions[1]
    margin = top_1.confidence - top_2.confidence
    predicted_label = top_1.class_name

    risky_prediction = any(predicted_label in pair for pair in confusion_pairs)

    if mode == "video":
        return Decision(
            band="confirm",
            title="Confirm dish",
            recommended_action=(
                "Ask the user to confirm because sampled frames are not fully aligned."
            ),
            top_1_top_2_margin=margin,
        )

    if risky_prediction and margin < policy["margin_threshold"]:
        return Decision(
            band="review",
            title="Review prediction",
            recommended_action="Flag for review because this matches a known confusion risk.",
            top_1_top_2_margin=margin,
        )
    if predicted_label in hard_classes and top_1.confidence < policy["auto_confidence"]:
        return Decision(
            band="confirm",
            title="Confirm dish",
            recommended_action="Ask the user to confirm because this is a hard predicted class.",
            top_1_top_2_margin=margin,
        )
    if (
        top_1.confidence >= policy["auto_confidence"]
        and margin >= policy["margin_threshold"]
        and predicted_label not in hard_classes
    ):
        return Decision(
            band="auto_accept",
            title="Auto-accept",
            recommended_action="Accept the top prediction automatically.",
            top_1_top_2_margin=margin,
        )
    if top_1.confidence >= policy["suggest_confidence"]:
        return Decision(
            band="suggest",
            title="Show suggestions",
            recommended_action="Show ranked suggestions for user selection.",
            top_1_top_2_margin=margin,
        )
    return Decision(
        band="confirm",
        title="Confirm dish",
        recommended_action="Ask the user to confirm before applying a label.",
        top_1_top_2_margin=margin,
    )


def predict_mock(mode: str = "image") -> PredictionResponse:
    """Return a deterministic mock prediction response."""
    raw_predictions = MOCK_VIDEO_PREDICTIONS if mode == "video" else MOCK_IMAGE_PREDICTIONS
    predictions = build_predictions(raw_predictions)
    return PredictionResponse(
        model_name=MODEL_NAME,
        mode=mode,
        temperature=TEMPERATURE,
        top_predictions=predictions,
        decision=build_decision(mode, predictions),
        artifact_status=artifact_status(),
    )


def predict_image_bytes(image_bytes: bytes) -> PredictionResponse:
    """Predict Food-101 classes using real artifacts when available."""
    if artifact_status() != "ready":
        return predict_mock(mode="image")

    try:
        runtime = load_runtime()
        torch = runtime["torch"]
        functional = runtime["functional"]
        image = runtime["image_class"].open(BytesIO(image_bytes)).convert("RGB")
        image_tensor = runtime["transform"](image).unsqueeze(0).to(runtime["device"])

        with torch.no_grad():
            logits = runtime["model"](image_tensor).cpu()
            probabilities = functional.softmax(logits / runtime["temperature"], dim=1)
            top_probabilities, top_indices = probabilities.topk(5, dim=1)

        predictions = [
            Prediction(
                rank=rank + 1,
                class_name=runtime["class_names"][class_index],
                confidence=confidence,
            )
            for rank, (class_index, confidence) in enumerate(
                zip(top_indices[0].tolist(), top_probabilities[0].tolist())
            )
        ]
        return PredictionResponse(
            model_name=MODEL_NAME,
            mode="image",
            temperature=runtime["temperature"],
            top_predictions=predictions,
            decision=build_decision(
                "image",
                predictions,
                policy=runtime["policy"],
                hard_classes=runtime["hard_classes"],
                confusion_pairs=runtime["confusion_pairs"],
            ),
            artifact_status="ready",
        )
    except Exception:
        return predict_mock(mode="image")
