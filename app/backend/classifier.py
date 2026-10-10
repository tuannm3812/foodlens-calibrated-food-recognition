"""Classifier head construction and prediction-object scoring.

Pure, torch/artifact-shape helpers with no runtime or artifact-directory
dependency. Must not import inference.
"""

from typing import Any

from .schemas import Prediction


def make_classifier_head(torch_nn: Any, in_features: int) -> Any:
    """Create the project-standard Food-101 classifier head."""
    return torch_nn.Sequential(
        torch_nn.Linear(in_features, 512),
        torch_nn.ReLU(),
        torch_nn.Linear(512, 256),
        torch_nn.ReLU(),
        torch_nn.Linear(256, 101),
    )


def build_classifier_model(torchvision_models: Any, torch_nn: Any, architecture: str) -> Any:
    """Build an untrained classifier of a supported architecture with the project head.

    Both architectures swap their final linear layer for the same head, so a
    checkpoint trained by the project loads with no missing or unexpected keys.

    Args:
        torchvision_models: The ``torchvision.models`` module.
        torch_nn: The ``torch.nn`` module.
        architecture: ``"resnet50"`` or ``"convnext_tiny"``.

    Raises:
        ValueError: For any other architecture.
    """
    if architecture == "resnet50":
        model = torchvision_models.resnet50(weights=None)
        model.fc = make_classifier_head(torch_nn, model.fc.in_features)
    elif architecture == "convnext_tiny":
        model = torchvision_models.convnext_tiny(weights=None)
        model.classifier[2] = make_classifier_head(torch_nn, model.classifier[2].in_features)
    else:
        raise ValueError(f"Unsupported classifier architecture: {architecture!r}.")
    return model


def build_predictions(raw_predictions: tuple[tuple[str, float], ...]) -> list[Prediction]:
    """Convert raw label-score tuples to API prediction objects."""
    return [
        Prediction(rank=index + 1, class_name=class_name, confidence=confidence)
        for index, (class_name, confidence) in enumerate(raw_predictions)
    ]
