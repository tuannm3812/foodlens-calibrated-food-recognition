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


def build_predictions(raw_predictions: tuple[tuple[str, float], ...]) -> list[Prediction]:
    """Convert raw label-score tuples to API prediction objects."""
    return [
        Prediction(rank=index + 1, class_name=class_name, confidence=confidence)
        for index, (class_name, confidence) in enumerate(raw_predictions)
    ]
