import { DECISION_BANDS } from "./labels";
import type {
  BackendMultiFoodResponse,
  BackendRuntimeStatus,
  DecisionBand,
} from "./types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isDecisionBand(value: unknown): value is DecisionBand {
  return isString(value) && DECISION_BANDS.includes(value as DecisionBand);
}

function isBoundingBox(value: unknown): boolean {
  return (
    isRecord(value) &&
    isNumber(value.x1) &&
    isNumber(value.y1) &&
    isNumber(value.x2) &&
    isNumber(value.y2) &&
    isNumber(value.source_width) &&
    isNumber(value.source_height)
  );
}

function isTopKPrediction(value: unknown): value is [string, number] {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    isString(value[0]) &&
    isNumber(value[1])
  );
}

function isDecisionThresholds(value: unknown): value is Record<string, number> {
  return isRecord(value) && Object.values(value).every(isNumber);
}

function isBackendRegionPrediction(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }

  const detector = value.detector;
  const foodlens = value.foodlens;
  const artifacts = value.artifacts;
  const bbox = value.bbox;

  return (
    isString(value.source_id) &&
    isNumber(value.detection_index) &&
    (bbox === undefined || isBoundingBox(bbox)) &&
    isRecord(detector) &&
    isString(detector.label) &&
    isString(detector.proposal_role) &&
    isNumber(detector.confidence) &&
    isNumber(detector.crop_area_ratio) &&
    isRecord(foodlens) &&
    isString(foodlens.top_label) &&
    isNumber(foodlens.top_confidence) &&
    isDecisionBand(foodlens.decision_band) &&
    Array.isArray(foodlens.top_k_predictions) &&
    foodlens.top_k_predictions.every(isTopKPrediction) &&
    isRecord(artifacts) &&
    isString(artifacts.crop_path) &&
    isString(artifacts.crop_artifact_path) &&
    isString(artifacts.figure_path) &&
    (artifacts.crop_data_url === undefined ||
      artifacts.crop_data_url === null ||
      isString(artifacts.crop_data_url))
  );
}

function isBackendMultiFoodResponse(
  value: unknown,
): value is BackendMultiFoodResponse {
  return (
    isRecord(value) &&
    isString(value.model) &&
    isNumber(value.temperature) &&
    isNumber(value.top_k) &&
    isDecisionThresholds(value.decision_thresholds) &&
    isString(value.detector_status) &&
    isNumber(value.crop_count) &&
    isString(value.artifact_status) &&
    (value.fallback_reason === undefined ||
      value.fallback_reason === null ||
      isString(value.fallback_reason)) &&
    Array.isArray(value.predictions) &&
    value.predictions.every(isBackendRegionPrediction)
  );
}

function isBackendRuntimeStatus(value: unknown): value is BackendRuntimeStatus {
  if (!isRecord(value)) {
    return false;
  }

  const classifier = value.classifier;
  const detector = value.detector;
  const multiFood = value.multi_food;

  return (
    isRecord(classifier) &&
    isString(classifier.status) &&
    isString(classifier.artifact_status) &&
    isString(classifier.artifact_dir) &&
    isRecord(classifier.artifacts) &&
    isRecord(detector) &&
    isString(detector.status) &&
    isString(detector.dependency) &&
    typeof detector.dependency_available === "boolean" &&
    isString(detector.weights_path) &&
    typeof detector.weights_found === "boolean" &&
    isString(detector.weights_source) &&
    isRecord(multiFood) &&
    isString(multiFood.mode) &&
    isString(multiFood.detector_status)
  );
}

export {
  isRecord,
  isNumber,
  isString,
  isDecisionBand,
  isBoundingBox,
  isTopKPrediction,
  isDecisionThresholds,
  isBackendRegionPrediction,
  isBackendMultiFoodResponse,
  isBackendRuntimeStatus,
};
