import { LOCAL_DEMO_RESPONSE } from "./demoData";
import type {
  AnalyzerResult,
  BackendMultiFoodResponse,
  DecisionBand,
  ResultSource,
  UiRegionPrediction,
} from "./types";

const API_BASE_URL = "http://127.0.0.1:8000";

const DECISION_ACTIONS = {
  auto_accept: "Accept the strongest crop label while keeping alternatives available.",
  suggest: "Show ranked suggestions for the user to select.",
  confirm: "Ask the user to confirm before applying a label.",
  review: "Flag this result for extra review because it matches a known risk.",
} as const;

const DECISION_BANDS: DecisionBand[] = [
  "auto_accept",
  "suggest",
  "confirm",
  "review",
];

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

function resultSource(response: BackendMultiFoodResponse): ResultSource {
  if (response.detector_status === "local_demo") {
    return "local_demo";
  }

  if (response.fallback_reason || response.detector_status.includes("fallback")) {
    return "backend_fallback";
  }

  return "live";
}

function actionCopyForDecisionBand(
  decisionBand: DecisionBand,
  hasStrongestRegion: boolean,
): string {
  if (!hasStrongestRegion) {
    return "No usable crop was returned. Ask the user to try another image.";
  }

  return DECISION_ACTIONS[decisionBand];
}

export function normalizeMultiFoodResponse(
  response: BackendMultiFoodResponse,
): AnalyzerResult {
  const regions: UiRegionPrediction[] = response.predictions
    .slice()
    .sort((a, b) => b.foodlens.top_confidence - a.foodlens.top_confidence)
    .map((prediction, index) => ({
      ...prediction,
      displayIndex: index + 1,
    }));
  const strongest = regions[0];
  const decisionBand = strongest
    ? isDecisionBand(strongest.foodlens.decision_band)
      ? strongest.foodlens.decision_band
      : "review"
    : "confirm";

  return {
    modelName: `${response.model} · Multi-food`,
    temperature: response.temperature,
    detectorStatus: response.detector_status,
    artifactStatus: response.artifact_status,
    fallbackReason: response.fallback_reason ?? undefined,
    source: resultSource(response),
    strongestLabel: strongest?.foodlens.top_label ?? "no_detection",
    strongestConfidence: strongest?.foodlens.top_confidence ?? 0,
    decisionBand,
    actionCopy: actionCopyForDecisionBand(decisionBand, Boolean(strongest)),
    topPredictions: strongest?.foodlens.top_k_predictions ?? [["no_detection", 0]],
    regions,
  };
}

export function toLocalDemoResult(): AnalyzerResult {
  return {
    ...normalizeMultiFoodResponse(LOCAL_DEMO_RESPONSE),
    source: "local_demo",
    detectorStatus: "local_demo",
    fallbackReason: "frontend_local_demo",
    actionCopy:
      "Showing local demo data because the API is unavailable or returned an invalid response.",
  };
}

export function combineFrameResults(results: AnalyzerResult[]): AnalyzerResult {
  const first = results[0] ?? toLocalDemoResult();
  const regions = results.flatMap((result, frameIndex) =>
    result.regions.map((region) => ({
      ...region,
      source_id: `video frame ${frameIndex + 1}`,
    })),
  );
  const strongest = regions
    .slice()
    .sort((a, b) => b.foodlens.top_confidence - a.foodlens.top_confidence)[0];
  const decisionBand = strongest?.foodlens.decision_band ?? first.decisionBand;

  return {
    ...first,
    modelName: first.modelName.replace("Multi-food", "Video review"),
    detectorStatus: `${first.detectorStatus} · ${results.length} frames`,
    strongestLabel: strongest?.foodlens.top_label ?? first.strongestLabel,
    strongestConfidence:
      strongest?.foodlens.top_confidence ?? first.strongestConfidence,
    decisionBand,
    actionCopy: actionCopyForDecisionBand(decisionBand, Boolean(strongest)),
    topPredictions: strongest?.foodlens.top_k_predictions ?? first.topPredictions,
    regions: regions.map((region, index) => ({
      ...region,
      displayIndex: index + 1,
    })),
  };
}

export async function predictMultiFoodImage(file: File): Promise<AnalyzerResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/predict/multi-food/image`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`FoodLens API returned ${response.status}`);
  }

  const body = (await response.json()) as unknown;
  if (!isBackendMultiFoodResponse(body)) {
    throw new Error("FoodLens API returned an invalid multi-food response.");
  }

  return normalizeMultiFoodResponse(body);
}
