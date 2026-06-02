import { LOCAL_DEMO_RESPONSE } from "./demoData";
import type {
  AnalyzerResult,
  BackendRuntimeStatus,
  BackendMultiFoodResponse,
  DecisionBand,
  ResultSource,
  RuntimeStatusSummary,
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

const DETECTOR_STATUS_LABELS: Record<string, string> = {
  live_yolo: "Live detector + classifier",
  live_yolo_classifier_fallback: "Live detector, classifier fallback",
  live_yolo_whole_image_fallback: "Whole image fallback",
  fallback_demo: "Backend demo fallback",
  local_demo: "Local demo",
};

const FALLBACK_REASON_LABELS: Record<string, string> = {
  classifier_load_error: "Classifier load error",
  classifier_inference_error: "Classifier inference error",
  detector_inference_error: "Detector inference error",
  detector_runtime_unavailable: "Detector runtime unavailable",
  frontend_local_demo: "Local demo response",
  inference_error: "Inference error",
  invalid_image: "Invalid image",
  missing_artifacts: "Classifier artifacts missing",
  missing_classifier_artifacts: "Classifier artifacts missing",
  no_detector_regions: "No detector regions",
  video_mock: "Video mock response",
};

const DETECTOR_ROLE_LABELS: Record<string, string> = {
  direct_food: "Food region",
  fallback_region: "Whole image review",
  serving_container: "Serving area",
};

export class FoodLensApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "FoodLensApiError";
  }
}

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

function labelFromToken(token: string): string {
  return token
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function detectorStatusLabel(detectorStatus: string): string {
  return DETECTOR_STATUS_LABELS[detectorStatus] ?? labelFromToken(detectorStatus);
}

function modelNameLabel(model: string): string {
  if (model.includes("Video review")) {
    return model;
  }

  return `${model} · Multi-food`;
}

function fallbackReasonLabel(fallbackReason?: string | null): string | undefined {
  if (!fallbackReason) {
    return undefined;
  }

  return FALLBACK_REASON_LABELS[fallbackReason] ?? labelFromToken(fallbackReason);
}

function artifactStatusLabel(artifactStatus: string): string {
  if (artifactStatus === "ready") {
    return "Classifier ready";
  }

  if (artifactStatus === "mock") {
    return "Classifier fallback";
  }

  return labelFromToken(artifactStatus);
}

function detectorLabel(label: string): string {
  if (label === "whole_image") {
    return "Whole image";
  }

  return label;
}

function detectorRoleLabel(proposalRole: string): string {
  return DETECTOR_ROLE_LABELS[proposalRole] ?? labelFromToken(proposalRole);
}

function regionStatusLabel(region: BackendMultiFoodResponse["predictions"][number]): string {
  if (region.detector.label === "whole_image" || region.detector.proposal_role === "fallback_region") {
    return "Whole image fallback";
  }

  return "Detector crop";
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
      detectorLabel: detectorLabel(prediction.detector.label),
      detectorRoleLabel: detectorRoleLabel(prediction.detector.proposal_role),
      regionStatusLabel: regionStatusLabel(prediction),
    }));
  const strongest = regions[0];
  const decisionBand = strongest
    ? isDecisionBand(strongest.foodlens.decision_band)
      ? strongest.foodlens.decision_band
      : "review"
    : "confirm";

  return {
    modelName: modelNameLabel(response.model),
    temperature: response.temperature,
    detectorStatus: response.detector_status,
    detectorStatusLabel: detectorStatusLabel(response.detector_status),
    artifactStatus: response.artifact_status,
    artifactStatusLabel: artifactStatusLabel(response.artifact_status),
    fallbackReason: response.fallback_reason ?? undefined,
    fallbackReasonLabel: fallbackReasonLabel(response.fallback_reason),
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
    detectorStatusLabel: "Local demo",
    fallbackReason: "frontend_local_demo",
    fallbackReasonLabel: "Local demo response",
    actionCopy:
      "Showing local demo data because the API is unavailable or returned an invalid response.",
  };
}

export function combineFrameResults(
  results: AnalyzerResult[],
  sampleTimes: number[] = [],
): AnalyzerResult {
  const first = results[0] ?? toLocalDemoResult();
  if (results.length === 0) {
    return {
      ...first,
      modelName: first.modelName.replace("Multi-food", "Video review"),
      detectorStatus: `${first.detectorStatus} · 0 frames`,
      regions: [],
    };
  }

  const regions = results.flatMap((result, frameIndex) =>
    result.regions.map((region) => ({
      ...region,
      source_id: `video frame ${frameIndex + 1}`,
      sourceTimeSeconds: sampleTimes[frameIndex],
    })),
  );
  const strongest = regions
    .slice()
    .sort((a, b) => b.foodlens.top_confidence - a.foodlens.top_confidence)[0];
  const decisionBand: DecisionBand = "confirm";
  const fallbackFrameCount = results.filter(
    (result) =>
      result.source === "backend_fallback" ||
      Boolean(result.fallbackReason) ||
      result.detectorStatus.includes("fallback"),
  ).length;
  const detectorStatusLabel =
    fallbackFrameCount > 0
      ? `${results.length} frames · ${regions.length} regions · ${fallbackFrameCount} fallback ${
          fallbackFrameCount === 1 ? "frame" : "frames"
        }`
      : `${results.length} frames · ${regions.length} regions`;

  return {
    ...first,
    modelName: first.modelName.replace("Multi-food", "Video review"),
    detectorStatus: `video_review · ${results.length} frames`,
    detectorStatusLabel,
    fallbackReason: undefined,
    fallbackReasonLabel: undefined,
    source: "video_review",
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

async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as unknown;
    if (isRecord(body) && isString(body.detail)) {
      return body.detail;
    }
  } catch {
    return `FoodLens API returned ${response.status}`;
  }

  return `FoodLens API returned ${response.status}`;
}

async function postUrlPrediction(
  endpoint: string,
  url: string,
): Promise<AnalyzerResult> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });

  if (!response.ok) {
    throw new FoodLensApiError(await parseErrorMessage(response), response.status);
  }

  const body = (await response.json()) as unknown;
  if (!isBackendMultiFoodResponse(body)) {
    throw new Error("FoodLens API returned an invalid multi-food response.");
  }

  return normalizeMultiFoodResponse(body);
}

export function isUserInputApiError(error: unknown): error is FoodLensApiError {
  return (
    error instanceof FoodLensApiError &&
    error.status >= 400 &&
    error.status < 500
  );
}

export function predictMultiFoodImageUrl(url: string): Promise<AnalyzerResult> {
  return postUrlPrediction("/predict/multi-food/image-url", url);
}

export function predictMultiFoodYoutubeUrl(url: string): Promise<AnalyzerResult> {
  return postUrlPrediction("/predict/multi-food/youtube-url", url);
}

export async function fetchRuntimeStatus(): Promise<RuntimeStatusSummary> {
  const response = await fetch(`${API_BASE_URL}/runtime/status`);
  if (!response.ok) {
    throw new Error(`FoodLens runtime status returned ${response.status}`);
  }

  const body = (await response.json()) as unknown;
  if (!isBackendRuntimeStatus(body)) {
    throw new Error("FoodLens API returned an invalid runtime status.");
  }

  const classifierReady = body.classifier.status === "ready";
  const detectorReady = body.detector.status === "ready";
  const ready = classifierReady && detectorReady;

  return {
    ready,
    title: ready ? "System ready" : "System degraded",
    classifierLabel: classifierReady ? "Classifier ready" : "Classifier missing",
    detectorLabel: detectorReady ? "Detector ready" : "Detector missing",
    modeLabel: detectorStatusLabel(body.multi_food.detector_status),
  };
}
