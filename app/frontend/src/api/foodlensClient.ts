import { detectorStatusLabel } from "./labels";
import { isBackendMultiFoodResponse, isBackendRuntimeStatus, isRecord, isString } from "./guards";
import { normalizeMultiFoodResponse } from "./normalize";
import type { AnalyzerResult, RuntimeStatusSummary } from "./types";

export { normalizeMultiFoodResponse, toLocalDemoResult, combineFrameResults } from "./normalize";

const API_BASE_URL = "http://127.0.0.1:8000";

export class FoodLensApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "FoodLensApiError";
  }
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
