import { LOCAL_DEMO_RESPONSE } from "./demoData";
import {
  actionCopyForDecisionBand,
  artifactStatusLabel,
  detectorLabel,
  detectorRoleLabel,
  detectorStatusLabel,
  fallbackReasonLabel,
  modelNameLabel,
  regionStatusLabel,
} from "./labels";
import { isDecisionBand } from "./guards";
import type {
  AnalyzerResult,
  BackendMultiFoodResponse,
  DecisionBand,
  ResultSource,
  UiRegionPrediction,
} from "./types";

function resultSource(response: BackendMultiFoodResponse): ResultSource {
  if (response.detector_status === "local_demo") {
    return "local_demo";
  }

  if (response.fallback_reason || response.detector_status.includes("fallback")) {
    return "backend_fallback";
  }

  return "live";
}

function normalizeMultiFoodResponse(
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

function toLocalDemoResult(): AnalyzerResult {
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

function combineFrameResults(
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

export { resultSource, normalizeMultiFoodResponse, toLocalDemoResult, combineFrameResults };
