import type { BackendMultiFoodResponse, DecisionBand } from "./types";

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

function actionCopyForDecisionBand(
  decisionBand: DecisionBand,
  hasStrongestRegion: boolean,
): string {
  if (!hasStrongestRegion) {
    return "No usable crop was returned. Ask the user to try another image.";
  }

  return DECISION_ACTIONS[decisionBand];
}

export {
  DECISION_ACTIONS,
  DECISION_BANDS,
  DETECTOR_STATUS_LABELS,
  FALLBACK_REASON_LABELS,
  DETECTOR_ROLE_LABELS,
  labelFromToken,
  detectorStatusLabel,
  modelNameLabel,
  fallbackReasonLabel,
  artifactStatusLabel,
  detectorLabel,
  detectorRoleLabel,
  regionStatusLabel,
  actionCopyForDecisionBand,
};
