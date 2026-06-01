export type DecisionBand = "auto_accept" | "suggest" | "confirm" | "review";
export type ResultSource = "live" | "backend_fallback" | "local_demo";

export type BoundingBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  source_width: number;
  source_height: number;
};

export type BackendRegionPrediction = {
  source_id: string;
  detection_index: number;
  bbox?: BoundingBox;
  detector: {
    label: string;
    proposal_role: string;
    confidence: number;
    crop_area_ratio: number;
  };
  foodlens: {
    top_label: string;
    top_confidence: number;
    decision_band: DecisionBand;
    top_k_predictions: Array<[string, number]>;
  };
  artifacts: {
    crop_path: string;
    crop_artifact_path: string;
    figure_path: string;
    crop_data_url?: string | null;
  };
};

export type BackendMultiFoodResponse = {
  model: string;
  temperature: number;
  top_k: number;
  decision_thresholds: Record<string, number>;
  detector_status: string;
  crop_count: number;
  predictions: BackendRegionPrediction[];
  artifact_status: string;
  fallback_reason?: string | null;
};

export type UiRegionPrediction = BackendRegionPrediction & {
  displayIndex: number;
  detectorLabel: string;
  detectorRoleLabel: string;
  regionStatusLabel: string;
};

export type AnalyzerResult = {
  modelName: string;
  temperature: number;
  detectorStatus: string;
  detectorStatusLabel: string;
  artifactStatus: string;
  artifactStatusLabel: string;
  fallbackReason?: string;
  fallbackReasonLabel?: string;
  source: ResultSource;
  strongestLabel: string;
  strongestConfidence: number;
  decisionBand: DecisionBand;
  actionCopy: string;
  topPredictions: Array<[string, number]>;
  regions: UiRegionPrediction[];
};

export type BackendRuntimeStatus = {
  classifier: {
    status: string;
    artifact_status: string;
    artifact_dir: string;
    artifacts: Record<string, unknown>;
  };
  detector: {
    status: string;
    dependency: string;
    dependency_available: boolean;
    weights_path: string;
    weights_found: boolean;
    weights_source: string;
  };
  multi_food: {
    mode: string;
    detector_status: string;
  };
};

export type RuntimeStatusSummary = {
  ready: boolean;
  title: string;
  classifierLabel: string;
  detectorLabel: string;
  modeLabel: string;
};
