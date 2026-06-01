import type { BackendMultiFoodResponse } from "./types";

export const LOCAL_DEMO_RESPONSE: BackendMultiFoodResponse = {
  model: "resnet50_ft_v2",
  temperature: 0.958111,
  top_k: 5,
  decision_thresholds: {
    auto_accept: 0.85,
    suggest: 0.5,
  },
  detector_status: "local_demo",
  crop_count: 3,
  artifact_status: "mock",
  fallback_reason: "frontend_local_demo",
  predictions: [
    {
      source_id: "demo_shared_plate",
      detection_index: 0,
      bbox: {
        x1: 72,
        y1: 54,
        x2: 420,
        y2: 332,
        source_width: 640,
        source_height: 420,
      },
      detector: {
        label: "bowl",
        proposal_role: "serving_container",
        confidence: 0.54,
        crop_area_ratio: 0.36,
      },
      foodlens: {
        top_label: "ravioli",
        top_confidence: 0.972,
        decision_band: "auto_accept",
        top_k_predictions: [
          ["ravioli", 0.972],
          ["gnocchi", 0.018],
          ["lasagna", 0.004],
        ],
      },
      artifacts: {
        crop_path: "local-demo/ravioli.jpg",
        crop_artifact_path: "app://local-demo/ravioli.jpg",
        figure_path: "local-demo/figure.jpg",
      },
    },
    {
      source_id: "demo_shared_plate",
      detection_index: 1,
      bbox: {
        x1: 355,
        y1: 92,
        x2: 596,
        y2: 338,
        source_width: 640,
        source_height: 420,
      },
      detector: {
        label: "cake",
        proposal_role: "direct_food",
        confidence: 0.58,
        crop_area_ratio: 0.22,
      },
      foodlens: {
        top_label: "falafel",
        top_confidence: 0.241,
        decision_band: "confirm",
        top_k_predictions: [
          ["falafel", 0.241],
          ["donuts", 0.195],
          ["garlic_bread", 0.112],
        ],
      },
      artifacts: {
        crop_path: "local-demo/falafel.jpg",
        crop_artifact_path: "app://local-demo/falafel.jpg",
        figure_path: "local-demo/figure.jpg",
      },
    },
    {
      source_id: "demo_shared_plate",
      detection_index: 2,
      bbox: {
        x1: 248,
        y1: 255,
        x2: 526,
        y2: 402,
        source_width: 640,
        source_height: 420,
      },
      detector: {
        label: "bowl",
        proposal_role: "serving_container",
        confidence: 0.44,
        crop_area_ratio: 0.15,
      },
      foodlens: {
        top_label: "ramen",
        top_confidence: 0.768,
        decision_band: "suggest",
        top_k_predictions: [
          ["ramen", 0.768],
          ["pho", 0.034],
          ["miso_soup", 0.023],
        ],
      },
      artifacts: {
        crop_path: "local-demo/ramen.jpg",
        crop_artifact_path: "app://local-demo/ramen.jpg",
        figure_path: "local-demo/figure.jpg",
      },
    },
  ],
};
