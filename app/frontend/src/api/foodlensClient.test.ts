import { afterEach, describe, expect, it, vi } from "vitest";

import { LOCAL_DEMO_RESPONSE } from "./demoData";
import {
  combineFrameResults,
  FoodLensApiError,
  fetchRuntimeStatus,
  isUserInputApiError,
  normalizeMultiFoodResponse,
  predictMultiFoodImage,
  predictMultiFoodImageUrl,
  predictMultiFoodYoutubeUrl,
  toLocalDemoResult,
} from "./foodlensClient";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("normalizeMultiFoodResponse", () => {
  it("sorts regions by confidence and labels backend fallback", () => {
    const result = normalizeMultiFoodResponse({
      ...LOCAL_DEMO_RESPONSE,
      detector_status: "fallback_demo",
      fallback_reason: "missing_artifacts",
    });

    expect(result.source).toBe("backend_fallback");
    expect(result.fallbackReason).toBe("missing_artifacts");
    expect(result.strongestLabel).toBe("ravioli");
    expect(result.strongestConfidence).toBe(0.972);
    expect(result.decisionBand).toBe("auto_accept");
    expect(result.regions[0].displayIndex).toBe(1);
  });

  it("adds human-readable live detector and fallback labels", () => {
    const liveResult = normalizeMultiFoodResponse({
      ...LOCAL_DEMO_RESPONSE,
      detector_status: "live_yolo",
      artifact_status: "ready",
      fallback_reason: null,
    });
    const detectorFallbackResult = normalizeMultiFoodResponse({
      ...LOCAL_DEMO_RESPONSE,
      detector_status: "live_yolo_classifier_fallback",
      fallback_reason: "missing_classifier_artifacts",
    });

    expect(liveResult.detectorStatusLabel).toBe("Live detector + classifier");
    expect(liveResult.artifactStatusLabel).toBe("Classifier ready");
    expect(liveResult.fallbackReasonLabel).toBeUndefined();
    expect(detectorFallbackResult.detectorStatusLabel).toBe("Live detector, classifier fallback");
    expect(detectorFallbackResult.fallbackReasonLabel).toBe("Classifier artifacts missing");
  });

  it("labels whole-image fallback regions clearly", () => {
    const result = normalizeMultiFoodResponse({
      ...LOCAL_DEMO_RESPONSE,
      detector_status: "live_yolo_whole_image_fallback",
      fallback_reason: "no_detector_regions",
      predictions: [
        {
          ...LOCAL_DEMO_RESPONSE.predictions[0],
          detector: {
            ...LOCAL_DEMO_RESPONSE.predictions[0].detector,
            label: "whole_image",
            proposal_role: "fallback_region",
          },
        },
      ],
    });

    expect(result.detectorStatusLabel).toBe("Whole image fallback");
    expect(result.fallbackReasonLabel).toBe("No detector regions");
    expect(result.regions[0].regionStatusLabel).toBe("Whole image fallback");
  });

  it("maps detector proposal roles to human-readable labels", () => {
    const result = normalizeMultiFoodResponse({
      ...LOCAL_DEMO_RESPONSE,
      predictions: [
        {
          ...LOCAL_DEMO_RESPONSE.predictions[0],
          detector: {
            ...LOCAL_DEMO_RESPONSE.predictions[0].detector,
            proposal_role: "fallback_region",
          },
        },
      ],
    });

    expect(result.regions[0].detectorRoleLabel).toBe("Whole image review");
    expect(result.regions[0].detectorRoleLabel).not.toBe("fallback_region");
  });

  it("returns a local demo result when the frontend fallback is used", () => {
    const result = toLocalDemoResult();

    expect(result.source).toBe("local_demo");
    expect(result.detectorStatus).toBe("local_demo");
    expect(result.actionCopy).toContain("local demo");
  });

  it("returns no_detection defaults when there are no predictions", () => {
    const result = normalizeMultiFoodResponse({
      ...LOCAL_DEMO_RESPONSE,
      predictions: [],
    });

    expect(result.strongestLabel).toBe("no_detection");
    expect(result.strongestConfidence).toBe(0);
    expect(result.decisionBand).toBe("confirm");
    expect(result.topPredictions).toEqual([["no_detection", 0]]);
  });

  it("does not mutate the original prediction order", () => {
    const response = {
      ...LOCAL_DEMO_RESPONSE,
      predictions: [
        LOCAL_DEMO_RESPONSE.predictions[1],
        LOCAL_DEMO_RESPONSE.predictions[2],
        LOCAL_DEMO_RESPONSE.predictions[0],
      ],
    };

    normalizeMultiFoodResponse(response);

    expect(response.predictions.map((prediction) => prediction.foodlens.top_label)).toEqual([
      "falafel",
      "ramen",
      "ravioli",
    ]);
  });

  it("uses review action copy for an unknown decision band passed directly to normalization", () => {
    const result = normalizeMultiFoodResponse({
      ...LOCAL_DEMO_RESPONSE,
      predictions: [
        {
          ...LOCAL_DEMO_RESPONSE.predictions[0],
          foodlens: {
            ...LOCAL_DEMO_RESPONSE.predictions[0].foodlens,
            decision_band: "unknown" as never,
          },
        },
      ],
    });

    expect(result.decisionBand).toBe("review");
    expect(result.actionCopy).toContain("extra review");
  });

  it("keeps backend video review model labels intact", () => {
    const result = normalizeMultiFoodResponse({
      ...LOCAL_DEMO_RESPONSE,
      model: "resnet50_ft_v2 · Video review",
      detector_status: "fallback_demo · 2 frames",
    });

    expect(result.modelName).toBe("resnet50_ft_v2 · Video review");
    expect(result.detectorStatus).toBe("fallback_demo · 2 frames");
  });
});

describe("combineFrameResults", () => {
  it("combines frame regions and summarizes the strongest crop", () => {
    const first = normalizeMultiFoodResponse({
      ...LOCAL_DEMO_RESPONSE,
      detector_status: "live_yolo",
      predictions: [LOCAL_DEMO_RESPONSE.predictions[1]],
    });
    const second = normalizeMultiFoodResponse({
      ...LOCAL_DEMO_RESPONSE,
      detector_status: "live_yolo",
      predictions: [
        {
          ...LOCAL_DEMO_RESPONSE.predictions[0],
          foodlens: {
            ...LOCAL_DEMO_RESPONSE.predictions[0].foodlens,
            top_confidence: 0.991,
            top_k_predictions: [["ravioli", 0.991]],
          },
        },
      ],
    });

    const result = combineFrameResults([first, second]);

    expect(result.modelName).toBe("resnet50_ft_v2 · Video review");
    expect(result.detectorStatus).toBe("live_yolo · 2 frames");
    expect(result.strongestLabel).toBe("ravioli");
    expect(result.strongestConfidence).toBe(0.991);
    expect(result.topPredictions).toEqual([["ravioli", 0.991]]);
    expect(result.regions.map((region) => region.source_id)).toEqual([
      "video frame 1",
      "video frame 2",
    ]);
    expect(result.regions.map((region) => region.displayIndex)).toEqual([1, 2]);
  });

  it("falls back to local demo metadata when no frame results are provided", () => {
    const result = combineFrameResults([]);

    expect(result.source).toBe("local_demo");
    expect(result.detectorStatus).toBe("local_demo · 0 frames");
    expect(result.regions).toEqual([]);
  });

  it("keeps video summaries in confirm even when the strongest frame is auto accepted", () => {
    const first = normalizeMultiFoodResponse({
      ...LOCAL_DEMO_RESPONSE,
      predictions: [
        {
          ...LOCAL_DEMO_RESPONSE.predictions[1],
          foodlens: {
            ...LOCAL_DEMO_RESPONSE.predictions[1].foodlens,
            top_confidence: 0.42,
            decision_band: "confirm",
          },
        },
      ],
    });
    const second = normalizeMultiFoodResponse({
      ...LOCAL_DEMO_RESPONSE,
      predictions: [
        {
          ...LOCAL_DEMO_RESPONSE.predictions[0],
          foodlens: {
            ...LOCAL_DEMO_RESPONSE.predictions[0].foodlens,
            top_confidence: 0.98,
            decision_band: "auto_accept",
          },
        },
      ],
    });

    const result = combineFrameResults([first, second]);

    expect(result.decisionBand).toBe("confirm");
    expect(result.actionCopy).toContain("confirm before applying");
    expect(result.actionCopy).not.toContain("Accept the strongest crop label");
  });
});

describe("predictMultiFoodImage", () => {
  it("rejects a payload missing required root fields", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => {
          const { top_k: _topK, ...payload } = LOCAL_DEMO_RESPONSE;

          return payload;
        },
      })),
    );

    const file = new File(["demo"], "demo.jpg", { type: "image/jpeg" });

    await expect(predictMultiFoodImage(file)).rejects.toThrow(
      "FoodLens API returned an invalid multi-food response.",
    );
  });

  it("rejects an invalid nested prediction payload", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          ...LOCAL_DEMO_RESPONSE,
          predictions: [
            {
              ...LOCAL_DEMO_RESPONSE.predictions[0],
              foodlens: {
                ...LOCAL_DEMO_RESPONSE.predictions[0].foodlens,
                decision_band: "unknown",
              },
            },
          ],
        }),
      })),
    );

    const file = new File(["demo"], "demo.jpg", { type: "image/jpeg" });

    await expect(predictMultiFoodImage(file)).rejects.toThrow(
      "FoodLens API returned an invalid multi-food response.",
    );
  });
});

describe("URL prediction clients", () => {
  it("posts direct image URLs to the image URL endpoint", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => LOCAL_DEMO_RESPONSE,
    }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await predictMultiFoodImageUrl("https://example.com/plate.jpg");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/predict/multi-food/image-url",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: "https://example.com/plate.jpg" }),
      },
    );
    expect(result.strongestLabel).toBe("ravioli");
  });

  it("posts YouTube URLs to the YouTube URL endpoint", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        ...LOCAL_DEMO_RESPONSE,
        model: "resnet50_ft_v2 · Video review",
      }),
    }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await predictMultiFoodYoutubeUrl(
      "https://www.youtube.com/watch?v=abc123",
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/predict/multi-food/youtube-url",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: "https://www.youtube.com/watch?v=abc123" }),
      },
    );
    expect(result.modelName).toBe("resnet50_ft_v2 · Video review");
  });

  it("classifies 4xx URL API errors as user input errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 400,
        json: async () => ({ detail: "Enter a public media URL." }),
      })),
    );

    await expect(predictMultiFoodImageUrl("http://127.0.0.1/plate.jpg")).rejects.toThrow(
      "Enter a public media URL.",
    );

    try {
      await predictMultiFoodImageUrl("http://127.0.0.1/plate.jpg");
    } catch (error) {
      expect(error).toBeInstanceOf(FoodLensApiError);
      expect(isUserInputApiError(error)).toBe(true);
    }
  });
});

describe("fetchRuntimeStatus", () => {
  it("summarizes ready classifier and detector runtime status", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          classifier: {
            status: "ready",
            artifact_status: "ready",
            artifact_dir: "/repo/app/artifacts",
            artifacts: {},
          },
          detector: {
            status: "ready",
            dependency: "ultralytics",
            dependency_available: true,
            weights_path: "/repo/yolo11n.pt",
            weights_found: true,
            weights_source: "auto_discovered",
          },
          multi_food: {
            mode: "live_yolo_classifier",
            detector_status: "live_yolo",
          },
        }),
      })),
    );

    const result = await fetchRuntimeStatus();

    expect(result.ready).toBe(true);
    expect(result.title).toBe("System ready");
    expect(result.classifierLabel).toBe("Classifier ready");
    expect(result.detectorLabel).toBe("Detector ready");
    expect(result.modeLabel).toBe("Live detector + classifier");
  });
});
