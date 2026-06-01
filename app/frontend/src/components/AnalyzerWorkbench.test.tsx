import { act, render, renderHook, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AnalyzerWorkbench } from "./AnalyzerWorkbench";
import { PreviewStage } from "./PreviewStage";
import { UploadControls } from "./UploadControls";
import type { AnalyzerResult } from "../api/types";
import { fetchRuntimeStatus, predictMultiFoodImage } from "../api/foodlensClient";
import { useAnalyzer } from "../state/useAnalyzer";

vi.mock("../api/foodlensClient", () => ({
  combineFrameResults: (results: AnalyzerResult[]) => results[0] ?? createResult("fallback", 0.1),
  fetchRuntimeStatus: vi.fn(async () => ({
    ready: true,
    title: "System ready",
    classifierLabel: "Classifier ready",
    detectorLabel: "Detector ready",
    modeLabel: "Live detector + classifier",
  })),
  predictMultiFoodImage: vi.fn(),
  toLocalDemoResult: () => createResult("ravioli", 0.972, "local_demo"),
}));

function createResult(
  label: string,
  confidence: number,
  source: AnalyzerResult["source"] = "live",
): AnalyzerResult {
  return {
    modelName: "test-model · Multi-food",
    temperature: 1,
    detectorStatus: source,
    detectorStatusLabel: source === "live" ? "Live detector + classifier" : "Local demo",
    artifactStatus: "mock",
    artifactStatusLabel: "Classifier fallback",
    fallbackReason: source === "local_demo" ? "frontend_local_demo" : undefined,
    fallbackReasonLabel: source === "local_demo" ? "Local demo response" : undefined,
    source,
    strongestLabel: label,
    strongestConfidence: confidence,
    decisionBand: confidence > 0.85 ? "auto_accept" : "suggest",
    actionCopy: "Test action copy.",
    topPredictions: [[label, confidence]],
    regions: [
      {
        source_id: `source-${label}`,
        detection_index: 0,
        displayIndex: 1,
        bbox: {
          x1: 10,
          y1: 10,
          x2: 90,
          y2: 90,
          source_width: 100,
          source_height: 100,
        },
        detector: {
          label: "plate",
          proposal_role: "direct_food",
          confidence: 0.8,
          crop_area_ratio: 0.64,
        },
        foodlens: {
          top_label: label,
          top_confidence: confidence,
          decision_band: confidence > 0.85 ? "auto_accept" : "suggest",
          top_k_predictions: [[label, confidence]],
        },
        artifacts: {
          crop_path: `${label}.jpg`,
          crop_artifact_path: `${label}.jpg`,
          figure_path: `${label}-figure.jpg`,
        },
        regionStatusLabel: "Detector crop",
      },
    ],
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });

  return { promise, resolve, reject };
}

describe("AnalyzerWorkbench", () => {
  beforeEach(() => {
    vi.mocked(fetchRuntimeStatus).mockClear();
    vi.mocked(predictMultiFoodImage).mockReset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders the Precision Lab product shell", async () => {
    render(<AnalyzerWorkbench />);
    const decisionPolicy = screen.getByLabelText("Decision policy");

    expect(screen.getByRole("banner")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "FoodLens home" })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Product navigation" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Analyze" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(screen.getByRole("button", { name: "Review" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Models" })).toBeDisabled();
    expect(screen.getByRole("heading", { name: "Analysis Result" })).toBeInTheDocument();
    expect(screen.getByText("Live API · Image/video upload · Calibrated crop review")).toBeInTheDocument();
    expect(decisionPolicy).toBeInTheDocument();
    expect(within(decisionPolicy).getByText("Auto-accept")).toBeInTheDocument();
    expect(within(decisionPolicy).getByText("Suggest")).toBeInTheDocument();
    expect(within(decisionPolicy).getByText("Confirm")).toBeInTheDocument();
    expect(within(decisionPolicy).getByText("Review")).toBeInTheDocument();
    expect(screen.getByText("No input selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sample" })).toBeInTheDocument();
    expect(await screen.findByText("System ready")).toBeInTheDocument();
  });

  it("renders local demo fallback after selecting sample", async () => {
    const user = userEvent.setup();
    render(<AnalyzerWorkbench />);

    await user.click(screen.getByRole("button", { name: "Sample" }));

    expect(screen.getByText("ravioli")).toBeInTheDocument();
    expect(screen.getAllByText("Local demo").length).toBeGreaterThan(0);
    expect(screen.getByText("Detected regions")).toBeInTheDocument();
  });

  it("keeps the result card hierarchy after loading the sample", async () => {
    const user = userEvent.setup();
    render(<AnalyzerWorkbench />);

    await user.click(screen.getByRole("button", { name: "Sample" }));

    expect(screen.getByLabelText("Decision summary")).toHaveClass("decision-card");
    expect(screen.getAllByText("Local demo").length).toBeGreaterThan(0);
    expect(screen.getByText("ravioli")).toBeInTheDocument();
    expect(screen.getByText("97.2%")).toBeInTheDocument();
    expect(screen.getByText("Detected regions")).toBeInTheDocument();
    expect(screen.getByText("Decision")).toBeInTheDocument();
    expect(screen.getByText("Confidence")).toBeInTheDocument();
    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getByText("Detector")).toBeInTheDocument();
  });

  it("selects crop cards and highlights the matching preview box", async () => {
    const user = userEvent.setup();
    const result = createResult("ravioli", 0.91);
    result.regions = [
      result.regions[0],
      {
        ...result.regions[0],
        source_id: "source-ramen",
        displayIndex: 2,
        detection_index: 1,
        foodlens: {
          ...result.regions[0].foodlens,
          top_label: "ramen",
          top_confidence: 0.61,
        },
        detector: {
          ...result.regions[0].detector,
          label: "bowl",
          proposal_role: "serving_container",
        },
        regionStatusLabel: "Detector crop",
      },
    ];
    vi.mocked(predictMultiFoodImage).mockResolvedValue(result);
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:food-preview"),
      revokeObjectURL: vi.fn(),
    });
    const file = new File(["image"], "food.jpg", { type: "image/jpeg" });

    render(<AnalyzerWorkbench />);
    await user.upload(screen.getByLabelText("Upload"), file);
    await screen.findByText("ravioli");

    await user.click(screen.getByRole("button", { name: /Region 2: ramen/i }));

    expect(screen.getByText("Selected crop")).toBeInTheDocument();
    expect(screen.getByText("Classifier: ramen")).toBeInTheDocument();
    expect(screen.getByText("Detector: bowl")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Detected region 2" })).toHaveClass(
      "bbox-overlay--selected",
    );
  });

  it("keeps stale image analysis from overwriting the current request", async () => {
    const first = deferred<AnalyzerResult>();
    const second = deferred<AnalyzerResult>();
    vi.mocked(predictMultiFoodImage)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const firstFile = new File(["first"], "first.jpg", { type: "image/jpeg" });
    const secondFile = new File(["second"], "second.jpg", { type: "image/jpeg" });
    const { result } = renderHook(() => useAnalyzer());

    await act(async () => {
      void result.current.analyzeImage(firstFile);
    });
    await act(async () => {
      void result.current.analyzeImage(secondFile);
    });

    await act(async () => {
      second.resolve(createResult("second_upload", 0.91));
      await second.promise;
    });
    expect(result.current.result?.strongestLabel).toBe("second_upload");

    await act(async () => {
      first.resolve(createResult("first_upload", 0.99));
      await first.promise;
    });
    expect(result.current.result?.strongestLabel).toBe("second_upload");
  });
});

describe("UploadControls", () => {
  it("routes selected video files to the video upload callback", async () => {
    const user = userEvent.setup();
    const onUploadImage = vi.fn();
    const onVideoSelected = vi.fn();
    const file = new File(["video"], "demo.mp4", { type: "video/mp4" });

    render(
      <UploadControls
        mode="video"
        status="idle"
        onModeChange={vi.fn()}
        onUploadImage={onUploadImage}
        onVideoSelected={onVideoSelected}
        onSample={vi.fn()}
        onClear={vi.fn()}
      />,
    );

    const input = screen.getByLabelText("Upload") as HTMLInputElement;
    expect(input).toHaveAttribute("accept", "video/*");
    expect(input).not.toBeDisabled();

    await user.upload(input, file);

    expect(onVideoSelected).toHaveBeenCalledWith(file);
    expect(onUploadImage).not.toHaveBeenCalled();
  });
});

describe("PreviewStage", () => {
  it("fits tall image overlay planes by height", () => {
    const result = createResult("portrait_source", 0.91);
    result.regions[0].bbox = {
      x1: 20,
      y1: 40,
      x2: 80,
      y2: 160,
      source_width: 100,
      source_height: 200,
    };
    const { container } = render(
      <PreviewStage mode="image" previewUrl="blob:image-preview" result={result} />,
    );

    const layer = container.querySelector(".preview-image-layer") as HTMLElement;
    expect(layer).toHaveStyle({
      aspectRatio: "100 / 200",
      width: "auto",
      height: "100%",
    });
  });

  it("renders video previews with a video element", () => {
    render(
      <PreviewStage mode="video" previewUrl="blob:video-preview" result={null} />,
    );

    const video = screen.getByLabelText("Selected food video") as HTMLVideoElement;
    expect(video.tagName).toBe("VIDEO");
    expect(video).toHaveAttribute("src", "blob:video-preview");
    expect(video).toHaveAttribute("controls");
  });
});
